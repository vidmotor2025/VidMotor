import argparse
import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import importlib
import pickle as pkl
import warnings
import re
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from result_writer import TimeResultWriter
from SOTA.ProtoGCN.loss_utils import loss_graph


def seed_torch(seed):
    """Ensure reproducible results across runs by fixing all random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    """Define and parse command-line arguments for ProtoGCN."""
    parser = argparse.ArgumentParser(description='ProtoGCN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--work-dir', default='./results/temp', help='working directory for storing results')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (in epochs)')
    parser.add_argument('--save-path', default='./results/reproduce_results_time.xlsx', help='path to save results')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader to use')
    parser.add_argument('--num-worker', type=int, default=0, help='number of workers for the data loader')
    parser.add_argument('--test-feeder-args', default=dict(), help='arguments for the validation data loader')
    parser.add_argument('--data-centered', type=int, default=7,
                        help='the indices of key points used for centralization')
    parser.add_argument('--data-dim', type=str, default='3D', help='dimension of the skeleton sequence')
    parser.add_argument('--data-norm', default='zscore', help='normalization method for the skeleton sequence')
    parser.add_argument('--label-type', default='class_label', help='type of label (class or score)')
    parser.add_argument('--score-norm', type=str2bool, default=False,
                        help='if true, score labels will be normalized to the range 0-1')
    parser.add_argument('--model', type=str, default='None', help='model to use')
    parser.add_argument('--model-args', default=dict(), help='arguments for the model')
    parser.add_argument('--loss-weight', default=0.2, help='weights for contrastive loss')
    parser.add_argument('--train-batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=4, help='test batch size')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='indices of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--base-lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--num-epoch', type=int, default=1, help='number of epochs to train')
    return parser


class Processor():
    """Main running controller that manages model, optimizer, and running process."""

    def __init__(self, arg):
        self.arg = arg

    def load_model(self, ce_weights=None):
        """Dynamically import and initialize model architecture."""
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(output_device)

        # Define classification loss
        self.loss = nn.CrossEntropyLoss(weight=ce_weights).cuda(output_device)

        # Enable multi-GPU training if multiple devices are specified
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)

    def load_optimizer(self):
        """Initialize optimizer according to user-specified configuration."""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.arg.base_lr,
            momentum=0.9,
            nesterov=self.arg.nesterov,
            weight_decay=self.arg.weight_decay)
        total_iters = self.iters_per_epoch * self.arg.num_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_iters,
            eta_min=0
        )

    def train(self):
        """Main training loop for one epoch."""
        self.model.train()
        loss_value = []
        for train_loader in self.data_loaders['train']:
            for batch_idx, (data, label, index) in enumerate(train_loader):
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                assert data.shape[1] == 1
                data = data[:, 0]
                label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                output, get_graph = self.model(data)
                loss_ce = self.loss(output, label)
                loss_csc = loss_graph(output, get_graph, label, self.arg.model_args['num_class'], self.output_device)
                loss = loss_ce + self.arg.loss_weight * loss_csc.mean()
                loss_value.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def eval(self):
        """Evaluate model performance on test data."""
        self.model.eval()

        # Run several dry passes to warm up GPU and stabilize timing
        with torch.no_grad():
            for batch_idx, (data, _, index) in enumerate(self.data_loaders['test'][0]):
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                bs, nc = data.shape[:2]
                data = data.reshape((bs * nc,) + data.shape[2:])
                for _ in range(10):  # Warm-up iterations (not counted)
                    _ = self.model(data)

        # Create CUDA events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize GPU before timing to avoid pending operations
        torch.cuda.synchronize()
        start_event.record()

        # Perform multiple inference passes to compute average
        with torch.no_grad():
            for batch_idx, (data, _, index) in enumerate(self.data_loaders['test'][0]):
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                bs, nc = data.shape[:2]
                data = data.reshape((bs * nc,) + data.shape[2:])
                for _ in range(10):
                    _ = self.model(data)

        # Record the end event and synchronize
        end_event.record()
        torch.cuda.synchronize()

        # Compute average inference time per sample (in milliseconds)
        avg_time = start_event.elapsed_time(end_event) / (len(self.data_loaders['test'][0].dataset) * 10)
        print(f'\t\t\tCurrent repitition: inference time {avg_time:.4f} milliseconds')
        return avg_time

    def start(self):
        """Entry point: running process including data split, training, and evaluation."""
        self.data_loaders = {"train": [], "test": []}
        type_mapping = {
            'train_batch_size': int,
            'test_batch_size': int,
            'base_lr': float,
            'num_epoch': int,
            'data_seed': int
        }
        for attr, func in type_mapping.items():
            setattr(self.arg, attr, func(getattr(self.arg, attr)))
        test_dataset_config = arg.test_feeder_args[0]
        try:
            dataset = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim, self.arg.data_norm,
                             self.arg.score_norm, self.arg.label_type)
        except FileNotFoundError as e:
            print(f'Skipping dataset {test_dataset_config.get("name", "unknown")}: {e}')
            import sys
            sys.exit(1)
        sample_name = dataset.sample_name  # Each sample name encodes subject info

        # Determine subject naming rules for different datasets
        dataset_list_0 = ['EHE', 'IRDS', 'PD-Walkway-Gait', 'Walking-Treadmill-Gait', 'TRSP-Seated-Motion',
                          'SPHERE-Stair-Gait', 'SPHERE-Surface-Gait']
        dataset_list_1 = ['SPHERE-Sit', 'SPHERE-Stand', 'PD-Round-Gait', 'PD4T-Round-Gait']
        dataset_list_2 = ['SSBD-Line-Gait']
        dataset_list_3 = ['PD4T-Leg-Agility']

        # Extract subject identifiers for each dataset
        if any(name in dataset.dataset_name for name in dataset_list_0):
            dataset_subjects = sorted(set([i.split('_')[0] for i in sample_name]))
        elif any(name in dataset.dataset_name for name in dataset_list_1):
            dataset_subjects = sorted(set([i.split('_')[1] for i in sample_name]))
        elif any(name in dataset.dataset_name for name in dataset_list_2):
            dataset_subjects = sorted(set([i.split('_')[0] + '_' + i.split('_')[1] for i in sample_name]))
        elif any(name in dataset.dataset_name for name in dataset_list_3):
            dataset_subjects = sorted(set([i.split('_')[2] for i in sample_name]))
        else:
            raise ValueError("Undefined dataset subjects.")

        # Random subject-wise 80/20 split for train/test
        random.seed(self.arg.data_seed)
        random.shuffle(dataset_subjects)
        num_train = int(0.8 * len(dataset_subjects))
        train_subjects, test_subjects = dataset_subjects[:num_train], dataset_subjects[num_train:]

        # Match sample names to subject split
        if 'PD4T' in dataset.dataset_name:
            train_sample_name = [s for s in sample_name if any('_' + sub in s for sub in train_subjects)]
            test_sample_name = [s for s in sample_name if any('_' + sub in s for sub in test_subjects)]
        else:
            train_sample_name = [s for s in sample_name if any(sub + '_' in s for sub in train_subjects)]
            test_sample_name = [s for s in sample_name if any(sub + '_' in s for sub in test_subjects)]
        assert len(train_sample_name) + len(test_sample_name) == len(sample_name)

        # Initialize train/test datasets
        train_set = Feeder(test_dataset_config, data_centered=self.arg.data_centered, data_dim=self.arg.data_dim,
                           data_norm=self.arg.data_norm, score_norm=self.arg.score_norm,
                           label_type=self.arg.label_type, data_indices=train_sample_name, stage='train',
                           num_clips=1)
        test_set = Feeder(test_dataset_config, data_centered=self.arg.data_centered, data_dim=self.arg.data_dim,
                          data_norm=self.arg.data_norm, score_norm=self.arg.score_norm,
                          label_type=self.arg.label_type, data_indices=test_sample_name, stage='test',
                          num_clips=10)

        # Compute class-balanced weights for loss
        num_class = self.arg.model_args['num_class']
        count_class = [train_set.label.count(c) for c in range(num_class)]
        total = len(train_set.label)
        weights = [total / count_class[c] if count_class[c] != 0 else 0 for c in range(num_class)]
        ce_class_weights = torch.tensor(weights)

        # Setup model and save configuration
        self.load_model(ce_class_weights)

        # Build dataLoaders
        self.data_loaders['train'].append(
            DataLoader(dataset=train_set, batch_size=self.arg.train_batch_size, shuffle=True,
                       num_workers=self.arg.num_worker, drop_last=True))
        self.data_loaders['test'].append(
            DataLoader(dataset=test_set, batch_size=self.arg.test_batch_size, shuffle=False,
                       num_workers=self.arg.num_worker, drop_last=False))
        self.iters_per_epoch = len(self.data_loaders['train'][0])

        # Setup optimizer
        self.load_optimizer()

        # Epoch loop
        for epoch in range(0, self.arg.num_epoch):
            self.train()
            if (epoch + 1) % self.arg.eval_interval == 0:
                current_time = self.eval()

        return test_dataset_config['name'], current_time


def update_arg_from_params(arg, dataset_name, idx):
    with open(Path(__file__).resolve().parents[1] / 'plot_figures/results_examples/reproduce.pkl', 'rb') as f:
        param_dict = pkl.load(f)
    p = param_dict['reproducibility'][dataset_name]
    mapping = {
        'bs': ['train_batch_size'],
        'lr': ['base_lr'],
        'ds': ['data_seed']
    }
    for k, v in mapping.items():
        for key_name in v:
            if k in ['lr', 'bs']:
                setattr(arg, key_name, p[k])
            else:
                setattr(arg, key_name, p[k][idx])
    return arg


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name, base_module='SOTA.ProtoGCN'):
    if not name.startswith(('SOTA.', 'VidMotor.')):
        name = f'{base_module}.{name}'
    module_path, class_name = name.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), class_name)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    dataset_names = []
    time_values = []
    for idx in range(10):
        print(f'\t\tRepitition #{idx + 1}:')
        Feeder = import_class(arg.feeder)
        arg = update_arg_from_params(arg, Path(p.config).stem, idx)
        seed_torch(arg.seed)
        processor = Processor(arg)
        dataset_name, time = processor.start()
        dataset_names.append(dataset_name)
        time_values.append(time)
    os.makedirs(os.path.dirname(arg.save_path), exist_ok=True)
    writer = TimeResultWriter(arg.save_path)
    writer.write_result(
        dataset=list(set(dataset_names))[0],
        model_name='ProtoGCN',
        runtime=np.mean(time_values)
    )
    print(f'\tSummary: Average inference time {np.mean(time_values):.4f}')
