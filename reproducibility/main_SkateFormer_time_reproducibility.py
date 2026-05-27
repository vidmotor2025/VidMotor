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
warnings.filterwarnings("ignore")
import sys
import re
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from result_writer import TimeResultWriter
from SOTA.SkateFormer.loss_utils import LabelSmoothingCrossEntropy


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
    """Define and parse command-line arguments for SkateFormer."""
    parser = argparse.ArgumentParser(description='SkateFormer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
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
    parser.add_argument('--train-batch-size', type=int, default=4, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
    parser.add_argument('--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='indices of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--warmup-lr', type=float, default=1e-7)
    parser.add_argument('--warmup_prefix', type=bool, default=False)
    parser.add_argument('--warm_up_epoch', type=int, default=25)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--num-epoch', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='weight decay for optimizer')
    parser.add_argument('--grad-clip', type=bool, default=True)
    parser.add_argument('--grad-max', type=float, default=1.0)
    return parser


class Processor():
    """Main running controller that manages model, optimizer, and running process."""
    def __init__(self, arg):
        self.arg = arg

    def load_model(self):
        """Dynamically import and initialize model architecture."""
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(output_device)

        # Define classification loss
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)

        # Enable multi-GPU training if multiple devices are specified
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)

    def load_optimizer(self):
        """Initialize optimizer according to user-specified configuration."""
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def train(self):
        """Main training loop for one epoch."""
        self.model.train()
        loss_value = []
        for train_loader in self.data_loaders['train']:
            for batch_idx, (data, label, index_t, index) in enumerate(train_loader):
                self.global_step += 1
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                index_t = Variable(index_t.float().cuda(self.output_device), requires_grad=False)
                output = self.model(data, index_t)
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                if self.arg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
                self.optimizer.step()

    def eval(self):
        """Evaluate model performance and measure inference time."""
        self.model.eval()

        # Run several dry passes to warm up GPU and stabilize timing
        with torch.no_grad():
            for batch_idx, (data, _, index_t, index) in enumerate(self.data_loaders['test'][0]):
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                index_t = Variable(index_t.float().cuda(self.output_device), requires_grad=False)
                for _ in range(10):  # Warm-up iterations (not counted)
                    _ = self.model(data, index_t)

        # Create CUDA events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize GPU before timing to avoid pending operations
        torch.cuda.synchronize()
        start_event.record()

        # Perform multiple inference passes to compute average
        with torch.no_grad():
            for batch_idx, (data, _, index_t, index) in enumerate(self.data_loaders['test'][0]):
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                index_t = Variable(index_t.float().cuda(self.output_device), requires_grad=False)
                for _ in range(10):  # Repeat inference 10 times for stability
                    _ = self.model(data, index_t)

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
        train_set = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim,
                            self.arg.data_norm, self.arg.score_norm, self.arg.label_type, data_length=None,
                            data_indices=train_sample_name, stage='train', require_index_t=True)
        test_set = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim,
                            self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                            data_length=train_set.data_length, data_indices=test_sample_name, stage='test',
                            require_index_t=True)

        # Setup model, optimizer, and save configuration
        self.load_model()
        self.load_optimizer()
        self.global_step = 0

        # Build dataLoaders
        self.data_loaders['train'].append(
            DataLoader(dataset=train_set, batch_size=self.arg.train_batch_size, shuffle=True,
                        num_workers=self.arg.num_worker, drop_last=True))
        self.data_loaders['test'].append(
            DataLoader(dataset=test_set, batch_size=self.arg.test_batch_size, shuffle=False,
                        num_workers=self.arg.num_worker, drop_last=False))

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


def import_class(name, base_module='SOTA.SkateFormer'):
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
        model_name='SkateFormer',
        runtime=np.mean(time_values)
    )
    print(f'\tSummary: Average inference time {np.mean(time_values):.4f}')
