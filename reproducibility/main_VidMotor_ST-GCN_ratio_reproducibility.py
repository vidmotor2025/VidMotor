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
import sys
from pathlib import Path
import importlib
import re
import pickle as pkl
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from VidMotor.loss_utils import counterfactual_loss, non_causal_loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from result_writer import RatioResultWriter


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
    """Define and parse command-line arguments for VidMotor Fine-tuning (ST-GCN)."""
    parser = argparse.ArgumentParser(description='VidMotor Fine-tuning (ST-GCN)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--work-dir', default='./results/temp', help='working directory for storing results')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (in epochs)')
    parser.add_argument('--save-path', default='./results/reproduce_results_10%training.xlsx', help='path to save results')
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
    parser.add_argument('--model-pretrain', type=str, default='None', help='path to a pretrained model')
    parser.add_argument('--model-nc', type=str, default='None', help='the causal model to use')
    parser.add_argument('--model-nc-args', default=dict(), help='arguments for the causal model')
    parser.add_argument('--train-batch-size', type=int, default=4, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=4, help='test batch size')
    parser.add_argument('--weight-cf', type=float, default=1.0, help='weight for counterfactual loss')
    parser.add_argument('--margin-cf', type=float, default=0.6, help='weight for counterfactual loss')
    parser.add_argument('--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='indices of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--num-epoch', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--train-ratio', type=float, default=1.0, help='training data ratio')
    return parser


class Processor():
    """Main finetuning controller that manages model, optimizer, and finetuning process."""
    def __init__(self, arg):
        self.arg = arg

    def load_model(self, ce_weights=None):
        """Dynamically import and initialize model architecture."""
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        NCHead = import_class(self.arg.model_nc)
        self.nchead = NCHead(**self.arg.model_nc_args).cuda(output_device)

        # Load pretrained model weights
        if self.arg.model_pretrain != 'None':
            model_pretrain = torch.load(self.arg.model_pretrain, map_location='cpu')
            model_dict = self.model.state_dict()
            pretrained_dict = {}
            for k, v in model_pretrain.items():
                new_k = k[7:] if k.startswith('module.') else k
                if new_k in model_dict and 'fcn' not in new_k:
                    pretrained_dict[new_k] = v
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        # Define classification loss
        self.loss = nn.CrossEntropyLoss(weight=ce_weights).cuda(output_device)

        # Enable multi-GPU training if multiple devices are specified
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)

    def load_optimizer(self):
        """Initialize optimizer according to user-specified configuration."""
        if self.arg.optimizer == 'SGD':
            self.optimizer_main = optim.SGD(self.model.parameters(), lr=self.arg.base_lr, momentum=0.9, nesterov=True,
                                            weight_decay=self.arg.weight_decay)
            self.optimizer_nc = optim.SGD(self.nchead.parameters(), lr=self.arg.base_lr, momentum=0.9, nesterov=True,
                                          weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer_main = optim.Adam(self.model.parameters(), lr=self.arg.base_lr,
                                             weight_decay=self.arg.weight_decay)
            self.optimizer_nc = optim.Adam(self.nchead.parameters(), lr=self.arg.base_lr,
                                           weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def train(self):
        """Main training loop for one epoch."""
        self.model.train()
        loss_value, loss_nc_value = [], []
        for train_loader in self.data_loaders['train']:
            for batch_idx, (data, label, index) in enumerate(train_loader):
                if label.unique().numel() == 1:
                    continue  # Skip batches with only one class
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                output_c, feature_nc, output_cf = self.model(data, label)

                # Compute causal and counterfactual losses
                loss_c = self.loss(output_c, label)
                loss_cf = counterfactual_loss(output_cf[:, :], label, margin=self.arg.margin_cf)
                loss = loss_c + self.arg.weight_cf * loss_cf
                loss_value.append(loss.item())

                # Backward pass for main model
                self.optimizer_main.zero_grad()
                loss.backward()
                self.optimizer_main.step()

                # Compute non-causal loss and update non-causal head
                output_nc = self.nchead(feature_nc.detach())
                loss_nc = non_causal_loss(output_c.detach(), output_nc)
                loss_nc_value.append(loss_nc.item())
                self.optimizer_nc.zero_grad()
                loss_nc.backward()
                self.optimizer_nc.step()

    def eval(self):
        """Evaluate model performance on test data."""
        self.model.eval()
        with torch.no_grad():
            for test_loader in self.data_loaders['test']:
                score_frag = []
                for batch_idx, (data, _, index) in enumerate(test_loader):
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                    output_c = self.model(data, label=None)
                    score_frag.append(output_c.cpu().numpy())

                # Concatenate predictions and compute accuracy
                output_score = np.concatenate(score_frag)
                true_label = np.array(test_loader.dataset.ori_label)
                accuracy = test_loader.dataset.top_k(output_score, 1)
                probs = F.softmax(torch.from_numpy(output_score), dim=1).numpy()
                n_classes = self.arg.model_args['num_class']
                if n_classes == 2:
                    auc = roc_auc_score(y_true=true_label, y_score=probs[:, 1])
                else:
                    labels = list(range(n_classes))
                    auc = roc_auc_score(y_true=true_label, y_score=probs, multi_class='ovr', average='macro',
                                        labels=labels)
                if accuracy > getattr(self, 'final_acc', 0):
                    self.final_acc = accuracy
                    self.metrics['auc'] = auc


    def start(self):
        """Entry point: running process including data split, training, and evaluation."""
        self.data_loaders = {"train": [], "test": []}
        type_mapping = {
            'train_batch_size': int,
            'test_batch_size': int,
            'base_lr': float,
            'num_epoch': int,
            'data_seed': int,
            'seed': int
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
        ori_train_subjects, test_subjects = dataset_subjects[:num_train], dataset_subjects[num_train:]

        # Due to the fact that, in some datasets, selecting only 10% of the training subjects would result in fewer than one subject being sampled
        # or would fail to cover all categories within the dataset, the selection ratio for these special datasets had to be adjusted accordingly.
        # The adjusted ratio represents the minimum feasible proportion that still allows fine-tuning.
        if dataset.dataset_name in ['SPHERE-Stair-Gait', 'SPHERE-Surface-Gait', 'Walking-Treadmill-Gait',
                                    'EHE-Hands-Wave', 'EHE-Waist-BendL', 'EHE-Waist-BendR', 'PD-Walkway-Gait'] and self.arg.train_ratio == 0.1:
            self.arg.train_ratio = 0.2
        elif dataset.dataset_name in ['SPHERE-Sit', 'SPHERE-Stand', ] and self.arg.train_ratio == 0.1:
            self.arg.train_ratio = 0.3
        # Select training subset size according to the given ratio
        num_sub_train = int(self.arg.train_ratio * len(ori_train_subjects))
        if num_sub_train == 0:
            return test_dataset_config['name'], None
        else:
            random_count = 0
            while random_count < 1000:
                # Randomly select training subjects
                random.shuffle(ori_train_subjects)
                train_subjects = ori_train_subjects[:num_sub_train]

                # Match sample names to subject split
                if 'PD4T' in dataset.dataset_name:
                    train_sample_name = [s for s in sample_name if any('_' + sub in s for sub in train_subjects)]
                    test_sample_name = [s for s in sample_name if any('_' + sub in s for sub in test_subjects)]
                else:
                    train_sample_name = [s for s in sample_name if any(sub + '_' in s for sub in train_subjects)]
                    test_sample_name = [s for s in sample_name if any(sub + '_' in s for sub in test_subjects)]

                # Initialize train/test datasets
                train_set = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim,
                                    self.arg.data_norm, self.arg.score_norm, self.arg.label_type, data_length=None,
                                    data_indices=train_sample_name)
                test_set = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim,
                                    self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                    data_length=train_set.data_length, data_indices=test_sample_name)

                # Ensure all classes are represented in training set
                if len(set(train_set.label)) != self.arg.model_args['num_class']:
                    random_count += 1
                    if random_count >= 1000:
                        return test_dataset_config['name'], None
                else:
                    # Compute class-balanced weights for loss
                    num_class = self.arg.model_args['num_class']
                    count_class = [train_set.label.count(c) for c in range(num_class)]
                    total = len(train_set.label)
                    weights = [total / count_class[c] if count_class[c] != 0 else 0 for c in range(num_class)]
                    ce_class_weights = torch.tensor(weights)

                    # Setup model and optimizer
                    self.load_model(ce_class_weights)
                    self.load_optimizer()

                    # Build dataLoaders
                    self.data_loaders['train'].append(
                        DataLoader(dataset=train_set, batch_size=self.arg.train_batch_size, shuffle=True,
                                    num_workers=self.arg.num_worker, drop_last=True))
                    self.data_loaders['test'].append(
                        DataLoader(dataset=test_set, batch_size=self.arg.test_batch_size, shuffle=False,
                                    num_workers=self.arg.num_worker, drop_last=False))

                    # Epoch loop
                    self.final_acc = 0.0
                    self.metrics = {}
                    for epoch in range(0, self.arg.num_epoch):
                        self.train()
                        if (epoch + 1) % self.arg.eval_interval == 0:
                            self.eval()
                    return test_dataset_config['name'], self.metrics['auc']


def update_arg_from_params(arg, dataset_name, idx):
    with open(Path(__file__).resolve().parents[1] / 'plot_figures/results_examples/reproduce.pkl', 'rb') as f:
        param_dict = pkl.load(f)
    p = param_dict['ratio_reproducibility'][dataset_name]
    mapping = {
        'bs': ['train_batch_size', 'test_batch_size'],
        'lr': ['base_lr'],
        'ds': ['data_seed'],
        's': ['seed']
    }
    for k, v in mapping.items():
        for key_name in v:
            if k in ['lr', 'bs', 's']:
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


def import_class(name, base_module='VidMotor'):
    if not name.startswith('VidMotor.'):
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
    aucs = []
    for idx in range(10):
        Feeder = import_class(arg.feeder)
        arg = update_arg_from_params(arg, Path(p.config).stem, idx)
        seed_torch(arg.seed)
        processor = Processor(arg)
        dataset_name, auc = processor.start()
        if auc is None:
            continue
        dataset_names.append(dataset_name)
        aucs.append(auc)
    os.makedirs(os.path.dirname(arg.save_path), exist_ok=True)
    writer = RatioResultWriter(arg.save_path)
    # Write result only if we collected valid data
    if len(dataset_names) > 0 and len(aucs) > 0:
        writer.write_result(
            dataset=list(set(dataset_names))[0],
            model_name='Ours',
            auc=np.mean(aucs) * 100
        )
        print(f'\tSummary: Average AUROC {np.mean(aucs):.2f}')
    else:
        print('\tNo valid results to write.')
