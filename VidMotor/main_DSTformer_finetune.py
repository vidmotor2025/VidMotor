import argparse
import os
import time
import numpy as np
import yaml
import pickle
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
from functools import partial
from loss_utils import *


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
    """Define and parse command-line arguments for VidMotor Fine-tuning (DSTformer)."""
    parser = argparse.ArgumentParser(description='VidMotor Fine-tuning (DSTformer)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--work-dir', default='./results/temp', help='working directory for storing results')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    parser.add_argument('--save-interval', type=int, default=50, help='interval (in epochs) for saving models')
    parser.add_argument('--save-score', type=str2bool, default=True, help='whether to save the predicted classification scores')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (in epochs)')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader to use')
    parser.add_argument('--num-worker', type=int, default=0, help='number of workers for the data loader')
    parser.add_argument('--test-feeder-args', default=dict(), help='arguments for the validation data loader')
    parser.add_argument('--data-centered', type=int, default=7, help='the indices of key points used for centralization')
    parser.add_argument('--data-dim', type=str, default='3D', help='dimension of the skeleton sequence')
    parser.add_argument('--data-norm', default='zscore', help='normalization method for the skeleton sequence')
    parser.add_argument('--label-type', default='class_label', help='type of label (class or score)')
    parser.add_argument('--score-norm', type=str2bool, default=False, help='if true, score labels will be normalized to the range 0-1')
    parser.add_argument('--scale-range-train', default=[1, 1], help='data scale for training set')
    parser.add_argument('--scale-range-test', default=[1, 1], help='data scale for test set')
    parser.add_argument('--backbone', type=str, default='None', help='the backbone model to use')
    parser.add_argument('--backbone-args', default=dict(), help='arguments for the backbone model')
    parser.add_argument('--model', type=str, default='None', help='model to use')
    parser.add_argument('--model-args', default=dict(), help='arguments for the model')
    parser.add_argument('--model-pretrain', type=str, default='None', help='path to a pretrained model')
    parser.add_argument('--model-nc', type=str, default='None', help='non-causal head to use')
    parser.add_argument('--model-nc-args', default=dict(), help='arguments for non-causal head')
    parser.add_argument('--lr-backbone', type=float, default=0.0001, help='initial learning rate for backbone model')
    parser.add_argument('--lr-head', type=float, default=0.0001, help='initial learning rate for model')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='indices of GPUs for training or testing')
    parser.add_argument('--train-batch-size', type=int, default=4, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=4, help='test batch size')
    parser.add_argument('--weight-cf', type=float, default=1.0, help='weight for counterfactual loss')
    parser.add_argument('--margin-cf', type=float, default=0.6, help='margin for counterfactual loss')
    parser.add_argument('--num-epoch', type=int, default=300, help='number of epochs to train')
    return parser


class Processor():
    """Main finetuning controller that manages model, optimizer, and finetuning process."""
    def __init__(self, arg):
        self.arg = arg
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def load_model(self, ce_weights=None):
        """Dynamically import and initialize model architecture."""
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Backbone = import_class(self.arg.backbone)
        self.model_backbone = Backbone(norm_layer=partial(nn.LayerNorm, eps=1e-6), **self.arg.backbone_args)
        Model = import_class(self.arg.model)
        self.model = Model(backbone=self.model_backbone, **self.arg.model_args).cuda(output_device)
        NCHead = import_class(self.arg.model_nc)
        self.nchead = NCHead(**self.arg.model_nc_args).cuda(output_device)

        # Load pretrained model weights
        print("Loading weights: ", self.arg.model_pretrain)
        if self.arg.model_pretrain != 'None':
            model_pretrain = torch.load(self.arg.model_pretrain, map_location='cpu')
            model_backbone_dict = self.model.state_dict()
            pretrained_dict = {}
            for k, v in model_pretrain.items():
                new_k = k[7:] if k.startswith('module.') else k
                if new_k in model_backbone_dict and 'head.bn' not in new_k and 'head.fc1' not in new_k and 'head.fc2' not in new_k:
                    pretrained_dict[new_k] = v
            model_backbone_dict.update(pretrained_dict)
            self.model.load_state_dict(model_backbone_dict)

        # Define classification loss
        self.loss = nn.CrossEntropyLoss(weight=ce_weights).cuda(output_device)

        # Enable multi-GPU training if multiple devices are specified
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)

    def load_optimizer(self):
        """Initialize optimizer according to user-specified configuration."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.optimizer_main = optim.AdamW(
            [
                {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": self.arg.lr_backbone},
                {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": self.arg.lr_head}
            ],
            lr=self.arg.lr_backbone,
            weight_decay=self.arg.weight_decay
        )
        self.optimizer_nc = optim.AdamW(self.nchead.parameters(),
            lr=self.arg.lr_head,
            weight_decay=self.arg.weight_decay
        )

    def save_arg(self):
        """Save runtime arguments and configuration file."""
        arg_dict = vars(self.arg)
        with open(f'{self.save_path}/config.yaml', 'w') as f:
            yaml.dump(arg_dict, f)

    def print_log(self, str, path, print_time=True):
        """Log messages to console and file timestamps."""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        with open(f'{path}/log.txt', 'a') as f:
            print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def train(self, epoch, save_model=False):
        """Main training loop for one epoch."""
        self.model.train()
        self.print_log(f'Training epoch: {epoch + 1}', self.save_path)
        loss_value, loss_nc_value = [], []
        for train_loader in self.data_loaders['train']:
            self.record_time()
            with tqdm(train_loader, ncols=100) as t:
                for batch_idx, (data, label, index) in enumerate(t):
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

        # Log statistics after each epoch
        self.print_log(
            f'\tLearning rates: backbone={self.optimizer_main.param_groups[0]["lr"]:.8f}, '
            f'head={self.optimizer_main.param_groups[1]["lr"]:.8f}',
            self.save_path
        )
        self.print_log(f'\tMean training loss: main {np.mean(loss_value):.4f}.', self.save_path)

        # Optionally save model checkpoint
        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_path}/epoch-{epoch + 1}.pt')

    def eval(self, epoch, save_score=False):
        """Evaluate model performance on test data."""
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}', self.save_path)
        with torch.no_grad():
            for test_loader in self.data_loaders['test']:
                with tqdm(test_loader, ncols=100) as t:
                    score_frag = []
                    for batch_idx, (data, _, index) in enumerate(t):
                        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                        output_c = self.model(data, label=None)
                        score_frag.append(output_c.cpu().numpy())
                        
                    # Concatenate predictions and compute accuracy
                    output_score = np.concatenate(score_frag)
                    accuracy = test_loader.dataset.top_k(output_score, 1)
                    self.print_log(f'\tTest Dataset: {test_loader.dataset.dataset_name}; Accuracy: {accuracy * 100:.2f}', self.save_path)
                    
                    # Optionally save prediction scores
                    if save_score:
                        with open(f'{self.save_path}/epoch{epoch + 1}_score.pkl', 'wb') as f:
                            pickle.dump(dict(zip(test_loader.dataset.sample_name, output_score)), f)

    def start(self):
        """Entry point: running process including data split, training, and evaluation."""
        self.data_loaders = {"train": [], "test": []}
        # Iterate over each dataset
        for test_dataset_config in arg.test_feeder_args:
            try:
                dataset = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim, self.arg.data_norm,
                                 self.arg.score_norm, self.arg.label_type, is_full_set=True)
            except FileNotFoundError as e:
                print(f'Skipping dataset {test_dataset_config.get("name", "unknown")}: {e}')
                continue
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

            # Prepare output directory and logs
            self.save_path = f"{self.arg.work_dir}/{dataset.dataset_name}"
            os.makedirs(self.save_path, exist_ok=True)
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n', self.save_path)

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
                               data_indices=train_sample_name, scale_range=self.arg.scale_range_train, is_train=True)
            test_set = Feeder(test_dataset_config, self.arg.data_centered, self.arg.data_dim,
                              self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                              data_length=train_set.data_length,
                              data_indices=test_sample_name, scale_range=self.arg.scale_range_test, is_train=False)
            
            # Compute class-balanced weights for loss
            num_class = self.arg.model_args['num_class']
            count_class = [train_set.label.count(c) for c in range(num_class)]
            total = len(train_set.label)
            weights = [total / count_class[c] if count_class[c] != 0 else 0 for c in range(num_class)]
            ce_class_weights = torch.tensor(weights)

            # Setup model, optimizer, and save configuration
            self.save_arg()
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
            for epoch in range(0, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                if (epoch + 1) % self.arg.eval_interval == 0:
                    self.eval(epoch, save_score=self.arg.save_score)
                
            # Clear data loaders for next dataset
            self.data_loaders['train'].clear()
            self.data_loaders['test'].clear()
            self.print_log(f'work_dir:{self.save_path}', self.save_path)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


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
    Feeder = import_class(arg.feeder)
    seed_torch(arg.seed)
    processor = Processor(arg)
    processor.start()
