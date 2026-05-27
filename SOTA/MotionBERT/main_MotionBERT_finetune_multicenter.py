import argparse
import os
import time
import numpy as np
import yaml
import pickle
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
from functools import partial
from model.tools import *


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
    """Define and parse command-line arguments for MotionBERT Fine-tuning."""
    parser = argparse.ArgumentParser(description='MotionBERT Fine-tuning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--work-dir', default='./results/temp', help='working directory for storing results')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    parser.add_argument('--save-interval', type=int, default=50, help='interval (in epochs) for saving models')
    parser.add_argument('--save-score', type=str2bool, default=True, help='whether to save the predicted classification scores')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (in epochs)')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader to use')
    parser.add_argument('--num-worker', type=int, default=0, help='number of workers for the data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='arguments for the training data loader')
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
    parser.add_argument('--lr-backbone', type=float, default=0.0005, help='initial learning rate for backbone model')
    parser.add_argument('--lr-head', type=float, default=0.005, help='initial learning rate for model')
    parser.add_argument('--lr-decay', type=float, default=0.99, help='decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='indices of GPUs for training or testing')
    parser.add_argument('--train-batch-size', type=int, default=4, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=4, help='test batch size')
    parser.add_argument('--num-epoch', type=int, default=30, help='number of epochs to train')
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

        # Load pretrained model weights
        checkpoint = torch.load(self.arg.model_pretrain, map_location=lambda storage, loc: storage)['model_pos']
        self.model_backbone = load_pretrained_weights(self.model_backbone, checkpoint)

        # Define classification loss
        self.loss = nn.CrossEntropyLoss(weight=ce_weights).cuda(output_device)

        # Enable multi-GPU training if multiple devices are specified
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)

    def load_optimizer(self):
        """Initialize optimizer according to user-specified configuration."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.optimizer = optim.AdamW(
            [
                {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": self.arg.lr_backbone},
                {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": self.arg.lr_head},
            ],
            lr=self.arg.lr_backbone,
            weight_decay=self.arg.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.arg.lr_decay)

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
        loss_value = []
        for train_loader in self.data_loaders['train']:
            self.record_time()
            with tqdm(train_loader, ncols=100) as t:
                for batch_idx, (data, label, index) in enumerate(t):
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                    label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    loss_value.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # Log statistics after each epoch
        self.print_log(
            f'\tLearning rates: backbone={self.optimizer.param_groups[0]["lr"]:.8f}, '
            f'head={self.optimizer.param_groups[1]["lr"]:.8f}',
            self.save_path
        )
        self.print_log(f'\tMean training loss: main {np.mean(loss_value):.4f}.', self.save_path)
        
        # Optionally save model checkpoint
        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_path}/epoch-{epoch + 1}.pt')

    def eval(self, save_score=False):
        """Evaluate model performance on test data."""
        self.model.eval()
        with torch.no_grad():
            for test_loader in self.data_loaders['test']:
                with tqdm(test_loader, ncols=100) as t:
                    score_frag = []
                    for batch_idx, (data, _, index) in enumerate(t):
                        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                        output = self.model(data)
                        score_frag.append(output.cpu().numpy())
                        
                    # Concatenate predictions and compute accuracy
                    output_score = np.concatenate(score_frag)
                    accuracy = test_loader.dataset.top_k(output_score, 1)
                    self.print_log(f'\tTest Dataset: {test_loader.dataset.dataset_name}; Accuracy: {accuracy * 100:.2f}', self.save_path)
                    
                    # Optionally save prediction scores
                    if save_score:
                        out_dir = os.path.join(self.save_path, test_loader.dataset.dataset_name)
                        os.makedirs(out_dir, exist_ok=True)
                        with open(f'{out_dir}/epoch{self.arg.num_epoch}_score.pkl', 'wb') as f:
                            pickle.dump(dict(zip(test_loader.dataset.sample_name, output_score)), f)

    def start(self):
        """Entry point: running process including data split, training, and evaluation."""
        self.data_loaders = {"train": [], "test": []}

        # Initialize train dataset
        try:
            train_set = Feeder(self.arg.train_feeder_args[0], self.arg.data_centered, self.arg.data_dim,
                               self.arg.data_norm, self.arg.score_norm, self.arg.label_type, data_length=None,
                               scale_range=self.arg.scale_range_train, is_train=True)
        except FileNotFoundError as e:
            print(f'Skipping dataset {self.arg.train_feeder_args[0].get("name", "unknown")}: {e}')
            import sys
            sys.exit(1)
        
        # Prepare output directory and logs
        self.save_path = f"{self.arg.work_dir}/{train_set.dataset_name}"
        os.makedirs(self.save_path, exist_ok=True)
        self.print_log(f'Parameters:\n{str(vars(self.arg))}\n', self.save_path)

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
        
        # Build training dataloader
        self.data_loaders['train'].append(
            DataLoader(dataset=train_set, batch_size=self.arg.train_batch_size, shuffle=True,
                       num_workers=self.arg.num_worker, drop_last=True))

        # Epoch training loop
        for epoch in range(0, self.arg.num_epoch):
            save_model = (epoch + 1 == self.arg.num_epoch)
            self.train(epoch, save_model=save_model)
            self.scheduler.step()
        self.print_log(f'work_dir:{self.save_path}', self.save_path)

        # Load and evaluate across multiple test centers
        test_set_center1 = Feeder(self.arg.test_feeder_args[0], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, scale_range=self.arg.scale_range_test,
                                  is_train=False)
        test_set_center2 = Feeder(self.arg.test_feeder_args[1], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, scale_range=self.arg.scale_range_test,
                                  is_train=False)
        test_set_center3 = Feeder(self.arg.test_feeder_args[2], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, scale_range=self.arg.scale_range_test,
                                  is_train=False)
        test_set_center4 = Feeder(self.arg.test_feeder_args[3], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, scale_range=self.arg.scale_range_test,
                                  is_train=False)
        self.data_loaders['test'].append(
            DataLoader(dataset=test_set_center1, batch_size=self.arg.test_batch_size, shuffle=False,
                       num_workers=self.arg.num_worker, drop_last=False))
        self.data_loaders['test'].append(
            DataLoader(dataset=test_set_center2, batch_size=self.arg.test_batch_size, shuffle=False,
                       num_workers=self.arg.num_worker, drop_last=False))
        self.data_loaders['test'].append(
            DataLoader(dataset=test_set_center3, batch_size=self.arg.test_batch_size, shuffle=False,
                       num_workers=self.arg.num_worker, drop_last=False))
        self.data_loaders['test'].append(
            DataLoader(dataset=test_set_center4, batch_size=self.arg.test_batch_size, shuffle=False,
                       num_workers=self.arg.num_worker, drop_last=False))
        self.eval(save_score=self.arg.save_score)


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
