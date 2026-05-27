import argparse
import os
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
from timm.scheduler.cosine_lr import CosineLRScheduler
from loss_utils import LabelSmoothingCrossEntropy
import warnings
warnings.filterwarnings("ignore")


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
    """Define and parse command-line arguments for MMN."""
    parser = argparse.ArgumentParser(description='MMN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--work-dir', default='./results/temp', help='working directory for storing results')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    parser.add_argument('--save-interval', type=int, default=50, help='interval (in epochs) for saving models')
    parser.add_argument('--save-score', type=str2bool, default=True,
                        help='whether to save the predicted classification scores')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (in epochs)')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader to use')
    parser.add_argument('--num-worker', type=int, default=0, help='number of workers for the data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='arguments for the training data loader')
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
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def load_model(self):
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

    def load_scheduler(self, n_iter_per_epoch):
        """Initialize the learning rate scheduler for training."""
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)
        self.lr_scheduler = None
        if self.arg.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
                lr_min=self.arg.min_lr,
                warmup_lr_init=self.arg.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.arg.warmup_prefix
            )
        else:
            raise ValueError()

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
                for batch_idx, (data, label, index_t, index) in enumerate(t):
                    self.lr_scheduler.step_update(self.global_step)
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

        # Log statistics after each epoch
        self.print_log(f'\tLearning rate: {self.optimizer.param_groups[0]["lr"]:.8f}.', self.save_path)
        self.print_log(f'\tMean training loss: {np.mean(loss_value):.4f}.', self.save_path)

        # Optionally save model checkpoint
        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_path}/epoch-{epoch + 1}.pt')

    def eval(self):
        """Evaluate model performance on test data."""
        self.model.eval()

        # Run several dry passes to warm up GPU and stabilize timing
        for loader_index in range(4):
            with torch.no_grad():
                for batch_idx, (data, _, index_t, index) in enumerate(self.data_loaders['test'][loader_index]):
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
                for batch_idx, (data, _, index_t, index) in enumerate(self.data_loaders['test'][loader_index]):
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                    index_t = Variable(index_t.float().cuda(self.output_device), requires_grad=False)
                    for _ in range(10):  # Repeat inference 100 times for stability
                        _ = self.model(data, index_t)

            # Record the end event and synchronize
            end_event.record()
            torch.cuda.synchronize()

            # Compute average inference time per sample (in milliseconds)
            avg_time = start_event.elapsed_time(end_event) / (len(self.data_loaders['test'][loader_index].dataset) * 10)
            self.print_log(f'Average inference time: {avg_time} milliseconds', self.save_path)

    def start(self):
        """Entry point: running process including data split, training, and evaluation."""
        self.data_loaders = {"train": [], "test": []}

        # Initialize train datasets
        try:
            train_set = Feeder(self.arg.train_feeder_args[0], self.arg.data_centered, self.arg.data_dim,
                               self.arg.data_norm, self.arg.score_norm, self.arg.label_type, stage='train')
        except FileNotFoundError as e:
            print(f'Skipping dataset {self.arg.train_feeder_args[0].get("name", "unknown")}: {e}')
            import sys
            sys.exit(1)

        # Prepare output directory and logs
        self.save_path = f"{self.arg.work_dir}/{train_set.dataset_name}_inference-time"
        os.makedirs(self.save_path, exist_ok=True)
        self.print_log(f'Parameters:\n{str(vars(self.arg))}\n', self.save_path)

        # Setup model, optimizer, and save configuration
        self.save_arg()
        self.load_model()
        self.load_optimizer()
        self.global_step = 0

        # Build training dataLoaders
        self.data_loaders['train'].append(
            DataLoader(dataset=train_set, batch_size=self.arg.train_batch_size, shuffle=True,
                       num_workers=self.arg.num_worker, drop_last=True))
        self.load_scheduler(len(self.data_loaders['train'][0]))

        # Epoch training loop
        for epoch in range(0, self.arg.num_epoch):
            save_model = (epoch + 1 == self.arg.num_epoch)
            self.train(epoch, save_model=save_model)
        self.print_log(f'work_dir:{self.save_path}', self.save_path)

        # Load and evaluate across multiple test centers
        test_set_center1 = Feeder(self.arg.test_feeder_args[0], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, stage='test')
        test_set_center2 = Feeder(self.arg.test_feeder_args[1], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, stage='test')
        test_set_center3 = Feeder(self.arg.test_feeder_args[2], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, stage='test')
        test_set_center4 = Feeder(self.arg.test_feeder_args[3], self.arg.data_centered, self.arg.data_dim,
                                  self.arg.data_norm, self.arg.score_norm, self.arg.label_type,
                                  data_length=train_set.data_length, stage='test')
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
        self.eval()


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
