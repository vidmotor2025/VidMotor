import argparse
import os
import time
import numpy as np
import yaml
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
    """Define and parse command-line arguments for VidMotor Pretraining (DSTformer)."""
    parser = argparse.ArgumentParser(description='VidMotor Pretraining (DSTformer)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--work-dir', default='./results/temp', help='working directory for storing results')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    parser.add_argument('--save-interval', type=int, default=100, help='interval (in epochs) for saving models')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader to use')
    parser.add_argument('--num-worker', type=int, default=0, help='number of workers for the data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='arguments for the validation data loader')
    parser.add_argument('--data-centered', type=int, default=7, help='the indices of key points used for centralization')
    parser.add_argument('--data-dim', type=str, default='3D', help='dimension of the skeleton sequence')
    parser.add_argument('--data-norm', default='zscore', help='normalization method for the skeleton sequence')
    parser.add_argument('--label-type', default='score_label', help='type of label (class or score)')
    parser.add_argument('--score-norm', type=str2bool, default=True, help='if true, score labels will be normalized to the range 0-1')
    parser.add_argument('--scale-range-train', default=[1, 1], help='data scale for training set')
    parser.add_argument('--backbone', type=str, default='None', help='the backbone model to use')
    parser.add_argument('--backbone-args', default=dict(), help='arguments for the backbone model')
    parser.add_argument('--model', type=str, default='None', help='model to use')
    parser.add_argument('--model-args', default=dict(), help='arguments for the model')
    parser.add_argument('--contrastive-weight', type=float, default=0.5, help='weight for pretraining contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature parameter in pretraining contrastive loss')
    parser.add_argument('--lr-backbone', type=float, default=0.00002, help='initial learning rate for backbone model')
    parser.add_argument('--lr-head', type=float, default=0.00002, help='initial learning rate for model')
    parser.add_argument('--lr-decay', type=float, default=0.99, help='decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='indices of GPUs for training or testing')
    parser.add_argument('--num-epoch', type=int, default=900, help='number of epochs to train')
    return parser


class Processor():
    """Main finetuning controller that manages model, optimizer, and finetuning process."""
    def __init__(self, arg):
        self.arg = arg
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        self.save_arg()
        self.load_model()
        self.load_optimizer()

    def load_model(self):
        """Dynamically import and initialize model architecture."""
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Backbone = import_class(self.arg.backbone)
        self.model_backbone = Backbone(norm_layer=partial(nn.LayerNorm, eps=1e-6), **self.arg.backbone_args)
        Model = import_class(self.arg.model)
        self.model = Model(backbone=self.model_backbone, **self.arg.model_args).cuda(output_device)

        # Define regression loss
        self.loss = nn.MSELoss().cuda(output_device)

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

    def save_arg(self):
        """Save runtime arguments and configuration file."""
        arg_dict = vars(self.arg)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            yaml.dump(arg_dict, f)

    def print_log(self, str, print_time=True):
        """Log messages to console and file timestamps."""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
            print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def pretrain(self, epoch, save_model=False):
        """Main training loop for one epoch."""
        self.model.train()
        self.print_log(f'Training epoch: {epoch + 1}')
        loss_value, loss_con_value, loss_pred_value = [], [], []
        for train_loader in self.data_loaders['train']:
            self.record_time()
            print(train_loader.dataset.dataset_name)
            with tqdm(train_loader, ncols=100) as t:
                # Dataset-specific training logic
                if train_loader.dataset.dataset_name == 'AGF-Olympics':
                    # Only regression loss
                    for batch_idx, (sample_name, data, label, index) in enumerate(t):
                        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                        label = Variable(label.float().cuda(self.output_device), requires_grad=False)
                        output, feats = self.model(data)
                        loss = self.loss(output.squeeze(1), label)
                        loss_value.append(loss.item())
                        loss_pred_value.append(loss.item())
                        loss_con_value.append(0.0)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                elif train_loader.dataset.dataset_name in ['Rhythmic-Gymnastics', 'Push-Up']:
                    # Regression loss + sample-level contrastive loss
                    for batch_idx, (sample_name, data, label, class_label, index) in enumerate(t):
                        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                        label = Variable(label.float().cuda(self.output_device), requires_grad=False)
                        output, feats = self.model(data)
                        loss = self.loss(output.squeeze(1), label)
                        loss_pred_value.append(loss.item())
                        # Contrastive loss (sample-based)
                        if self.arg.contrastive_weight != 0:
                            loss_contrastive = cross_individual_contrastive_loss(feats, class_label, temperature=self.arg.temperature, type='sample')
                            loss_con_value.append(loss_contrastive.item())
                            loss = loss + self.arg.contrastive_weight * loss_contrastive
                        loss_value.append(loss.item())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                else:
                    # Regression loss + cross-individual contrastive loss
                    prev_feats, prev_classes, prev_subjects = None, None, None
                    for batch_idx, (subject_name, sample_name, data, label, class_label, index) in enumerate(t):
                        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                        label = Variable(label.float().cuda(self.output_device), requires_grad=False)
                        output, feats = self.model(data)
                        loss = self.loss(output.squeeze(1), label)
                        loss_pred_value.append(loss.item())
                        # Contrastive loss (individual-based)
                        if self.arg.contrastive_weight != 0:
                            if len(set(subject_name)) > 1:  # intra-batch
                                loss_contrastive = cross_individual_contrastive_loss(feats, class_label, subject_name, temperature=self.arg.temperature, type='subject')
                                loss_con_value.append(loss_contrastive.item())
                                loss = loss + self.arg.contrastive_weight * loss_contrastive
                            else:  # cross-batch
                                if prev_feats is not None:
                                    loss_contrastive = cross_individual_contrastive_loss(feats, class_label, subject_name, prev_feats, prev_classes, prev_subjects, temperature=self.arg.temperature, type='subject')
                                    loss_con_value.append(loss_contrastive.item())
                                    loss = loss + self.arg.contrastive_weight * loss_contrastive
                                else:
                                    loss_con_value.append(0.0)
                        loss_value.append(loss.item())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        if self.arg.contrastive_weight != 0 and len(set(subject_name)) == 1:
                            # Cache the current batch as prev for the next batch
                            prev_feats = feats.detach()
                            prev_classes = class_label
                            prev_subjects = subject_name

        # Log statistics after each epoch
        self.print_log(
            f'\tLearning rates: backbone={self.optimizer.param_groups[0]["lr"]:.8f}, '
            f'head={self.optimizer.param_groups[1]["lr"]:.8f}'
        )
        self.print_log(
            f'\tMean training loss: {np.mean(loss_value):.4f}. '
            f'Mean contrastive loss: {np.mean(loss_con_value):.4f}. '
            f'Mean prediction loss: {np.mean(loss_pred_value):.4f}.'
        )

        # Optionally save model checkpoint
        if save_model:
            torch.save(self.model.state_dict(), f'{self.arg.work_dir}/epoch-{epoch + 1}.pt')

    def start(self):
        """Entry point: running process including data split, training, and evaluation."""
        self.data_loaders = {"train": []}
        self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # Iterate over each dataset
        for train_dataset_config in arg.train_feeder_args:
            try:
                dataset = Feeder(train_dataset_config, self.arg.data_centered, self.arg.data_dim, self.arg.data_norm,
                                 self.arg.score_norm, self.arg.label_type, data_length=None,
                                 scale_range=self.arg.scale_range_train, is_train=True)
            except FileNotFoundError as e:
                print(f'Skipping dataset {train_dataset_config.get("name", "unknown")}: {e}')
                continue

            # Build dataLoaders
            if 'SubjectBatchSampler' in dataset.data_sampler:
                Sampler = import_class(dataset.data_sampler)
                self.data_loaders['train'].append(DataLoader(
                    dataset=dataset, batch_sampler=Sampler(dataset, dataset.batch_size),
                    num_workers=self.arg.num_worker))
            elif 'SequentialSampler' in dataset.data_sampler:
                self.data_loaders['train'].append(DataLoader(
                    dataset=dataset, batch_size=dataset.batch_size, shuffle=True, num_workers=self.arg.num_worker,
                    drop_last=True))

        # Epoch loop
        for epoch in range(0, self.arg.num_epoch):
            save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
            self.pretrain(epoch, save_model=save_model)

        self.print_log(f'work_dir: {self.arg.work_dir}')


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
