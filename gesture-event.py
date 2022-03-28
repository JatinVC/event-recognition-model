# imports
# python core imports
from venv import create
from typing import Callable, Dict, Optional, Tuple
from abc import abstractmethod
import struct
import os
from concurrent.futures import ThreadPoolExecutor
import time
import math
from re import M
import shutil
import argparse
import logging
import sys

# python external modules
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from posixpath import split

# torch imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision.datasets import DatasetFolder, utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# spikingjelly imports
from spikingjelly.clock_driven import functional, surrogate, layer
import spikingjelly.event_driven.neuron as neuron
import spikingjelly.event_driven.encoding as encoding
from datasets.__init__ import *


# config variables
np_savez = np.savez_compressed
_seed_ = 2020
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

# logging setup for production only
logging.basicConfig(filename=f'logs/runlog.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger()
sys.stderr.write = logger.error
sys.stdout.write = logger.info

# trying using event parameter
# neural network parameters
class Net(nn.Module):
    def __init__(self, m, T):
        super().__init__()
        self.tempotron = neuron.Tempotron(480*640*m, 10, T)
    
    def forward(self, x: torch.Tensor):
        return self.tempotron(x, 'v_max')

parser = argparse.ArgumentParser(description='spikingjelly Tempotron MNIST Training')

parser.add_argument('--device', default='cuda:0', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='./output', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=16, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
# parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')
parser.add_argument('-m', default=16, type=int, help='使用高斯调谐曲线编码每个像素点使用的神经元数量，例如“16”\n input neuron number for encoding a piexl in GaussianTuning encoder, e.g., "16"')

args = parser.parse_args("")
device = args.device

data_dir = args.dataset_dir
log_dir = args.log_dir
model_output_dir = args.model_output_dir

batch_size = args.batch_size
T = args.T
learning_rate = args.lr
train_epoch = args.epoch
m = args.m

encoder = encoding.GaussianTuning(n=1, m=m, x_min=torch.zeros(size=[1]).to(device), x_max=torch.ones(size=[1]).to(device))

writer = SummaryWriter(log_dir)

train_set = FYPDataset(data_dir, train=True, data_type='frame', split_by='number', frames_number=T)
test_set = FYPDataset(data_dir, train=False, data_type='frame', split_by='number', frames_number=T)

train_data_loader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

test_data_loader = DataLoader(
    dataset= train_set,
    batch_size = batch_size,
    shuffle=False,
    drop_last=False
)

net = Net(m, T).to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

train_times = 0
max_test_accuracy = 0

if __name__ == '__main__':
    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        net.train()
        train_correct_sum = 0
        train_sum = 0
        for img, label in train_data_loader:
            img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]
            in_spikes = encoder.encode(img.to(device), T)  # [batch_size, 1, 784, m]
            in_spikes = in_spikes.view(in_spikes.shape[0], -1)  # [batch_size, 784*m]

            optimizer.zero_grad()

            v_max = net(in_spikes)
            loss = neuron.Tempotron.mse_loss(v_max, net.tempotron.v_threshold, label.to(device), 10)
            loss.backward()
            optimizer.step()

            train_correct_sum += (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
            train_sum += label.numel()

            train_batch_acc = (v_max.argmax(dim=1) == label.to(device)).float().mean().item()
            writer.add_scalar('train_batch_acc', train_batch_acc, train_times)

            train_times += 1
        # train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with torch.no_grad():
            correct_num = 0
            img_num = 0
            for img, label in test_data_loader:
                img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]

                in_spikes = encoder.encode(img.to(device), T)  # [batch_size, 1, 784, m]
                in_spikes = in_spikes.view(in_spikes.shape[0], -1)  # [batch_size, 784*m]
                v_max = net(in_spikes)
                correct_num += (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
                img_num += img.shape[0]
            test_accuracy = correct_num / img_num
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        print()
    
    # 保存模型
    torch.save(net, model_output_dir + "/eventdata.ckpt")
    # 读取模型
    # net = torch.load(model_output_dir + "/tempotron_snn_mnist.ckpt")