from venv import create
from torchvision.datasets import DatasetFolder, utils
from typing import Callable, Dict, Optional, Tuple
from abc import abstractmethod
import scipy.io
import struct
import numpy as np
import torch.utils.data
import os
from concurrent.futures import ThreadPoolExecutor
import time
import multiprocessing
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import math
import tqdm
import shutil
import pandas as pd
from re import M
from tkinter import W
from tkinter.filedialog import test
from torch.utils.data import Dataset, DataLoader
from posixpath import split


np_savez = np.savez_compressed


def play_frame(x: torch.Tensor or np.ndarray, save_gif_to: str = None) -> None:
    '''
    :param x: frames with ``shape=[T, 2, H, W]``
    :type x: torch.Tensor or np.ndarray
    :param save_gif_to: If ``None``, this function will play the frames. If ``True``, this function will not play the frames
        but save frames to a gif file in the directory ``save_gif_to``
    :type save_gif_to: str
    :return: None
    '''

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]
    img_tensor[:, 2] = x[:, 1]
    if save_gif_to is None:
        while True:
            for t in range(img_tensor.shape[0]):
                    plt.imshow(to_img(img_tensor[t]))
                    plt.pause(0.01)
    else:
        img_list = []
        for t in range(img_tensor.shape[0]):
            img_list.append(to_img(img_tensor[t]))
        img_list[0].save(save_gif_to, save_all=True, append_images=img_list[1:], loop=0)
        print(f'Save frames to [{save_gif_to}].')


def load_matlab_mat(file_name: str) -> Dict:
    '''
    :param file_name: path of the matlab's mat file
    :type file_name: str
    :return: a dict whose keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
    :rtype: Dict
    '''
    events = scipy.io.loadmat(file_name)
    return {
        'x': events['x'].squeeze(),
        'y': events['y'].squeeze(),
        'p': events['pol'].squeeze(),
        't': events['ts'].squeeze()
    }


def load_raw(file_name: str) -> Dict:
    '''
    :param file_name: path of the raw file
    :type file_name: str
    :return: a dict whose keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
    :rtype: Dict
    '''

    df = pd.read_csv(f'./data/{file_name}', names=['x', 'y', 'p', 't'], sep=',')

    return {
        'x': np.array(df['x']),
        'y': np.array(df['y']),
        'p': np.array(df['p']),
        't': np.array(df['t'])
    }


def load_npz_frames(file_name: str) -> np.ndarray:
    '''
    :param file_name: path of the npz file that saves the frames
    :type file_name: str
    :return: frames
    :rtype: np.ndarray
    '''

    return np.load(file_name, allow_pickle=True)['frames']


def integrate_events_segment_to_frame(events: Dict, H: int, W: int, j_l: int=0, j_r: int = -1) -> np.ndarray:
    '''
    :param events: a dict whose keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
    :type events: Dict
    :param H: height of the frame
    :type H: int
    :param W: weight of the frame
    :type W: int
    :param j_l: the start index of the integral interval, which is included
    :type j_l: int
    :param j_r: the right index of the integral interval, which is not included
    :type j_r:
    :return: frames
    :rtype: np.ndarray
    Denote a two channels frame as :math:`F` and a pixel at :math:`(p, x, y)` as :math:`F(p, x, y)`, the pixel value is integrated from the events data whose indices are in :math:`[j_{l}, j_{r})`:
    .. math::
        F(p, x, y) = \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})
    where :math:`\lfloor \cdot \rfloor` is the floor operation, :math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` is an indicator function and it equals 1 only when :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`.
    '''

    frame = np.zeros(shape=[2, H*W])
    x = events['x'][j_l: j_r].astype(int)
    y = events['y'][j_l: j_r].astype(int)
    p = events['p'][j_l: j_r]
    mask = []
    mask.append(p==0)
    mask.append(np.logical_not(mask[0]))

    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_number_per_pos = np.bincount(position)
        frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
    
    return frame.reshape((2, H, W))


def cal_fixed_frames_number_segment_index(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    '''
    :param events_t: events' t
    :type events_t: numpy.ndarray
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :return: a tuple ``(j_l, j_r)``
    :rtype: tuple
    Denote ``frames_num`` as :math:`M`, if ``split_by`` is ``'time'``, then
    .. math::
        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
    If ``split_by`` is ``'number'``, then
    .. math::
        j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
        j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
    '''

    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N

    elif split_by == 'time':
        dt = (events_t[-1] - events_t[0]) // frames_num
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1

        j_r[-1] = N
    else:
        raise NotImplementedError

    return j_l, j_r


def integrate_events_by_fixed_frames_number(events: Dict, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    '''
    :param events: a dict whose keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
    :type events: Dict
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :return: frames
    :rtype: np.ndarray
    Integrate events to frames by fixed frames number. See :class:`cal_fixed_frames_number_segment_index` and :class:`integrate_events_segment_to_frame` for more details.
    '''
    j_l, j_r = cal_fixed_frames_number_segment_index(events['t'], split_by, frames_num)
    frames = np.zeros([frames_num, 2, H, W])
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(events, H, W, j_l[i], j_r[i])
    return frames


def integrate_events_file_to_frames_file_by_fixed_frames_number(loader: Callable, events_np_file: str, output_dir: str, split_by: str, frames_num: int, H: int, W: int, print_save: bool = False) -> None:
    '''
    :param loader: a function that can load events from `events_np_file`
    :type loader: Callable
    :param events_np_file: path of the events np file
    :type events_np_file: str
    :param output_dir: output directory for saving the frames
    :type output_dir: str
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :param print_save: If ``True``, this function will print saved files' paths.
    :type print_save: bool
    :return: None
    Integrate a events file to frames by fixed frames number and save it. See :class:`cal_fixed_frames_number_segment_index` and :class:`integrate_events_segment_to_frame` for more details.
    '''
    fname = os.path.join(output_dir, os.path.basename(events_np_file))
    np_savez(fname, frames=integrate_events_by_fixed_frames_number(loader(events_np_file), split_by, frames_num, H, W))
    if print_save:
        print(f'Frames [{fname}] saved.')


def integrate_events_by_fixed_duration(events: Dict, duration: int, H: int, W: int) -> np.ndarray:
    '''
    :param events: a dict whose keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
    :type events: Dict
    :param duration: the time duration of each frame
    :type duration: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :return: frames
    :rtype: np.ndarray
    Integrate events to frames by fixed time duration of each frame.
    '''

    t = events['t']
    N = t.size

    frames = []
    left = 0
    right = 0
    while True:
        t_l = t[left]
        while True:
            if right == N or t[right] - t_l > duration:
                break
            else:
                right += 1
        # integrate from index [left, right)
        frames.append(np.expand_dims(integrate_events_segment_to_frame(events, H, W, left, right), 0))

        left = right

        if right == N:
            return np.concatenate(frames)

def integrate_events_file_to_frames_file_by_fixed_duration(loader: Callable, events_np_file: str, output_dir: str, duration: int, H: int, W: int, print_save: bool = False) -> None:
    '''
    :param loader: a function that can load events from `events_np_file`
    :type loader: Callable
    :param events_np_file: path of the events np file
    :type events_np_file: str
    :param output_dir: output directory for saving the frames
    :type output_dir: str
    :param duration: the time duration of each frame
    :type duration: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :param print_save: If ``True``, this function will print saved files' paths.
    :type print_save: bool
    :return: None
    Integrate events to frames by fixed time duration of each frame.
    '''

    frames = integrate_events_by_fixed_duration(loader(events_np_file), duration, H, W)
    fname, _ = os.path.splitext(os.path.basename(events_np_file))
    fname = os.path.join(output_dir, f'{fname}_{frames.shape[0]}.npz')
    np_savez(fname, frames=frames)
    if print_save:
        print(f'Frames [{fname}] saved.')
    return frames.shape[0]

def save_frames_to_npz_and_print(fname: str, frames):
    np_savez(fname, frames=frames)
    print(f'Frames [{fname}] saved.')

def create_same_directory_structure(source_dir: str, target_dir: str) -> None:
    '''
    :param source_dir: Path of the directory that be copied from
    :type source_dir: str
    :param target_dir: Path of the directory that be copied to
    :type target_dir: str
    :return: None
    Create the same directory structure in ``target_dir`` with that of ``source_dir``.
    '''
    for sub_dir_name in os.listdir(source_dir):
        source_sub_dir = os.path.join(source_dir, sub_dir_name)
        if os.path.isdir(source_sub_dir):
            target_sub_dir = os.path.join(target_dir, sub_dir_name)
            os.mkdir(target_sub_dir)
            print(f'Mkdir [{target_sub_dir}].')
            create_same_directory_structure(source_sub_dir, target_sub_dir)


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.random.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''

    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(tqdm.tqdm(origin_dataset)):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        
        label_idx[y].append(i)
    
    train_idx = []
    test_idx = []

    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])
    
    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def pad_sequence_collate(batch: list):
    '''
    :param batch: a list of samples that contains ``(x, y)``, where ``x.shape=[T, *]`` and ``y`` is the label
    :type batch: list
    :return: batched samples, where ``x`` is padded with the same length
    :rtype: tuple
    This function can be use as the ``collate_fn`` for ``DataLoader`` to process the dataset with variable length, e.g., a ``NeuromorphicDatasetFolder`` with fixed duration to integrate events to frames.
    Here is an example:
    .. code-block:: python
        class RandomLengthDataset(torch.utils.data.Dataset):
            def __init__(self, n=1000):
                super().__init__()
                self.n = n
            def __getitem__(self, i):
                return torch.rand([random.randint(1, 10), 28, 28]), random.randint(0, 10)
            def __len__(self):
                return self.n
        loader = torch.utils.data.DataLoader(RandomLengthDataset(n=32), batch_size=16, collate_fn=pad_sequence_collate)
        for x, y, z in loader:
            print(x.shape, y.shape, z)
    And the outputs are:
    .. code-block:: bash
        torch.Size([10, 16, 28, 28]) torch.Size([16]) tensor([ 1,  9,  3,  4,  1,  2,  9,  7,  2,  1,  5,  7,  4, 10,  9,  5])
        torch.Size([10, 16, 28, 28]) torch.Size([16]) tensor([ 1,  8,  7, 10,  3, 10,  6,  7,  5,  9, 10,  5,  9,  6,  7,  6])
    '''
    x_list = []
    x_len_list = []
    y_list = []
    for x, y in batch:
        x_list.append(torch.as_tensor(x))
        x_len_list.append(x.shape[0])
        y_list.append(y)

    return torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True), torch.as_tensor(y_list), torch.as_tensor(x_len_list)


def padded_sequence_mask(sequence_len: torch.Tensor, T=None):
    '''
    :param sequence_len: a tensor ``shape = [N]`` that contains sequences lengths of each batch element
    :type sequence_len: torch.Tensor
    :param T: The maximum length of sequences. If ``None``, the maximum element in ``sequence_len`` will be seen as ``T``
    :type T: int
    :return: a bool mask with shape = [T, N], where the padded position is ``False``
    :rtype: torch.Tensor
    Here is an example:
    .. code-block:: python
        x1 = torch.rand([2, 6])
        x2 = torch.rand([3, 6])
        x3 = torch.rand([4, 6])
        x = torch.nn.utils.rnn.pad_sequence([x1, x2, x3])  # [T, N, *]
        print('x.shape=', x.shape)
        x_len = torch.as_tensor([x1.shape[0], x2.shape[0], x3.shape[0]])
        mask = padded_sequence_mask(x_len)
        print('mask.shape=', mask.shape)
        print('mask=\\n', mask)
    And the outputs are:
    .. code-block:: bash
        x.shape= torch.Size([4, 3, 6])
        mask.shape= torch.Size([4, 3])
        mask=
         tensor([[ True,  True,  True],
                [ True,  True,  True],
                [False,  True,  True],
                [False, False,  True]])
    '''
    if T is None:
        T = sequence_len.max().item()
    N = sequence_len.numel()
    device_id = sequence_len.get_device()

    if device_id >= 0 and cupy is not None:
        mask = torch.zeros([T, N], dtype=bool, device=sequence_len.device)
        with cupy.cuda.Device(device_id):
            T = cupy.asarray(T)
            N = cupy.asarray(N)
            sequence_len, mask, T, N = cu_kernel_opt.get_contiguous(sequence_len.to(torch.int), mask, T, N)
            kernel_args = [sequence_len, mask, T, N]
            kernel = cupy.RawKernel(padded_sequence_mask_kernel_code, 'padded_sequence_mask_kernel', options=('-use_fast_math',), backend=('-use_fast_math',))
            blocks = cu_kernel_opt.cal_blocks(N)
            kernel(
                (blocks,), (1024,),
                cu_kernel_opt.wrap_args_to_raw_kernel(
                    device_id,
                    *kernel_args
                )
            )
            return mask

    else:
        t_seq = torch.arange(0, T).unsqueeze(1).repeat(1, N).to(sequence_len)  # [T, N]
        return t_seq < sequence_len.unsqueeze(0).repeat(T, 1)


class NeuromorphicDatasetFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        '''
        :param root: root path of the dataset
        :type root: str
        :param train: whether use the train set. Set ``True`` or ``False`` for those datasets provide train/test
            division, e.g., DVS128 Gesture dataset. If the dataset does not provide train/test division, e.g., CIFAR10-DVS,
            please set ``None`` and use :class:`~split_to_train_test_set` function to get train/test set
        :type train: bool
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number`
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param custom_integrate_function: a user-defined function that inputs are ``events, H, W``.
            ``events`` is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
            ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, H=128 and W=128 for the DVS128 Gesture dataset.
            The user should define how to integrate events to frames, and return frames.
        :type custom_integrate_function: Callable
        :param custom_integrated_frames_dir_name: The name of directory for saving frames integrating by ``custom_integrate_function``.
            If ``custom_integrated_frames_dir_name`` is ``None``, it will be set to ``custom_integrate_function.__name__``
        :type custom_integrated_frames_dir_name: str or None
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable
        The base class for neuromorphic dataset. Users can define a new dataset by inheriting this class and implementing
        all abstract methods. Users can refer to :class:`spikingjelly.datasets.dvs128_gesture.DVS128Gesture`.
        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``.
        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`cal_fixed_frames_number_segment_index` for
            more details.
        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration.
        If ``data_type == 'frame'``, ``frames_number`` is ``None``, ``duration`` is ``None``, and ``custom_integrate_function`` is not ``None``:
            events will be integrated by the user-defined function and saved to the ``custom_integrated_frames_dir_name`` directory in ``root`` directory.
        '''

        events_np_root = os.path.join(root, 'events_np')
        extract_root = os.path.join(root, 'data')

        if not os.path.exists(events_np_root):
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}]')
        print(f'Start to convert the origin data from [{extract_root}] to [{events_np_root}] in np.ndarray format.')
        self.create_events_np_files(extract_root, events_np_root)
    
        H, W = self.get_H_W()

        if data_type == 'event':
            _root = events_np_root
            _loader = np.load
            _transform = transform
            _target_transform = target_transform

        elif data_type == 'frame':
            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by == 'time' or split_by == 'number'

                frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')

                if os.path.exists(frames_np_root):
                    print(f'The directory {frames_np_root} already exists')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}]')

                    create_same_directory_structure(events_np_root, frames_np_root)

                    t_ckp = time.time()

                    for e_root, e_dirs, e_files in os.walk(events_np_root):
                        if e_files.__len__() > 0:
                            output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                            for e_file in e_files:
                                events_np_file = os.path.join(e_root, e_file)
                                print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                # tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number, self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W, True)
                                integrate_events_file_to_frames_file_by_fixed_frames_number(self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W, True)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
                
                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')

                if os.path.exists(frames_np_root):
                    print(f'The directory {frames_np_root} already exists')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}]')

                    create_same_directory_structure(events_np_root, frames_np_root)

                    t_ckp = time.time()
                    for e_root, e_dirs, e_files in os.walk(events_np_root):
                        if e_files.__len__() > 0:
                            output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                            for e_file in e_files:
                                events_np_file = os.path.join(e_root, e_file)
                                print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                # tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration, self.load_events_np, events_np_file, output_dir, duration, H, W, True)
                                integrate_events_file_to_frames_file_by_fixed_duration(self.load_events_np, events_np_file, output_dir, duration, H, W, True) 

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif custom_integrate_function is not None:
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    for e_root, e_dirs, e_files in os.walk(events_np_root):
                        if e_files.__len__() > 0:
                            output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                            for e_file in e_files:
                                events_np_file = os.path.join(e_root, e_file)
                                print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                # tpe.submit(save_frames_to_npz_and_print, os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W))
                                save_frames_to_npz_and_print(os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W))

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform


            else:
                raise ValueError('At least one of "frames_number", "duration" and "custom_integrate_function" should not be None.')

        if train is not None:
            if train:
                _root = os.path.join(_root, 'train')
            else:
                _root = os.path.join(_root, 'test')

        super().__init__(root=_root, loader=_loader, extensions=('.npz', ), transform=_transform,
                         target_transform=_target_transform)
    

    @staticmethod
    @abstractmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None
        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        pass

    @staticmethod
    @abstractmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        pass

    @staticmethod
    def load_events_np(fname: str):
        '''
        :param fname: file name
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        This function defines how to load a sample from `events_np`. In most cases, this function is `np.load`.
        But for some datasets, e.g., ES-ImageNet, it can be different.
        '''
        return np.load(fname)


class FYPDataset(NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the raw data file
        :type file_name: str
        :return: a dict where the keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function is written by referring to https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/__init__.py
        '''
        '''eventually make it so that the filename is sent, but for testing purposes, we only use this one file'''
        df = pd.read_csv("./data/collapse_8mm_bias_fo_1637_chris_1.csv", names=["x", "y", "p", "t"], sep=",")
        
        data = {
            'x': np.array(df['x']),
            'y': np.array(df['y']),
            'p': np.array(df['p']),
            't': np.array(df['t'])
        }

        return data


    @staticmethod
    def split_raw_to_np(fname: str, raw_file: str, csv_file: str, output_dir: str):
        '''
        :param fname: the filename of the file to split
        :type fname: str
        :param raw_file: the raw data that we want to split
        :type raw_file: str
        :param csv_file: the labels of the raw data file
        :type csv_file: str
        :param output_dir: the directory of the output files
        :type output_dir: str

        this function defines how to split a raw file into several smaller files which contain information 
        from one event as defined in our labelled csv_file
        '''
        global np_savez
        events = FYPDataset.load_origin_data(raw_file)
        print(f'Start to split [{raw_file}] to samples')

        # read csv file and get time stamps and label of each sample
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        label_file_num = {}
        for i in range(csv_data.shape[0]):
            print(f'{i}th iteration of np splitting for {raw_file}')
            label = csv_data[i][0]
            
            # make directory for labels when we come across the label for the first time
            if label not in label_file_num.keys():
                label_file_num[label] = 0
                os.mkdir(os.path.join(output_dir, str(label)))

            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np_savez(file_name, x=events['x'][mask], y=events['y'][mask], p=events['p'][mask], t=events['t'][mask])
            print(f'[{fname}] saved')
            label_file_num[label] += 1

    
    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: the root directory of the dataset
        :type extract_root: str
        :param events_np_root: the root directory of the split files
        :type events_np_root: str

        this function is the driver function which will go through all files in the dataset and split them
        into individual samples in the train and test sets
        '''
        raw_dir = os.path.join(extract_root)
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')

        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
            os.mkdir(test_dir)
        
        with open(os.path.join(raw_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
            os.path.join(raw_dir, 'trials_to_test.txt')) as trials_to_test_txt:

            # use multi-thread to accelerate the process
            
            t_ckp = time.time()

            for fname in trials_to_train_txt.readlines():
                fname = fname.strip()
                print(fname, end=',')
                if fname.__len__() > 0:
                    raw_file = os.path.join(raw_dir, fname)
                    fname = os.path.splitext(fname)[0]
                    # tpe.submit(FYPDataset.split_raw_to_np, fname, raw_file, os.path.join(raw_dir, fname), train_dir)
                    FYPDataset.split_raw_to_np(fname, raw_file, os.path.join(raw_dir, f'{fname}_labels.txt'), train_dir)

            for fname in trials_to_test_txt.readlines():
                fname = fname.strip()
                print(fname, end=',')
                if fname.__len__() > 0:
                    raw_file = os.path.join(raw_dir, fname)
                    fname = os.path.splitext(fname)[0]
                    # tpe.submit(FYPDataset.split_raw_to_np, fname, raw_file, os.path.join(raw_dir, fname), test_dir)
                    FYPDataset.split_raw_to_np(fname, raw_file, os.path.join(raw_dir, f'{fname}_labels.txt'), test_dir)
                
            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All raw files have been split to samples and saved into [{train_dir, test_dir}]')



    @staticmethod
    def get_H_W() -> Tuple:
        '''
        this function returns the height and width of the frame
        '''
        return 480, 640