from re import M
from tkinter import W
from tkinter.filedialog import test
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
from torchvision.datasets.utils import extract_archive
from torchvision.datasets import DatasetFolder, utils
from concurrent.futures import ThreadPoolExecutor
from spikingjelly import configure
import pandas as pd
from abc import abstractmethod
import scipy.io
import struct
import torch.utils.data
import os
import time
import multiprocessing
from torchvision import transforms
import torc
from matplotlib import pyplot as plt
import math
import tqdm
import shutil
from spikingjelly.datasets import *

np_savez = np.savez_compressed


# dataset folder stuff
'''
NeuromorphicDatasetFolder.py version 0.0.1
adjustments to NeuromorphicDatasetFolder class from spikingjelly module
adjusted to work with raw data provided by ASTRI and our own raw data.
'''

class NDF(DatasetFolder):
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
            Here is an example from SpikingJelly's tutorials:
            .. code-block:: python
                from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
                from typing import Dict
                import numpy as np
                import spikingjelly.datasets as sjds
                def integrate_events_to_2_frames_randomly(events: Dict, H: int, W: int):
                    index_split = np.random.randint(low=0, high=events['t'].__len__())
                    frames = np.zeros([2, 2, H, W])
                    frames[0] = sjds.integrate_events_segment_to_frame(events, H, W, 0, index_split)
                    frames[1] = sjds.integrate_events_segment_to_frame(events, H, W, index_split, events['t'].__len__())
                    return frames
                root_dir = 'D:/datasets/DVS128Gesture'
                train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)
                from spikingjelly.datasets import play_frame
                frame, label = train_set[500]
                play_frame(frame)
        '''

        events_np_root = os.path.join(root, 'events_np')
        extract_root = os.path.join(root)

        # Now let us convert the origin binary files to npz files
        if not os.path.isdir(events_np_root):
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}].')
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
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    # create the same directory structure
                    create_same_directory_structure(events_np_root, frames_np_root)

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                        print(f'1 Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(extract_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number, self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W, True)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')

                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                        print(f'2 Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(extract_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration, self.load_events_np, events_np_file, output_dir, duration, H, W, True)

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
                    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                        print(f'3 Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(extract_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(
                                        f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    tpe.submit(save_frames_to_npz_and_print, os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W))

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
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        pass

    @staticmethod
    @abstractmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        pass

    @staticmethod
    @abstractmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None
        This function defines how to extract download files.
        '''
        pass

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
            For example, this function returns ``(640,480)`` for our dataset
        :rtype: tuple
        '''
        pass

    @staticmethod
    def load_events_np(fname: str):
        '''
        :param fname: file name
        :return: a dict whose keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
        This function defines how to load a sample from `events_np`. In most cases, this function is `np.load`.
        But for some datasets, e.g., ES-ImageNet, it can be different.
        '''
        return np.load(fname)

class EventDataset(NDF):
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
        """
        The DVS128 Gesture dataset, which is proposed by `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>`_.
        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.
        """
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
        print('here')
        events = EventDataset.load_origin_data(raw_file)
        print(f'Start to split [{raw_file}] to samples')

        # read csv file and get time stamps and label of each sample
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',')

        label_file_num = [0] * 4
        for i in range(csv_data.shape[0]):
            label = csv_data[i][0] - 1
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
        train_dir = os.path.join(extract_root, 'train')
        test_dir = os.path.join(extract_root, 'test')

        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
            os.mkdir(test_dir)

            for label in range(4):
                os.mkdir(os.path.join(train_dir, str(label)))
                os.mkdir(os.path.join(test_dir, str(label)))
        
        with open(os.path.join(raw_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
            os.path.join(raw_dir, 'trials_to_test.txt')) as trials_to_test_txt:

            # use multi-thread to accelerate the process
            
            t_ckp = time.time()
        
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                print(f'4 Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}]')

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    print(fname, end=',')
                    if fname.__len__() > 0:
                        raw_file = os.path.join(raw_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(EventDataset.split_raw_to_np, fname, raw_file, os.path.join(raw_dir, fname), train_dir)
                    
                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    print(fname, end=',')
                    if fname.__len__() > 0:
                        raw_file = os.path.join(raw_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(EventDataset.split_raw_to_np, fname, raw_file, os.path.join(raw_dir, fname), test_dir)
                
            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All raw files have been split to samples and saved into [{train_dir, test_dir}]')


    @staticmethod
    def get_H_W() -> Tuple:
        '''
        this function returns the height and width of the frame
        '''
        return 480, 640