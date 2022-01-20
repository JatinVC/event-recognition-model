'''
NeuromorphicDatasetFolder.py version 0.0.1
adjustments to NeuromorphicDatasetFolder class from spikingjelly module
adjusted to work with raw data provided by ASTRI and our own raw data.
'''
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torch.utils.data
import os
import torch
from torchvision import transforms
import numpy as np
from abc import abstractmethod

class NeuromorphicDatasetFolder(DatasetFolder):
    def __init__(self,
                root: str,
                train: bool=None,
                split_by: str=None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None
                ) -> None:
                '''
                metavision_viewer.exe:param root: root path of the dataset folder
                :type root: str
                
                :param train: whether use the train set, if False, then use test set
                :type train: bool

                :param split_by: `time` or `number`
                :type split_by: int

                :param transform: a function/transform that takes in
                    a sample and returns a transformed version.
                :type transform: callable

                :param target_transform: a function/transform that takes
                    in the target and transforms it.
                :type target_transform: callable

                The base class for neuromorphic dataset. Users can define a new dataset by inheriting from this class and implementing
                all abstract methods.
                '''

                events_np_root = os.path.join(root)
                if os.path.exists(events_np_root):
                    # get the height and the weight of the dataset
                    _root = events_np_root
                    _loader = np.load
                    _transform = transform
                    _target_transform = target_transform

                    if train is not None:
                        if train:
                            _root = os.path.join(_root, 'train')
                        else:
                            _root = os.path.join(_root, 'test')
                    
                    super().__init__(root=_root, loader=_loader, extensions=('.npz', ), transform=_transform, target_transform=_target_transform)

    @staticmethod
    @abstractmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict where the keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin data
        '''
        pass

    # @staticmethod
    # @abstractmethod
    # def create_events_np_files(extract_root: str, events_np_root: str):
    #     '''
    #     :param extract_root: Root directory path which saves extracted files from downloaded files
    #     :type extract_root: str
    #     :param events_np_root: Root directory path which saves events files in the ``npz`` format
    #     :type events_np_root:
    #     :return: None
    #     This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
    #     '''
    #     pass

    # @staticmethod
    # @abstractmethod
    # def get_H_W() -> Tuple:
    #     '''
    #     :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W`` is the weight of the data.
    #         For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
    #     :rtype: tuple
    #     '''
    #     pass