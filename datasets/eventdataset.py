import os
from torch.utils.data import Dataset, Dataloader
import NeuromorphicDatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
from torchvision.datasets.utils import extract_archive
import multiprocessing
import time
import csv

class EventDataset(NeuromorphicDatasetFolder):
    def __init__(self,
                root: str,
                train: bool=None,
                split_by: str=None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None
                ) -> None:
                '''
                :param root: root path of the dataset folder
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
                assert train is not None
                super().__init__(root, train, split_by, transform, target_transform)
    
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the raw data file
        :type file_name: str
        :return: a dict where the keys are ``['x', 'y', 'p', 't']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function is written by referring to https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/__init__.py
        '''
        csvReader = csv.reader(open(file_name), delimiter=',')
        data = {
            'x': np.array([]),
            'y': np.array([]),
            'p': np.array([]),
            't': np.array([]) 
        }

        for line in csvReader:
            data['x'] = np.append(data['x'], line[0])
            data['y'] = np.append(data['y'], line[1])
            data['p'] = np.append(data['p'], line[2])
            data['t'] = np.append(data['t'], line[3])

        return data