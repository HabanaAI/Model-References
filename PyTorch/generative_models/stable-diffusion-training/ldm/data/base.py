import math
from abc import abstractmethod

import torch
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import os
import numpy as np
#import cv2
from PIL import Image
import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, file_path: str, rank, world_size,**kwargs):
        super().__init__()
        self.file_path = file_path
        self.folder_list = []
        self.file_list = []
        self.txt_list = []
        self.info = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end = self.info['end']
        #self.rank = int(rank)
        self.rank = get_rank()
        self.world_size = world_size
        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = self.start + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        self.num_records = self.iter_end - self.iter_start
        self.valid_ids = [i for i in range(self.iter_end)]
        #self.num_records = self.end - self.start
        #self.valid_ids = [i for i in range(self.end)]
        self.transforms = self.get_transforms(kwargs)
        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples (rank: {self.rank}/{world_size})\nstart {self.iter_start} end {self.iter_end}')

    def get_transforms(self,dataset_config):
        import torchvision
        from ldm.util import instantiate_from_config
        from einops import rearrange
        if 'image_transforms' in dataset_config:
            image_transforms = [instantiate_from_config(tt) for tt in dataset_config['image_transforms']]
        else:
            image_transforms = []

        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        return torchvision.transforms.Compose(image_transforms)

       #  if 'transforms' in dataset_config:
       #      transforms_config = OmegaConf.to_container(dataset_config.transforms)
       #  else:
       #      transforms_config = dict()

       #  transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
       #          if transforms_config[dkey] != 'identity' else identity
       #          for dkey in transforms_config}
       #  img_key = dataset_config.get('image_key', 'jpeg')
       #  transform_dict.update({img_key: image_transforms})

    def __len__(self):
        return self.iter_end - self.iter_start
        #return self.end - self.start

    def __iter__(self):
        #sample_iterator = self._sample_generator(self.start, self.end)
        sample_iterator = self._sample_generator(self.iter_start, self.iter_end)
        return sample_iterator

    def _sample_generator(self, start, end):
        for idx in range(start, end):
            file_name = self.file_list[idx]
            txt_name = self.txt_list[idx]
            f_ = open(txt_name, 'r')
            txt_ = f_.read()
            f_.close()
            #image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), 1)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = torch.from_numpy(image) / 255
            image = Image.open(file_name).convert('RGB')
            image = self.transforms(image)
            yield {"caption": txt_, "image":image}


    def _get_file_info(self, file_path):
        info = \
        {
            "start": 1,
            "end": 0,
        }
        self.folder_list = [file_path + i for i in os.listdir(file_path) if '.' not in i]
        for folder in self.folder_list:
            files = [folder + '/' + i for i in os.listdir(folder) if 'jpg' in i]
            txts = [k.replace('jpg', 'txt') for k in files]
            self.file_list.extend(files)
            self.txt_list.extend(txts)
        info['end'] = len(self.file_list)
        # with open(file_path, 'r') as fin:
        #     for _ in enumerate(fin):
        #         info['end'] += 1
        # self.txt_list = [k.replace('jpg', 'txt') for k in self.file_list]
        return info

class PRNGMixin(object):
    """
    Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing.
    """
    @property
    def prng(self):
       currentpid = os.getpid()
       if getattr(self, "_initpid", None) != currentpid:
          self._initpid = currentpid
          self._prng = np.random.RandomState()
       return self._prng

