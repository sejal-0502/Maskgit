import os, json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
from time import time
import random
import h5py
from PIL import Image
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torchvision import transforms

# Resizes the image (larger than the target size) and then crops the center with the target size
class RandomResizedCenterCrop(object):
    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation
        self.size = size
        self.fixed_params = None

    def get_params(self, img):
        if self.fixed_params is None:
            width, height = img.size
            area = height * width
            aspect_ratio = width / height

            target_area = random.uniform(*self.scale) * area

            new_width = int(round((target_area * aspect_ratio) ** 0.5))
            new_height = int(round((target_area / aspect_ratio) ** 0.5))
            x1 = (new_width - self.size) // 2
            y1 = (new_height - self.size) // 2
            self.fixed_params = (new_width, new_height, x1, y1)
        return self.fixed_params    

    def __call__(self, img):
        new_width, new_height, x1, y1 = self.get_params(img)
        img = img.resize((new_width, new_height), self.interpolation)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size))

    def reset(self):
        self.fixed_params = None

class MultiHDF5DatasetMultiFrame(Dataset):
    def __init__(self, size, hdf5_paths_file, num_frames, frame_rate=1, aug='resize_center', scale_min=0.15, scale_max=0.5):
        self.size = size
        self.num_frames = num_frames
        #if frame_rate != 1: raise NotImplementedError('frame_rate != 1 not tested yet')
        self.frame_rate = frame_rate  # how many frames to skip, e.g. if data is stored at 10Hz but we want 5Hz, frame_rate=2
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        #self.files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.files = [h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*500, rdcc_nslots=375000) for path in self.hdf5_paths]
        self.lengths = []
        self.file_keys = []
        for file in self.files:
            keys = list(file.keys())
            self.file_keys.append(keys)
            self.lengths.append({key: len(file[key]) for key in keys})

        self.total_length = sum(sum(lengths.values()) for lengths in self.lengths)
        print(f'Total length: {self.total_length}')
        # self.transform = transforms.Compose([transforms.Resize(self.size),
        #                                  transforms.CenterCrop((self.size, self.size)),
        #                                  transforms.ToTensor(),
        #                                  ])
        self.aug = aug
        if self.aug == 'resize_center':
            self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop((self.size, self.size)),
                                         transforms.ToTensor(),
                                         ])
        elif self.aug == 'random_resize_center':
            self.custom_crop = RandomResizedCenterCrop(size=self.size, scale=(scale_min, scale_max))
            self.transform = transforms.Compose([
                                        self.custom_crop,
                                        transforms.ToTensor(),
                                        ])
    
    def __len__(self):
        return self.total_length

    def apply_same_transform_to_all(self, frames, transform):
        return [transform(frame) for frame in frames]
    
    def __getitem__(self, idx):
        file_index = random.randint(0, len(self.files) - 1)
        #h5_file = h5py.File(self.hdf5_paths[file_index], 'r', rdcc_nbytes=1024*1024*1024*500, rdcc_nslots=375000)
        h5_file = self.files[file_index]

        key_index = random.randint(0, len(self.file_keys[file_index]) - 1)
        img_start_index = random.randint(0, self.lengths[file_index][self.file_keys[file_index][key_index]] - (self.num_frames+1)*self.frame_rate)

        key = self.file_keys[file_index][key_index]

        # Log cache stats
        # dataset = h5_file[key]
        # cache_stats = dataset.id.get_access_stats()
        # print(f"File: {self.hdf5_paths[file_index]}, Key: {key}, "
        #       f"Cache Hits: {cache_stats.chunk_cache_hits}, "
        #       f"Cache Misses: {cache_stats.chunk_cache_misses}, "
        #       f"Cache Evictions: {cache_stats.chunk_cache_evictions}")

        #images = [self.transform(Image.fromarray(h5_file[key][img_start_index+i])) for i in range(self.num_frames)]
        images = [Image.fromarray(h5_file[key][img_start_index+i]) for i in range(self.num_frames)]
        
        if self.aug == 'random_resize_center':
            self.custom_crop.reset()
        images = self.apply_same_transform_to_all(images, self.transform)
        # return only first frame
        return images[0]*2 -1

        #return torch.stack(images, dim=0)*2 -1
    
    def close(self):
        for file in self.files:
            file.close()


class MultiHDF5DatasetMultiFrameTest(MultiHDF5DatasetMultiFrame):
    def __init__(self, size, hdf5_paths_file, num_frames, frame_rate=1, aug='resize_center', scale_min=0.15, scale_max=0.5):
        super().__init__(size, hdf5_paths_file, num_frames, frame_rate, aug, scale_min, scale_max)

    def __getitem__(self, idx):
        file_index = random.randint(0, len(self.files) - 1)
        h5_file = h5py.File(self.hdf5_paths[file_index], 'r', rdcc_nbytes=1024*1024*1024*500, rdcc_nslots=1000000)
        while True:
            key_index = random.randint(0, len(self.file_keys[file_index]) - 1)
            random_frame_rate = random.randint(1, self.frame_rate)
            if self.lengths[file_index][self.file_keys[file_index][key_index]] - (self.num_frames+1)*random_frame_rate > 0:
                break
            else:
                print(f'File: {self.hdf5_paths[file_index]}, Key: {self.file_keys[file_index][key_index]} has length {self.lengths[file_index][self.file_keys[file_index][key_index]]}, trying another key')
        img_start_index = random.randint(0, self.lengths[file_index][self.file_keys[file_index][key_index]] - (self.num_frames+1)*random_frame_rate)

        key = self.file_keys[file_index][key_index]
        #images = [self.transform(Image.fromarray(h5_file[key][img_start_index+i])) for i in range(self.num_frames)]
        
        #images = [Image.fromarray(h5_file[key][img_start_index+i]) for i in range(self.num_frames)]
        # images with frame rate
        images = [Image.fromarray(h5_file[key][img_start_index+i*random_frame_rate]) for i in range(self.num_frames)]
        
        if self.aug == 'random_resize_center':
            self.custom_crop.reset()
        images = self.apply_same_transform_to_all(images, self.transform)

        #if self.frame_rate != 1:
        return torch.stack(images, dim=0)*2 -1, random_frame_rate
    
        