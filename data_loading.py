import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import default_collate

class DataLoaderCreator:

    cinic_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
    cinic_std_RGB = [0.24205776, 0.23828046, 0.25874835]

    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])

    random_rotation_transform = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])

    random_flipping_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])

    random_blur_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])

    random_all = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])

    def __init__(self, seed=123):
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def worker_init_fn(self, worker_id):
        np.random.seed(self.seed)

    def collate_fn(self, batch):
        cutmix = v2.CutMix(num_classes=10)
        return cutmix(*default_collate(batch))

    def create_data_loader(self, data_dir, batch_size, transform=standard_transform, shuffle=True):
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, worker_init_fn=self.worker_init_fn)
        return data_loader
    
    def create_data_loader_cutmix(self, data_dir, batch_size, transform=standard_transform, shuffle=True):
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, worker_init_fn=self.worker_init_fn, collate_fn=self.collate_fn)
        return data_loader