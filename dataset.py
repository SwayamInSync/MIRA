import random
from glob import glob
import numpy as np
import torch
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor


class TrainObjaverseDataset(Dataset):
    def __init__(self, dataset_dir, transforms, k=4, workers=4):
        self.dataset_dir = dataset_dir
        self.objects_path = glob(f"{dataset_dir}/*")
        self.k = k
        self.transforms = transforms
        self.workers = workers
        self._preload_paths()

    def _preload_paths(self):
        self.image_paths = []
        self.extrinsic_paths = []
        for obj_path in self.objects_path:
            render_images = sorted(glob(f"{obj_path}/*.png"))
            extrinsics = sorted(glob(f"{obj_path}/*.npy"))
            self.image_paths.append(render_images)
            self.extrinsic_paths.append(extrinsics)

    def __len__(self):
        return len(self.objects_path)*32

    def _load_data(self, params):
        obj_idx, index = params
        image = cv.imread(self.image_paths[obj_idx][index])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.transforms(Image.fromarray(image))
        cam_extrinsic = np.load(self.extrinsic_paths[obj_idx][index])
        cam_extrinsic = np.vstack([cam_extrinsic, np.array([[0, 0, 0, 1]],
                                                           dtype=np.float32)])
        cam_extrinsic = torch.tensor(cam_extrinsic, dtype=torch.float32).view(-1)
        return image, cam_extrinsic

    def __getitem__(self, idx):
        obj_index = idx // 32
        image_index = idx % 32
        supervision_indices = random.sample(range(32), self.k-1)
        while image_index in supervision_indices:
            supervision_indices = random.sample(range(32), self.k-1)

        supervision_indices.append(image_index)
        supervision_indices = supervision_indices[::-1]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = executor.map(self._load_data, [(obj_index, supervision_indices[i]) for i in range(self.k)])
        imgs, src_cam = zip(*results)

        imgs = torch.stack(imgs, dim=0)
        src_cam = torch.stack(src_cam, dim=0)
        return imgs, src_cam


class ValidObjaverseDataset(Dataset):
    def __init__(self, dataset_dir, transforms):
        self.dataset_dir = dataset_dir
        self.objects_path = glob(f"{dataset_dir}/*")
        self.transforms = transforms
        self._preload_paths()

    def _preload_paths(self):
        self.data = []
        for obj_path in self.objects_path:
            render_images = sorted(glob(f"{obj_path}/*.png"))
            extrinsics = sorted(glob(f"{obj_path}/*.npy"))
            self.data.extend(list(zip(render_images, extrinsics)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, cam = self.data[idx]
        image = cv.imread(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.transforms(Image.fromarray(image))
        cam_extrinsic = np.load(cam)
        cam_extrinsic = np.vstack([cam_extrinsic, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        cam_extrinsic = torch.tensor(cam_extrinsic, dtype=torch.float32).view(-1)
        return image, cam_extrinsic


def get_loader(dataset_path, transforms, batch_size, is_valid=False):
    if not is_valid:
        ds = TrainObjaverseDataset(dataset_path, transforms)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        return loader
    ds = ValidObjaverseDataset(dataset_path, transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader



if __name__ == "__main__":
    from torchvision import transforms
    from config import Config

    c = Config.from_json("train_config.json")
    train_transforms = transforms.Compose([
        transforms.Resize((c.source_size, c.source_size)),
        transforms.ToTensor()
    ])
    train_ds = TrainObjaverseDataset("temp_data/render", train_transforms)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    imgs, cam = next(iter(loader))
    print("#"*10, "Training Dataset", "#"*10)
    print(len(loader))
    print(imgs.shape, imgs.max(), imgs.min())
    print(cam.shape)

    print("#" * 10, "Valid Dataset", "#" * 10)
    valid_transforms = transforms.Compose([
        transforms.Resize((c.source_size, c.source_size)),
        transforms.ToTensor()
    ])
    ds = ValidObjaverseDataset("temp_data/render", valid_transforms)
    loader = DataLoader(ds, batch_size=1)
    print(len(loader))
    imgs, cam = next(iter(loader))
    print(imgs.shape, imgs.max(), imgs.min())
    print(cam.shape)