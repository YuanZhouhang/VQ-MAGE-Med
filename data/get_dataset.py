import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

CATEGORY = {
    'ham10000': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
    # 'ODIR': ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other'],
    'ODIR': ['AMD', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Other'],
}

class SingleDataset(Dataset):
    def __init__(self, name, root, transform, backbone_model, device):
        self.name = name
        self.root = root
        self.transform = transform
        if backbone_model is not None:
            self.backbone_model = backbone_model.to(device)
        else:
            self.backbone_model = None

        self.categories = CATEGORY[name]
        self.sample_roots = []
        for category in self.categories:
            self.sample_roots.extend(glob.glob(f"{self.root}/{category}/*"))

        if os.path.exists(os.path.join(self.root, 'features.pt')):
            self.image_list = torch.load(os.path.join(self.root, 'features.pt'))
            self.label_list = torch.load(os.path.join(self.root, 'labels.pt'))
        else:
            self.image_list = []
            self.label_list = []
            for image_path in tqdm(self.sample_roots):
                image = self.transform(Image.open(image_path)).to(device)
                if self.backbone_model is not None:
                    feature = self.backbone_model(image.unsqueeze(0))[-1].squeeze(0).cpu()
                else:
                    feature = image.cpu()
                label = self.categories.index(image_path.split(os.sep)[-2])
                self.image_list.append(feature)
                self.label_list.append(label)
            self.image_list = torch.stack(self.image_list)
            self.label_list = torch.tensor(self.label_list)
            torch.save(self.image_list, os.path.join(self.root, 'features.pt'))
            torch.save(self.label_list, os.path.join(self.root, 'labels.pt'))

        # image_list = [self.transform(Image.open(image_path)).to(device) for image_path in self.sample_roots]
        # self.feature_list = [self.backbone_model(image.unsqueeze(0))[-1].squeeze(0) for image in image_list].cpu()
        # self.label_list = [self.categories.index(image_path.split(os.sep)[-2]) for image_path in self.sample_roots].cpu()

    def __getitem__(self, idx):
        # image_path = self.sample_roots[idx]
        # label = self.categories.index(image_path.split(os.sep)[-2])
        # image = self.transform(Image.open(image_path))
        # feature = self.backbone_model(image.unsqueeze(0))[-1].squeeze(0)
        return self.image_list[idx], self.label_list[idx]
        # return image, label

    def __len__(self):
        return len(self.sample_roots)

class CombineDataset(Dataset):
    def __init__(self, name, root1, root2, transform, backbone_model, device):
        self.name = name
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        if backbone_model is not None:
            self.backbone_model = backbone_model.to(device)
        else:
            self.backbone_model = None
        self.device = device

        self.categories = CATEGORY[name]
        self.sample_roots = []
        for category in self.categories:
            self.sample_roots.extend(glob.glob(f"{self.root1}/{category}/*"))
            self.sample_roots.extend(glob.glob(f"{self.root2}/{category}/*"))

        if os.path.exists(os.path.join(self.root2, 'features.pt')):
            self.image_list = torch.load(os.path.join(self.root2, 'features.pt'))
            self.label_list = torch.load(os.path.join(self.root2, 'labels.pt'))
        else:
            self.image_list = []
            self.label_list = []
            for image_path in tqdm(self.sample_roots):
                image = self.transform(Image.open(image_path)).to(device)
                if self.backbone_model is not None:
                    feature = self.backbone_model(image.unsqueeze(0))[-1].squeeze(0).cpu()
                else:
                    feature = image.cpu()
                label = self.categories.index(image_path.split(os.sep)[-2])
                self.image_list.append(feature)
                self.label_list.append(label)
            self.image_list = torch.stack(self.image_list)
            self.label_list = torch.tensor(self.label_list)
            torch.save(self.image_list, os.path.join(self.root2, 'features.pt'))
            torch.save(self.label_list, os.path.join(self.root2, 'labels.pt'))

    def __getitem__(self, idx):
        # image_path = self.sample_roots[idx]
        # label = self.categories.index(image_path.split(os.sep)[-2])
        # image = self.transform(Image.open(image_path)).to(self.device)
        # feature = self.backbone_model(image.unsqueeze(0))[-1].squeeze(0)
        # return feature, label
        return self.image_list[idx], self.label_list[idx]

    def __len__(self):
        return len(self.sample_roots)

