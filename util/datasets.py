import os
import PIL

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_pretrain_dataset(args, category, transform):
    class CategoryFolder(Dataset):
        def __init__(self, data_dir=None, transforms=None):
            self.imgs_path = [os.path.join(data_dir, n) for n in os.listdir(data_dir)]
            self.transforms = transforms

        def __len__(self):
            return len(self.imgs_path)

        def __getitem__(self, idx):
            img = PIL.Image.open(self.imgs_path[idx]).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img, 0

    root = os.path.join(args.data_path, category)
    dataset = CategoryFolder(data_dir=root, transforms=transform)
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            scale=(0.2, 1.0),
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    size = 292
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BILINEAR if args.interpolation == 'bilinear' else
                          PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

