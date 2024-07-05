import torch
import torch.nn as nn
import argparse
import sys
sys.path.append('..')
from data.get_dataset import SingleDataset, CombineDataset
from torchvision import transforms
import torchmetrics
from mmpretrain import get_model

class LinearProbe(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(feat_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.linear(x)
        loss = self.criterion(x, y)
        return loss, x

def main():
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('using device:{}'.format(device))

    transform_list = [
        transforms.Resize((int(args.im_size), int(args.im_size))),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)

    if 'swin' in args.backbone:
        backbone_model = get_model(args.backbone, pretrained=True, head=None).to(device)
    else:
        backbone_model = get_model(args.backbone, pretrained=True, head=None, neck=None).to(device)
    for param in backbone_model.parameters():
        param.requires_grad = False

    if args.use_data == 'train':
        train_dataset = SingleDataset(args.dataset, args.root1, transform, backbone_model, device)
    elif args.use_data == 'train+synthetic':
        train_dataset = CombineDataset(args.dataset, args.root1, args.root2, transform, backbone_model, device)
    else:
        raise ValueError('wrong use_data:{}'.format(args.use_data))

    print('train dataset length: ', len(train_dataset))
    # print('feature dim: ', train_dataset[0][0].shape)
    test_dataset = SingleDataset(args.dataset, args.test_root, transform, backbone_model, device)
    print('test dataset length: ', len(test_dataset))

    if args.dataset == 'ham10000':
        num_classes = 7
    elif args.dataset == 'ODIR':
        num_classes = 8

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('train loader length: ', len(train_loader))
    labelk_count = [0 for i in range(num_classes)]
    for i, (x, y) in enumerate(train_loader):
        for j in range(len(y)):
            labelk_count[y[j]] += 1

    print('labelk_count: ', labelk_count)

    model = LinearProbe(train_dataset[0][0].shape[-1], num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(args.epoch):
        print('epoch:{}'.format(epoch))
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            loss, _ = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
            print('train loss:{}'.format(loss.item()))
        scheduler.step()

        model.eval()
        acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
        auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes).to(device)
        correct_pred = [0 for i in range(num_classes)]
        total_pred = [0 for i in range(num_classes)]
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            _, pred = model(x, y)
            acc.update(pred, y)
            auc.update(pred, y)
            for label, prediction in zip(y, pred):
                if label == prediction.argmax():
                    correct_pred[label] += 1
                total_pred[label] += 1
        print('acc:{}'.format(acc.compute()))
        print('auc:{}'.format(auc.compute()))
        for i in range(num_classes):
            print('class:{}, acc:{}'.format(i, correct_pred[i] / total_pred[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear probe')

    parser.add_argument('--backbone', type=str, default='swin-base_in21k-pre-3rdparty_in1k', help='backbone model')
    parser.add_argument('--dataset', type=str, default='ODIR', help='dataset name')
    parser.add_argument('--use_data', type=str, default='train+synthetic', help='use train data or train+synthetic data')
    parser.add_argument('--root1', type=str, default='/data1/home/yuanzhouhang/Result/MedAug/ODIR/train')
    parser.add_argument('--root2', type=str, default='/data1/home/yuanzhouhang/Result/LDM/ODIR')
    parser.add_argument('--test_root', type=str, default='/data1/home/yuanzhouhang/Result/MedAug/ODIR/test')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--weight_sampler_rate', type=float, default=-1, help='whether to use weighted sampler')

    args = parser.parse_args()
    print(args)
    main()