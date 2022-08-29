import os
import pandas as pd
import argparse
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils import data
import numpy as np
import time
import tqdm


class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, root, df_data, mode='train', transform=None, backend='cv2'):
        super(ImagenetteDataset, self).__init__()
        assert mode in ['train', 'valid']
        assert backend in ['cv2', 'PIL']

        self.root = root
        self.transform = transform
        self.backend = backend
        labels = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
        self.label_to_num = {v: k for k, v in enumerate(labels)}

        if mode == 'train':
            self.df_data = df_data[df_data['is_valid'] == False][:256]
        else:
            self.df_data = df_data[df_data['is_valid'] == True]

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        path = self.df_data.iloc[idx]['path']
        path = os.path.join(self.root, path)
        image = self.get_image(path)
        label = path.split('/')[-2]
        label = self.label_to_num[label]
        return image, label

    def get_image(self, path):
        if self.backend == 'cv2':
            image = cv2.imread(path)
            if self.transform:
                image = self.transform.call(image)
            image = np.transpose(image, (2, 0, 1))
            return torch.tensor(image, dtype=torch.float)
        if self.backend == 'PIL':
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image


def train(dataloader, model, num_epochs, criterion, optimizer):
    start = time.time()
    for epoch in range(num_epochs):
        model.train()

        for inputs, targets in tqdm.tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

    end = time.time()
    print(end-start)


def main():
    root_dir = "/Users/chuyang/Downloads/imagenette2-320"
    df_train = pd.read_csv(os.path.join(root_dir, "noisy_imagenette.csv"))
    print('load %d records' % len(df_train))

    ocv_transforms = cv2.Compose([
        cv2.RandomCrop((300, 300), (0,0,0,0)),
        cv2.RandomFlip(),
        cv2.Resize((500, 500)),
    ])

    pil_transforms = transforms.Compose([
        transforms.RandomCrop((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])

    # train_set = ImagenetteDataset(root_dir, df_train, 'train', ocv_transforms, backend='cv2')
    train_set = ImagenetteDataset(root_dir, df_train, 'train', pil_transforms, backend='PIL')

    train_loader = data.DataLoader(train_set, num_workers=0, batch_size=16, drop_last=True, shuffle=True)
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()



    train(train_loader, model, 1, criterion, optimizer)


if __name__ == '__main__':
    main()
