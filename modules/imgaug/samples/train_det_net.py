import os
import time

import numpy as np
import torch
import cv2
import argparse
import torchvision
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="PennFudanPed")
    parser.add_argument("--lr", type=float, default=3e-4)

    return parser.parse_args()


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def _get_boxes(self, mask):
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            yield xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask is array of size (H, W), all elements of array are integers
        # background is 0, and each distinct person is represented as a distinct integer starting from 1
        # you can treat mask as grayscale image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        boxes = []
        for x1, y1, x2, y2 in self._get_boxes(mask):
            # NOTE: in opencv, box is represented as (x, y, width, height)
            boxes.append([x1, y1, x2-x1, y2-y1])
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img, boxes = self.transforms.call(img, boxes)

        # 1. transpose from (h, w, c) to (c, h, w)
        # 2. normalize data into range 0-1
        # 3. convert from np.array to torch.tensor
        img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        boxes = [[x1, y1, x1+width, y1+height] for x1, y1, width, height in boxes]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        return img, boxes, labels

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def collate_fn(batch):
        images = list()
        boxes = list()
        labels = list()
        targets = list()

        for item in batch:
            images.append(item[0])
            # boxes.append(item[1])
            # labels.append(item[2])
            target = {"boxes": item[1], "labels": item[2]}
            targets.append(target)

        images = torch.stack(images, dim=0)

        return images, targets


def get_transforms():

    transforms = cv2.det.Compose([
        cv2.det.RandomFlip(),
        cv2.det.Resize((500, 500)),
    ])

    return transforms


def train(num_epochs, device, model, dataloader, optimizer):
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()

            images, targets = batch
            images = images.to(device)

            outputs = model(images, targets)
            losses = sum(outputs.values())

            losses.backward()
            optimizer.step()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = get_transforms()
    dataset = PennFudanDataset(args.root, transforms=transforms)

    indices = torch.randperm(len(dataset)).tolist()
    train_set = torch.utils.data.Subset(dataset, indices[:-50])
    test_set = torch.utils.data.Subset(dataset, indices[-50:])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, collate_fn=PennFudanDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=PennFudanDataset.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)

    parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    start = time.time()
    train(2, device, model, train_loader, optimizer)
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    main()
