Use imgaug with PyTorch {#tutorial_imgaug_pytorch}
==============================

@tableofcontents

@prev_tutorial{tutorial_imgaug_object_detection}

|    |    |
| -: | :- |
| Author | Chuyang Zhao |
| Compatibility | OpenCV >= 4.0 |

Introduction
------------
Imgaug is the data augmentation module in OpenCV which allows you to process
the data before putting them into the model. Because imgaug is implemented in
pure C++ and is backend with OpenCV's efficient image processing operations,
it runs faster and more efficiently than other existing Python-based
implementations. In this tutorial, I will demonstrate how to use imgaug
with PyTorch. Specifically, how to preprocess the data before putting
them into the PyTorch model for training or inference.


Goals
-----
In this tutorial, you will learn how to:
1. Use imgaug to perform data augmentation on your input data
2. Use imgaug with PyTorch for the image classification task
3. Use imgaug with PyTorch for the object detection task


Usage
-----
### Use imgaug with PyTorch in image classification task
In this section, we use Imagenette as the training dataset. You can download it [here](https://github.com/fastai/imagenette).

Firstly, we define the dataset of PyTorch as follows:

@code{.py}
class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, root, df_data, mode='train', transform=None):
        super(ImagenetteDataset, self).__init__()
        assert mode in ['train', 'valid']

        self.root = root
        self.transform = transform
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
        image = cv2.imread(path)
        if self.transform:
            image = self.transform.call(image)
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float)
@endcode

In this dataset, we use `transforms` which we defined below to perform
data augmentation on the image.

The transforms we used contain four data augmentation methods, and they are
composed into one using the cv::imgaug::Compose class.

@code{.py}
transforms = cv2.Compose([
    cv2.RandomCrop((300, 300), (0,0,0,0)),
    cv2.RandomFlip(),
    cv2.Resize((500, 500)),
    cv2.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
])
@endcode

@note The mean and std here we pass to cv.Normalize are [0.406, 0.456, 0.485] and [0.225, 0.224, 0.229]
respectively, which is slightly different from the mean and std of ImageNet (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
This is because the mean and std of ImageNet is for image in format RGB. But the image read by OpenCV is in BGR format.
So we need to change the order of the original mean and std of ImageNet to make it suitable for image read by OpenCV.

After constructing the dataset and building the model, we can start training our model:

@code{.py}
train_set = ImagenetteDataset(root_dir, df_train, 'train', transforms)

train_loader = data.DataLoader(train_set, num_workers=0, batch_size=16, drop_last=True, shuffle=True)
model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(in_features=512, out_features=10)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

train(train_loader, model, 1, criterion, optimizer)
@endcode

Complete code of the example is as follows:

@include samples/train_cls_net.py

### Use imgaug with PyTorch in object detection task

In this section, we use Penn-Fudan dataset for training the object detection model.
You can download the dataset from [here](https://www.cis.upenn.edu/~jshi/ped_html/).

Similarly, we first define the PyTorch dataset:
@code{.py}
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
@endcode

Then we define the transforms we use for data augmentation as:
@code{.py}
def get_transforms():
    transforms = cv2.det.Compose([
        cv2.det.RandomFlip(),
        cv2.det.Resize((500, 500)),
    ])
    return transforms
@endcode

Complete code the example is as follows:

@include samples/train_det_net.py