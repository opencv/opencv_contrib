import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot,opt.phase)
        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_trimap = os.path.join(self.dir_A, 'trimap')
        self.dir_fg = os.path.join(self.dir_A, 'fg')


        #self.A_paths = make_dataset(self.dir_A)
        self.fg_paths     = make_dataset(self.dir_fg)
        self.trimap_paths = make_dataset(self.dir_trimap)
        self.fg_paths     = sorted(self.fg_paths)
        self.trimap_paths = sorted(self.trimap_paths)
        self.transform    = get_transform(opt)

    def __getitem__(self, index):
        fg_path     = self.fg_paths[index]
        trimap_path = self.trimap_paths[index]
        A_fg        = Image.open(fg_path).convert('RGB')
        A_trimap    = Image.open(trimap_path).convert('L')
        #A_fg        = self.transform(A_fg)
        #A_trimap    = self.transform(A_trimap)

        A_fg        = A_fg.resize((320,320))
        A_trimap    = A_trimap.resize((320,320))
        A_fg        = transforms.ToTensor()(A_fg)
        A_trimap    = transforms.ToTensor()(A_trimap)

        input_nc = self.opt.input_nc

        return {'A_fg': A_fg, 'A_trimap' : A_trimap,'A_paths': fg_paths}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'TestDataset'
