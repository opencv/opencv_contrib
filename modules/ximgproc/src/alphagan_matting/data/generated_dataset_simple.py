import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import scipy.ndimage
import numpy as np
import math
# import pbcvt
# import colour_transfer

class GeneratedDatasetSimple(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_alpha = os.path.join(self.dir_AB, 'alpha')
        self.dir_fg = os.path.join(self.dir_AB, 'fg')
        self.dir_bg = os.path.join(self.dir_AB, 'bg')
        self.alpha_paths = sorted(make_dataset(self.dir_alpha))
        self.fg_paths = sorted(make_dataset(self.dir_fg))
        self.bg_paths = make_dataset(self.dir_bg)
        self.alpha_size = len(self.alpha_paths)
        self.bg_size = len(self.bg_paths)


    def __getitem__(self, index):
        index = index % self.alpha_size
        alpha_path = self.alpha_paths[index]
        fg_path = self.fg_paths[index]
        index_bg = random.randint(0, self.bg_size - 1)
        bg_path = self.bg_paths[index_bg]


        A_bg = Image.open(bg_path).convert('RGB')
        A_fg = Image.open(fg_path).convert('RGB')

        A_alpha = Image.open(alpha_path).convert('L')
        assert A_alpha.mode == 'L'


        A_trimap = self.generate_trimap(A_alpha)

        # A_bg = self.resize_bg(A_bg, A_fg)
        w_bg, h_bg = A_bg.size
        if w_bg < 321 or h_bg < 321:
            x = w_bg if w_bg < h_bg else h_bg
            ratio = 321/float(x)
            A_bg = A_bg.resize((int(np.ceil(w_bg*ratio)+1),int(np.ceil(h_bg*ratio)+1)), Image.BICUBIC)
        w_bg, h_bg = A_bg.size
        assert w_bg > 320 and h_bg > 320, '{} {}'.format(w_bg, h_bg)
        x = random.randint(0, w_bg-320-1)
        y = random.randint(0, h_bg-320-1)
        A_bg = A_bg.crop((x,y, x+320, y+320))

        crop_size = random.choice([320,480,640])
        # crop_size = random.choice([320,400,480,560,640,720])
        crop_center = self.find_crop_center(A_trimap)
        start_index_height = max(min(A_fg.size[1]-crop_size, crop_center[0] - int(crop_size/2) + 1), 0)
        start_index_width = max(min(A_fg.size[0]-crop_size, crop_center[1] - int(crop_size/2) + 1), 0)

        bbox = ((start_index_width,start_index_height,start_index_width+crop_size,start_index_height+crop_size))

        # A_bg = A_bg.crop(bbox)
        A_fg = A_fg.crop(bbox)
        A_alpha = A_alpha.crop(bbox)
        A_trimap = A_trimap.crop(bbox)

        if self.opt.which_model_netG == 'unet_256':
            A_bg = A_bg.resize((256,256))
            A_fg = A_fg.resize((256,256))
            A_alpha = A_alpha.resize((256,256))
            A_trimap = A_trimap.resize((256,256))
            assert A_alpha.mode == 'L'
        else:
            A_bg = A_bg.resize((320,320))
            A_fg = A_fg.resize((320,320))
            A_alpha = A_alpha.resize((320,320))
            A_trimap = A_trimap.resize((320,320))
            assert A_alpha.mode == 'L'

        if random.randint(0, 1):
            A_bg = A_bg.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if random.randint(0, 1):
            A_fg = A_fg.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            A_alpha = A_alpha.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            A_trimap = A_trimap.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        ## COLOR TRANSFER ##
        # if random.randint(0, 2) != 0:
        #     A_old = A_fg
        #     target = np.array(A_fg)
        #     palette = np.array(A_palette)
        #     recolor = colour_transfer.runCT(target, palette)
        #     A_fg = Image.fromarray(recolor)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        A_bg = transforms.ToTensor()(A_bg)
        A_fg = transforms.ToTensor()(A_fg)
        A_alpha = transforms.ToTensor()(A_alpha)
        A_trimap = transforms.ToTensor()(A_trimap)

        return {'A_bg': A_bg,
                'A_fg': A_fg,
                'A_alpha': A_alpha,
                'A_trimap': A_trimap,
                'A_paths': alpha_path}

    def resize_bg(self, bg, fg):
        bbox = fg.size
        w = bbox[0]
        h = bbox[1]
        bg_bbox = bg.size
        bw = bg_bbox[0]
        bh = bg_bbox[1]
        wratio = w / float(bw)
        hratio = h / float(bh)
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = bg.resize((int(np.ceil(bw*ratio)+1),int(np.ceil(bh*ratio)+1)), Image.BICUBIC)
        bg = bg.crop((0,0,w,h))

        return bg

    # def generate_trimap(self, alpha):
    #     trimap = np.array(alpha)
    #     kernel_sizes = [val for val in range(5,40)]
    #     kernel = random.choice(kernel_sizes)
    #     trimap[np.where((scipy.ndimage.grey_dilation(alpha,size=(kernel,kernel)) - alpha!=0))] = 128

    #     return Image.fromarray(trimap)
    def generate_trimap(self, alpha):
        trimap = np.array(alpha)
        grey = np.zeros_like(trimap)
        kernel_sizes = [val for val in range(2,20)]
        kernel = random.choice(kernel_sizes)
        # trimap[np.where((scipy.ndimage.grey_dilation(alpha,size=(kernel,kernel)) - alpha!=0))] = 128
        grey = np.where(np.logical_and(trimap>0, trimap<255), 128, 0)
        grey = scipy.ndimage.grey_dilation(grey, size=(kernel,kernel))
        trimap[grey==128] = 128

        return Image.fromarray(trimap)

    def find_crop_center(self, trimap):
        t = np.array(trimap)
        target = np.where(t==128)
        index = random.choice([i for i in range(len(target[0]))])
        return np.array(target)[:,index][:2]

    def rotatedRectWithMaxArea(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0,0
        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr,hr

    def __len__(self):
        return len(self.alpha_paths)

    def name(self):
        return 'GeneratedDataset'
