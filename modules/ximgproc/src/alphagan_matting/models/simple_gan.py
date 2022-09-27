import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import torch.nn as nn

class SimpleModel(BaseModel):
    def name(self):
        return 'SimpleModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        nb = opt.batchSize
        size = opt.fineSize


        #Number of input channels : 4(Image + trimap), Number of output channels : 1
        self.netG = networks.define_G(4, 1,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, pretrain=not opt.no_pretrain)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(4, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)



        if not self.isTrain or opt.continue_train:
            #Load a network
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:

            #We use an image pool , so that the discriminator does not forget what it did right/wrong before.
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionAlpha = networks.AlphaPredictionLoss()
            self.criterionComp = networks.CompLoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # initialize optimizers
            #Using Adam optimizer for both discriminator and generator
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        A_bg = input['A_bg']
        A_fg = input['A_fg']
        A_alpha = input['A_alpha']
        A_trimap = input['A_trimap']
        if len(self.gpu_ids) > 0:
            A_bg = A_bg.cuda(self.gpu_ids[0], async=True)
            A_fg = A_fg.cuda(self.gpu_ids[0], async=True)
            A_alpha = A_alpha.cuda(self.gpu_ids[0], async=True)
            A_trimap = A_trimap.cuda(self.gpu_ids[0], async=True)
        self.bg_A = A_bg
        self.fg_A = A_fg
        self.alpha_A = A_alpha
        self.trimap_A = A_trimap
        #image is composed of the foreground and the background, using alpha matte.
        self.img_A = self.composite(self.alpha_A, self.fg_A, self.bg_A)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_predict(self, input):
        A_img = input['A_img']
        A_trimap = input['A_trimap']
        if len(self.gpu_ids) > 0:
            A_img = A_img.cuda(self.gpu_ids[0], async=True)
            A_trimap = A_trimap.cuda(self.gpu_ids[0], async=True)
        self.A_trimap = A_trimap
        self.A_img = A_img
        #Our input is composed of the Image and the trimap. This is input to the generator.
        self.input_A = torch.cat((self.A_img, self.A_trimap), 1)
        self.image_paths = input['A_paths']

    def composite(self, alpha, fg, bg):
        img = torch.mul(alpha, fg) + torch.mul((1.0 - alpha), bg)
        return img

    def trimap_merge(self, alpha, trimap):

        # Using the already known regions from trimap
        final_alpha = torch.where(torch.eq(torch.ge(trimap, 0.4), torch.le(trimap, 0.6)), alpha, trimap)
        return final_alpha

    def forward(self):
        self.A_input = Variable(torch.cat((self.img_A, self.trimap_A), 1))
        self.A_fg = Variable(self.fg_A)
        self.A_trimap = Variable(self.trimap_A)
        self.A_bg = Variable(self.bg_A)
        self.A_img = Variable(self.img_A)
        self.A_alpha = Variable(self.alpha_A)
        # self.A_disc = Variable(torch.cat((self.img_A, self.trimap_A, self.alpha_A), 1))

    def predict(self):
        self.netG.eval()
        with torch.no_grad():
            self.real_A = Variable(self.A_img)
            self.fake_B_alpha = self.netG(Variable(self.input_A))
            self.trimap_A = Variable(self.A_trimap)
            self.fake_B = self.trimap_merge(self.fake_B_alpha, self.trimap_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_comp = self.fake_AB_pool.query(self.comp_disc)

        loss_D = self.backward_D_basic(self.netD, self.A_input, fake_comp)

        self.loss_D = loss_D.data[0]

    def backward_G(self):
        pred = self.netG(self.A_input)
        pred = self.trimap_merge(pred, self.A_trimap)
        comp = self.composite(pred, self.A_fg, self.A_bg)

        comp_disc = torch.cat((comp, self.A_trimap), 1)

        pred_fake = self.netD(comp_disc)
        loss_g = self.criterionGAN(pred_fake, True)

        loss_a = self.criterionAlpha(pred, self.A_alpha, self.A_trimap)

        loss_c = 0

        loss = loss_a + loss_c + loss_g

        loss.backward()

        self.pred = pred.data
        self.comp_disc = comp_disc.data

        self.loss_a = loss_a.data[0]
        #self.loss_c = loss_c.data[0]
        self.loss_c = 0
        self.loss_g = loss_g.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('alpha_loss', self.loss_a), ('comp_loss', self.loss_c), ('gan_loss', self.loss_g), ('D', self.loss_D)])
        return ret_errors

    def get_current_visuals(self):
        pred = util.tensor2im(self.pred)
        gt = util.tensor2im(self.A_alpha)
        img = util.tensor2im(self.A_img)
        trimap = util.tensor2im(self.A_trimap)
        bg = util.tensor2im(self.A_bg)
        fg = util.tensor2im(self.A_fg)

        ret_visuals = OrderedDict([('img', img), ('trimap', trimap), ('pred', pred), ('gt', gt), ('fg', fg), ('bg', bg)])
        return ret_visuals

    def get_current_visuals_predict(self):
        real_A = util.tensor2im(self.real_A.data)
        trimap_A = util.tensor2im(self.trimap_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        fake_B_alpha = util.tensor2im(self.fake_B_alpha)
        return OrderedDict([('real_A', real_A), ('trimap_A', trimap_A), ('fake_B_alpha', fake_B_alpha), ('fake_B', fake_B)])


    def save(self, label):
        #saving seperately the weights of generator and discriminator
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
