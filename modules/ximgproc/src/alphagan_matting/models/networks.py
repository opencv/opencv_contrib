import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np

###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    """
    Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    """

    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: lambda | step | plateau
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], pretrain=True):
    """
    Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        which_model_netG (str) -- the architecture's name: resnet50 | resnet50ASPP
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    """
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())


    if which_model_netG == 'resnet50':
        netG = ResnetX(id=50, gpu_ids=gpu_ids, pretrain=pretrain)
    elif which_model_netG == 'resnet50ASPP':
        netG = ResnetASPP(id=50,gpu_ids=gpu_ids,pretrain=pretrain)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])

    print('Using pretrained weights')

    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    """
    Create a discriminator
    Parameters:
        input_nc (int)          -- the number of channels in input images
        ndf (int)               -- the number of filters in the first conv layer
        which_model_netD (str)  -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)        -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)              -- the type of normalization layers used in the network.
        init_type (str)         -- the name of the initialization method
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    """
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    """
    Lgan(G,D)= log D(x)+log(1−D(C(G(x)))
    where x is a real input: an image composited from the ground-truth alpha and foreground appended with the trimap.
    C(y) is a composition function that takes the predicted alpha from G as an input and uses it to composite a fake image.
    G tries to generate alphas that are close to the ground-truth alpha, while D tries to
    distinguish real from fake composited images.
    G therefore tries to minimize Lgan against the discriminator D, which tries to maximize it.
    """
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class AlphaPredictionLoss(nn.Module):
    """
    It is the absolute difference between the ground truth alpha values and the predicted alpha values at each pixel.
    """
    def __init__(self):
        super(AlphaPredictionLoss, self).__init__()

    def forward(self, input, target, trimap):
        trimap_weights = torch.where(torch.eq(torch.ge(trimap, 0.4), torch.le(trimap, 0.6)), torch.ones_like(trimap), torch.zeros_like(trimap))
        unknown_region_size = trimap_weights.sum()
        diff = torch.sqrt(torch.add(torch.pow(input - target, 2), 1e-12))
        return torch.mul(diff, trimap_weights).sum() / unknown_region_size


class CompLoss(nn.Module):

    """
    Compostion Loss : Absolute difference between the ground truth RGB colors and the predicted RGB colors composited
    by the groundtruth foreground, the ground truth background and the predicted alpha mattes
    """

    def __init__(self):
        super(CompLoss, self).__init__()

    def forward(self, input, target, trimap, fg, bg):
        trimap_weights = torch.where(torch.eq(torch.ge(trimap, 0.4), torch.le(trimap, 0.6)), torch.ones_like(trimap), torch.zeros_like(trimap))
        unknown_region_size = trimap_weights.sum()


        comp_target = torch.mul(target, fg) + torch.mul((1.0 - target), bg)
        comp_input = torch.mul(input, fg) + torch.mul((1.0 - input), bg)

        diff = torch.sqrt(torch.add(torch.pow(comp_input - comp_target, 2), 1e-12))
        return torch.mul(diff, trimap_weights).sum() / unknown_region_size


class ResnetX(nn.Module):
    def __init__(self, id=50, gpu_ids=[], pretrain=True):
        super(ResnetX, self).__init__()
        self.encoder = ResnetEncoder(id, gpu_ids, pretrain)
        self.decoder = UNetDecoder(gpu_ids)

    def forward(self, input):
        x, ind = self.encoder(input)
        x = self.decoder(x, ind)

        return x


class ResnetEncoder(nn.Module):
    """
    Encoder has the same structure as that of Resnet50,but the last 2 layers have been removed.
    The shape of first channel has been changed, Resnet had 3 channels, but for this task we need 4 channels as we
    are also adding the trimap
    """

    def __init__(self, id=50, pretrain=True, gpu_ids=[]):
        super(ResnetEncoder, self).__init__()
        print('Pretrain: {}'.format(pretrain))
        if id==50:
            resnet = models.resnet50(pretrained=pretrain)

        modules = list(resnet.children())[:-2] # delete the last 2 layers.
        for m in modules:
            if 'MaxPool' in m.__class__.__name__:
                m.return_indices = True


        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weights = torch.zeros(64, 4, 7, 7)
        weights[:,:3,:,:] = modules[0].weight.data.view(64, 3, 7, 7)
        conv1.weight.data.copy_(weights)
        modules[0] = conv1

        self.pool1 = nn.Sequential(*modules[: 4])
        self.resnet = nn.Sequential(*modules[4:])


    def forward(self, input):
        x, ind = self.pool1(input)

        x = self.resnet(x)

        return x, ind

class UNetDecoder(nn.Module):

    """
    The decoder network of the generator is same as that of the UNetDecoder. It
    has seven upsampling convolutional blocks.Each upsampling convolutional block has an
    upsampling layer followed by a convolutional layer, a batch normalization layer and a ReLU activation function
    """
    def __init__(self, gpu_ids=[]):
        super(UNetDecoder, self).__init__()
        model = [nn.Conv2d(2048, 2048, kernel_size=1, padding=0),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=2, output_padding=1, bias=False),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True)]
        model += [nn.Conv2d(1024, 1024, kernel_size=5, padding=2),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, output_padding=1, bias=False),
                  nn.BatchNorm2d(512),
                  nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=5, padding=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, output_padding=1, bias=False),
                  # nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=5, padding=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True),
                  nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        self.model1 = nn.Sequential(*model)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        model = [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  nn.Conv2d(64, 1, kernel_size=5, padding=2),
                  nn.Sigmoid()]
        self.model2 = nn.Sequential(*model)

        init_weights(self.model1, 'xavier')
        init_weights(self.model2, 'xavier')

    def forward(self, input, ind):
        x = self.model1(input)
        x = self.unpool(x, ind)
        x = self.model2(x)

        return x



class ASPP_Module(nn.Module):
    def __init__(self, input_maps, dilation_series, padding_series, output_maps):
        super(ASPP_Module, self).__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(nn.Conv2d(input_maps, output_maps, kernel_size=1, stride=1, bias=False),
                                           nn.BatchNorm2d(output_maps, affine=affine_par)))

        for dilation, padding in zip(dilation_series, padding_series):
            self.branches.append(nn.Sequential(nn.Conv2d(input_maps, output_maps, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                                               nn.BatchNorm2d(output_maps, affine=affine_par)))

        for m in self.branches:
            m[0].weight.data.normal_(0, 0.01)

        image_level_features = [nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(input_maps, output_maps, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(output_maps, affine=affine_par)]
        self.image_level_features = nn.Sequential(*image_level_features)
        self.conv1x1 = nn.Conv2d(output_maps*(len(dilation_series)+2), output_maps, kernel_size=1, stride=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(output_maps, affine=affine_par)

    def forward(self, x):
        out = self.branches[0](x)
        for i in range(len(self.branches)-1):
            out = torch.cat([out, self.branches[i+1](x)], 1)

        image_features = nn.functional.upsample(self.image_level_features(x), size=(out.shape[2],out.shape[3]), mode='bilinear')
        out = torch.cat([out, image_features], 1)
        out = self.conv1x1(out)
        out = self.bn1x1(out)

        return out

class ResnetASPP(nn.Module):
    def __init__(self, id=50, gpu_ids=[], pretrain=True):
        super(ResnetX, self).__init__()
        self.encoder = ResnetASPPEncoder(id, gpu_ids, pretrain)
        self.decoder = UNetASPPDecoder(gpu_ids)

    def forward(self, input):
        x, ind = self.encoder(input)
        x = self.decoder(x, ind)

        return x



class ResnetASPPEncoder(nn.Module):
    """
    Encoder has the same structure as that of Resnet50,but the last 2 layers have been removed.
    The shape of first channel has been changed, Resnet had 3 channels, but for this task we need 4 channels as we
    are also adding the trimap. In this encoder , I have also added the ASPP module in the end.
    """

    def __init__(self, id=50, pretrain=True, gpu_ids=[]):
        super(ResnetASPPEncoder, self).__init__()
        print('Pretrain: {}'.format(pretrain))
        if id==50:
            resnet = models.resnet50(pretrained=pretrain)

        modules = list(resnet.children())[:-2] # delete the last 2 layers.
        for m in modules:
            if 'MaxPool' in m.__class__.__name__:
                m.return_indices = True


        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weights = torch.zeros(64, 4, 7, 7)
        weights[:,:3,:,:] = modules[0].weight.data.view(64, 3, 7, 7)
        conv1.weight.data.copy_(weights)
        modules[0] = conv1

        self.pool1      = nn.Sequential(*modules[: 4])
        self.resnet     = nn.Sequential(*modules[4:])
        self.ASPP_layer = ASPP_Module(2048, [6,12,18],[6,12,18], 1024)

    def forward(self, input):
        x, ind = self.pool1(input)

        x = self.resnet(x)
        x = self.ASPP_layer(x)

        return x, ind

class UNetASPPDecoder(nn.Module):

    """
    The decoder network of the generator is same as that of the UNetDecoder. It
    has seven upsampling convolutional blocks.Each upsampling convolutional block has an
    upsampling layer followed by a convolutional layer, a batch normalization layer and a ReLU activation function.The only
    difference in this
    """
    def __init__(self, gpu_ids=[]):
        super(UNetASPPDecoder, self).__init__()
        model = [nn.Conv2d(1024,1024, kernel_size=3, padding=1),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(1024, 1024, kernel_size=1, stride=2, output_padding=1, bias=False),
                 nn.BatchNorm2d(1024),
                 nn.ReLU(True)]
        model += [nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, output_padding=1, bias=False),
                  nn.BatchNorm2d(512),
                  nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=5, padding=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, output_padding=1, bias=False),
                  # nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=5, padding=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True),
                  nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        self.model1 = nn.Sequential(*model)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        model = [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=5, padding=2),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  nn.Conv2d(64, 1, kernel_size=5, padding=2),
                  nn.Sigmoid()]
        self.model2 = nn.Sequential(*model)

        init_weights(self.model1, 'xavier')
        init_weights(self.model2, 'xavier')

    def forward(self, input, ind):
        x = self.model1(input)
        x = self.unpool(x, ind)
        x = self.model2(x)

        return x





# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)
