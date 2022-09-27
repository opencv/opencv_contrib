## Architecture ##

#### Generator ####
A decoder encoder based architecture is used.


There are 2 options for the generator encoder.

<b>a.</b> Resnet50 minus the last 2 layers
<b>b.</b> Resnet50 + ASPP module

The Decoder network of the Generator network has seven upsampling convolutional blocks.
Each upsampling convolutional block has an upsampling layer, followed by a convolutional layer, a batch normalization layer and a ReLU activation function.

#### Discriminator ####
The discriminator used here is the PatchGAN discriminator. The implementation here is inspired from the implementation of CycleGAN from<br/>
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Again, 2 different types of discriminator are used
<b>a.</b> N Layer Patch gan discriminator, where the size of the patch is NxN, it is taken as 3x3 here
<b>b.</b> Pixel patch Patch gan discriminator, the discriminator classsifies every pixel.

## How to use
Use the dataroot argument to enter the directory where you have stored the data.
Structure the data in the following way.

train<br/>
-alpha  -bg  -fg

test<br/>
-fg   -trimap

The background I have used here is the MSCOCO dataset.


To train the model using Resnet50 without ASPP module

`!python train.py --dataroot ./ --model simple --dataset_mode generated_simple --which_model_netG resnet50 --name resnet50`

To test the model using Resnet without ASPP module

`!python test.py --dataroot ./  --dataset_mode single  --which_model_netG resnet50 --ntest 8 --model test --name resnet50`

To train the model using Resnet50 using ASPP module

`!python train.py --dataroot ./ --model simple --dataset_mode generated_simple --which_model_netG resnet50ASPP --name resnet50ASPP`

To test the model using Resnet50 using ASPP module

`!python test.py --dataroot ./  --dataset_mode single --which_model_netG resnet50ASPP --ntest 8 --model test`

## Results

#### Input:
![input](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/2.png)

#### Trimap:
![Trimap](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/donkey_tri.png)


#### AlphaGAN matting :
##### Generator:Resnet50,Discriminator:N Layer Patch GAN
![Output2](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/donkey_resnet50.png)

##### Generator:Resnet50,Discriminator:Pixel Patch GAN
![Output3](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/donkey.png)

##### Generator:Resnet50 + ASPP module,Discriminator:N Layer Patch GAN
![Output4](https://raw.githubusercontent.com/Nerdyvedi/GSOC-Opencv-matting/master/donkey_deeplab.png)


### Comparing with the Original implementation
(Average Rank on alphamatting.com has been shown)

|     Error type              |      Original implementation    | Resnet50 +N Layer   |   Resnet50 + Pixel | Resnet50 + ASPP module |
|     -----------             |      ------------------------   | -------------------  |   ----------------- | -----------|
| Sum of absolute differences |       11.7                      | 42.8                 |      43.8           |    53      |
| Mean square error           |       15                        | 45.8                 |      45.6           |    54.2    |
| Gradient error              |       14                        | 52.9                 |      52.7           |    55      |
| Connectivity error          |       29.6                      | 23.3                 |      22.6           |    32.8    |


### Training dataset used
I used the training dataset created by me using the software known as gimp.
[Link to created dataset](https://drive.google.com/open?id=1zQbk2Cu7QOBwzg4vVGqCWJwHGTwGppFe)

### What could be wrong ?

1.The dataset I created is not perfect. I tried to make it as perfect by marking every pixel in some cases as well, but it still ain't perfect.

2. The output of the generator is 320x320, then the alpha matte is resized to the original image. Maybe there is a loss in resizing the image, and a better upsampling method might just imporove the outputs.

3. The major reason the implementation isn't as good as original is that the author hasn't clearly mentioned how the skip connections are used. I am very sure about the architecture of the encoder and decoder, but the only thing I am unsure about is how skip connections are used.
