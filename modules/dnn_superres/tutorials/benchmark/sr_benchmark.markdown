Super-resolution benchmarking {#tutorial_dnn_superres_benchmark}
===========================

Benchmarking
----

The super-resolution module contains sample codes for benchmarking, in order to compare different models and algorithms.
Here is presented a sample code for performing benchmarking, and then a few benchmarking results are collected.
It was performed on an Intel i7-9700K CPU on an Ubuntu 18.04.02 OS.

Source Code of the sample
-----------

@includelineno dnn_superres/samples/dnn_superres_benchmark_quality.cpp

Explanation
-----------

-#  **Read and downscale the image**
    @code{.cpp}
     int width = img.cols - (img.cols % scale);
     int height = img.rows - (img.rows % scale);
     Mat cropped = img(Rect(0, 0, width, height));
     Mat img_downscaled;
     cv::resize(cropped, img_downscaled, cv::Size(), 1.0 / scale, 1.0 / scale);
    @endcode

    Resize the image by the scaling factor. Before that a cropping is necessary, so the images will align.

-#  **Set the model**
    @code{.cpp}
    DnnSuperResImpl sr;
    sr.readModel(path);
    sr.setModel(algorithm, scale);
    sr.upsample(img_downscaled, img_new);
    @endcode

    Instantiate a dnn super-resolution object. Read and set the algorithm and scaling factor.

-#  **Perform benchmarking**
    @code{.cpp}
    double psnr = PSNR(img_new, cropped);
    Scalar q = cv::quality::QualitySSIM::compute(img_new, cropped, cv::noArray());
    double ssim = mean(cv::Vec3f(q[0], q[1], q[2]))[0];
    @endcode

    Calculate PSNR and SSIM. Use OpenCVs PSNR (core opencv) and SSIM (contrib) functions to compare the images.
    Repeat it with other upscaling algorithms, such as other DL models or interpolation methods (eg. bicubic, nearest neighbor).

Benchmarking results
-----------

Dataset benchmarking
----

###General100 dataset

<center>

#####2x scaling factor


|               | Avg inference time in sec (CPU)| Avg PSNR | Avg SSIM |
| ------------- |:-------------------:| ---------:|--------:|
| ESPCN            | **0.008795** | 32.7059 | 0.9276 |
| EDSR             | 5.923450 | **34.1300** | **0.9447** |
| FSRCNN           | 0.021741 | 32.8886 | 0.9301 |
| LapSRN           | 0.114812 | 32.2681 | 0.9248 |
| Bicubic          | 0.000208 | 32.1638 | 0.9305 |
| Nearest neighbor | 0.000114 | 29.1665 | 0.9049 |
| Lanczos          | 0.001094 | 32.4687 | 0.9327 |

#####3x scaling factor

|               | Avg inference time in sec (CPU)| Avg PSNR | Avg SSIM |
| ------------- |:-------------------:| ---------:|--------:|
| ESPCN            | **0.005495**  | 28.4229 | 0.8474 |
| EDSR             | 2.455510    | **29.9828**  | **0.8801** |
| FSRCNN           | 0.008807   | 28.3068 | 0.8429 |
| LapSRN           | 0.282575    |26.7330   |0.8862  |
| Bicubic          | 0.000311 |26.0635  |0.8754  |
| Nearest neighbor | 0.000148 |23.5628  |0.8174  |
| Lanczos          | 0.001012  |25.9115  |0.8706  |


#####4x scaling factor

|               | Avg inference time in sec (CPU)| Avg PSNR | Avg SSIM |
| ------------- |:-------------------:| ---------:|--------:|
| ESPCN            | **0.004311** | 26.6870 | 0.7891 |
| EDSR             | 1.607570    | **28.1552** | **0.8317**  |
| FSRCNN           | 0.005302  | 26.6088 | 0.7863 |
| LapSRN           | 0.121229    |26.7383   |0.7896  |
| Bicubic          | 0.000311 |26.0635  |0.8754  |
| Nearest neighbor | 0.000148 |23.5628  |0.8174  |
| Lanczos          | 0.001012  |25.9115  |0.8706  |


</center>

Images
----

<center>

####2x scaling factor

|Set5: butterfly.png | size: 256x256 | ||
|:-------------:|:-------------------:|:-------------:|:----:|
|![Original](images/orig_butterfly.jpg)|![Bicubic interpolation](images/bicubic_butterfly.jpg)|![Nearest neighbor interpolation](images/nearest_butterfly.jpg)|![Lanczos interpolation](images/lanczos_butterfly.jpg) |
|PSRN / SSIM / Speed (CPU)|26.6645 / 0.9048 / 0.000201 |23.6854 / 0.8698 / **0.000075** | **26.9476** / **0.9075** / 0.001039|
![ESPCN](images/espcn_butterfly.jpg)| ![FSRCNN](images/fsrcnn_butterfly.jpg) | ![LapSRN](images/lapsrn_butterfly.jpg) | ![EDSR](images/edsr_butterfly.jpg)
|29.0341 / 0.9354 / **0.004157**| 29.0077 / 0.9345 / 0.006325 | 27.8212 / 0.9230 / 0.037937 | **30.0347** / **0.9453** / 2.077280 |

####3x scaling factor

|Urban100: img_001.png | size: 1024x644 | ||
|:-------------:|:-------------------:|:-------------:|:----:|
|![Original](images/orig_urban.jpg)|![Bicubic interpolation](images/bicubic_urban.jpg)|![Nearest neighbor interpolation](images/nearest_urban.jpg)|![Lanczos interpolation](images/lanczos_urban.jpg) |
|PSRN / SSIM / Speed (CPU)| 27.0474 / **0.8484** / 0.000391 | 26.0842 / 0.8353 / **0.000236** | **27.0704** / 0.8483 / 0.002234|
|![ESPCN](images/espcn_urban.jpg)| ![FSRCNN](images/fsrcnn_urban.jpg) | LapSRN is not trained for 3x <br/> because of its architecture  | ![EDSR](images/edsr_urban.jpg)
|28.0118 / 0.8588 / **0.030748**| 28.0184 / 0.8597 / 0.094173 |  | **30.5671** / **0.9019** / 9.517580 |


####4x scaling factor

|Set14: comic.png | size: 250x361 | ||
|:-------------:|:-------------------:|:-------------:|:----:|
|![Original](images/orig_comic.jpg)|![Bicubic interpolation](images/bicubic_comic.jpg)|![Nearest neighbor interpolation](images/nearest_comic.jpg)|![Lanczos interpolation](images/lanczos_comic.jpg) |
|PSRN / SSIM / Speed (CPU)| **19.6766** / **0.6413** / 0.000262 |18.5106 / 0.5879 / **0.000085** | 19.4948 / 0.6317 / 0.001098|
|![ESPCN](images/espcn_comic.jpg)| ![FSRCNN](images/fsrcnn_comic.jpg) | ![LapSRN](images/lapsrn_comic.jpg) | ![EDSR](images/edsr_comic.jpg)
|20.0417 / 0.6302 / **0.001894**| 20.0885 / 0.6384 / 0.002103 | 20.0676 / 0.6339 / 0.061640 | **20.5233** / **0.6901** / 0.665876 |

####8x scaling factor

|Div2K: 0006.png | size: 1356x2040 | |
|:-------------:|:-------------------:|:-------------:|
|![Original](images/orig_div2k.jpg)|![Bicubic interpolation](images/bicubic_div2k.jpg)|![Nearest neighbor interpolation](images/nearest_div2k.jpg)|
|PSRN / SSIM / Speed (CPU)| 26.3139 / **0.8033** / 0.001107| 23.8291 / 0.7340 / **0.000611** |
|![Lanczos interpolation](images/lanczos_div2k.jpg)| ![LapSRN](images/lapsrn_div2k.jpg) | |
|26.1565 / 0.7962 / 0.004782| **26.7046** / 0.7987 / 2.274290 | |

</center>