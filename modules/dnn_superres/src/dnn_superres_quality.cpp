// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/dnn_superres_quality.hpp"

namespace cv
{
    namespace dnn_superres
    {
        int DnnSuperResQuality::fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;

        double DnnSuperResQuality::fontScale = 1.0;

        Scalar DnnSuperResQuality::fontColor = Scalar(255,255,255);

        void DnnSuperResQuality::setFontFace(int fontface)
        {
            fontFace = fontface;
        }

        void DnnSuperResQuality::setFontScale(double fontscale)
        {
            fontScale = fontscale;
        }

        void DnnSuperResQuality::setFontColor(cv::Scalar fontcolor)
        {
            fontColor = fontcolor;
        }

        double DnnSuperResQuality::psnr(Mat img, Mat orig)
        {
            CV_Assert(!img.empty());
            CV_Assert(!orig.empty());
            CV_Assert(img.type() == CV_8UC3);
            CV_Assert(img.type() == orig.type());

            Mat img_float;
            img.convertTo(img_float, CV_32F);

            Mat orig_float;
            orig.convertTo(orig_float, CV_32F);

            Mat img_diff;
            absdiff(orig_float, img_float, img_diff);

            Mat img_mul = img_diff.mul(img_diff);

            Scalar sum = cv::sum(img_mul);

            double rgb_sum = 0.0;
            for(int i = 0; i < 4; i++){
                rgb_sum += sum[i];
            }

            double mse = rgb_sum / (double) (3.0 * img_float.total());

            double max = 255 * 255;
            double psnr = 10 * log10(max / (double) mse);

            return psnr;
        }

        double DnnSuperResQuality::ssim(Mat img, Mat orig)
        {
            CV_Assert(!img.empty());
            CV_Assert(!orig.empty());
            CV_Assert(img.type() == CV_8UC3);
            CV_Assert(img.type() == orig.type());

            Mat ycrcb_img, ycrcb_orig;
            cvtColor(img, ycrcb_img, COLOR_BGR2YCrCb);
            cvtColor(orig, ycrcb_orig, COLOR_BGR2YCrCb);

            Mat img_channels[3], orig_channels[3];
            split(img, img_channels);
            split(orig, orig_channels);

            Mat img_float;
            img_channels[0].convertTo(img_float, CV_32F);

            Mat orig_float;
            orig_channels[0].convertTo(orig_float, CV_32F);

            Mat mu_img, mu_orig;
            GaussianBlur( img_float, mu_img, Size(11, 11), 0, 0 );
            GaussianBlur( orig_float, mu_orig, Size(11, 11), 0, 0 );
            Mat mu_mul = mu_img.mul(mu_orig);
            mu_img = mu_img.mul(mu_img);
            mu_orig = mu_orig.mul(mu_orig);

            Mat sigma_img, sigma_orig;
            GaussianBlur( img_float.mul(img_float), sigma_img, Size(11, 11), 0, 0 );
            GaussianBlur( orig_float.mul(orig_float), sigma_orig, Size(11, 11), 0, 0 );
            sigma_img = sigma_img - mu_img;
            sigma_orig = sigma_orig - mu_orig;

            Mat sigma_mul;
            GaussianBlur( img_float.mul(orig_float), sigma_mul, Size(11, 11), 0, 0 );
            sigma_mul = sigma_mul - mu_mul;

            double c1 = (0.01 * 255) * (0.01 * 255);
            double c2 = (0.03 * 255) * (0.03 * 255);

            Mat nom1 = (2 * mu_mul + c1);
            Mat nom2 = (2 * sigma_mul + c2);

            Mat nominator = nom1.mul(nom2);

            Mat denom1 = mu_img + mu_orig + c1;
            Mat denom2 = sigma_img + sigma_orig + c2;

            Mat denominator = denom1.mul(denom2);

            Mat ssim = nominator / denominator;

            Scalar mssim = mean(ssim);

            return mssim[0];
        }

        void DnnSuperResQuality::benchmark(DnnSuperResImpl sr, Mat img,
                                            std::vector<double>& psnrValues,
                                            std::vector<double>& ssimValues,
                                            std::vector<double>& perfValues,
                                            bool showImg,
                                            bool showOutput)
        {
            CV_Assert(!img.empty());

            psnrValues = std::vector<double>();
            ssimValues = std::vector<double>();
            perfValues = std::vector<double>();

            if(showOutput)
                std::cout << "Start benchmarking" << std::endl;

            int scale = sr.getScale();

            int width = img.cols - (img.cols % scale);
            int height = img.rows - (img.rows % scale);
            Mat cropped = img(Rect(0, 0, width, height));
            Mat imgDownscaled;

            resize(cropped, imgDownscaled, Size(), 1.0/scale, 1.0/scale);

            Mat imgUpscaled;

			double elapsed = 0.0;

            #ifdef _WIN32
                LARGE_INTEGER st, et, el, fq;
            #else
                double start, end;
                timespec ts_beg, ts_end;
            #endif

            #ifdef _WIN32
                QueryPerformanceFrequency(&fq);
                QueryPerformanceCounter(&st);

                sr.upsample(imgDownscaled, imgUpscaled);

                QueryPerformanceCounter(&et);
                el.QuadPart = et.QuadPart - st.QuadPart;
				elapsed = static_cast<double>(el.QuadPart) / static_cast<double>(fq.QuadPart);
            #else
                clock_gettime(CLOCK_REALTIME, &ts_beg);
                start = ts_beg.tv_sec + ts_beg.tv_nsec;

                sr.upsample(imgDownscaled, imgUpscaled);

                clock_gettime(CLOCK_REALTIME, &ts_end);
                end = ts_end.tv_sec + ts_end.tv_nsec;
                elapsed = ((double)end-start) / 1e9;
            #endif

            double psnr_value = psnr(imgUpscaled, cropped);
            double ssim_value = ssim(imgUpscaled, cropped);

            psnrValues.push_back(psnr_value);
            ssimValues.push_back(ssim_value);
            perfValues.push_back(elapsed);

            if( showOutput )
            {
                std::cout << sr.getAlgorithm() << ":" << std::endl;
                std::cout << "Upsampling time: " << elapsed << std::endl;
                std::cout << "PSNR: " << psnr_value << std::endl;
                std::cout << "SSIM: " << ssim_value << std::endl;
                std::cout << "----------------------" << std::endl;
            }

            //BICUBIC
            Mat bicubic;

            #ifdef _WIN32
                QueryPerformanceFrequency(&fq);
                QueryPerformanceCounter(&st);

                resize(imgDownscaled, bicubic, Size(), scale, scale, INTER_CUBIC);

                QueryPerformanceCounter(&et);
                el.QuadPart = et.QuadPart - st.QuadPart;
                elapsed = static_cast<double>(el.QuadPart) / static_cast<double>(fq.QuadPart);
            #else
                clock_gettime(CLOCK_REALTIME, &ts_beg);
                start = ts_beg.tv_sec + ts_beg.tv_nsec;

                resize(imgDownscaled, bicubic, Size(), scale, scale, INTER_CUBIC );

                clock_gettime(CLOCK_REALTIME, &ts_end);
                end = ts_end.tv_sec + ts_end.tv_nsec;
                elapsed = ((double)end-start) / 1e9;
            #endif

            psnr_value = psnr(bicubic, cropped);
            ssim_value = ssim(bicubic, cropped);

            psnrValues.push_back(psnr_value);
            ssimValues.push_back(ssim_value);
            perfValues.push_back(elapsed);

            if( showOutput )
            {
                std::cout << "Bicubic \n" << "Upsampling time: " << elapsed << std::endl;
                std::cout << "PSNR: " << psnr_value << std::endl;
                std::cout << "SSIM: " << ssim_value << std::endl;
                std::cout << "----------------------" << std::endl;
            }

            //NEAREST NEIGHBOR
            Mat nearest;

            #ifdef _WIN32
                QueryPerformanceFrequency(&fq);
                QueryPerformanceCounter(&st);

                resize(imgDownscaled, nearest, Size(), scale, scale, INTER_NEAREST);

                QueryPerformanceCounter(&et);
                el.QuadPart = et.QuadPart - st.QuadPart;
                elapsed = static_cast<double>(el.QuadPart) / static_cast<double>(fq.QuadPart);
            #else
                clock_gettime(CLOCK_REALTIME, &ts_beg);
                start = ts_beg.tv_sec + ts_beg.tv_nsec;

                resize(imgDownscaled, nearest, Size(), scale, scale, INTER_NEAREST );

                clock_gettime(CLOCK_REALTIME, &ts_end);
                end = ts_end.tv_sec + ts_end.tv_nsec;
                elapsed = ((double)end-start) / 1e9;
            #endif

            psnr_value = psnr(nearest, cropped);
            ssim_value = ssim(nearest, cropped);

            psnrValues.push_back(psnr_value);
            ssimValues.push_back(ssim_value);
            perfValues.push_back(elapsed);

            if( showOutput )
            {
                std::cout << "Nearest neighbor \n" << "Upsampling time: " << elapsed << std::endl;
                std::cout << "PSNR: " << psnr_value << std::endl;
                std::cout << "SSIM: " << ssim_value << std::endl;
                std::cout << "----------------------" << std::endl;
            }

            //LANCZOS
            Mat lanczos;

            #ifdef _WIN32
                QueryPerformanceFrequency(&fq);
                QueryPerformanceCounter(&st);

                resize(imgDownscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4 );

                QueryPerformanceCounter(&et);
                el.QuadPart = et.QuadPart - st.QuadPart;
                elapsed = static_cast<double>(el.QuadPart) / static_cast<double>(fq.QuadPart);
            #else
                clock_gettime(CLOCK_REALTIME, &ts_beg);
                start = ts_beg.tv_sec + ts_beg.tv_nsec;

                resize(imgDownscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4 );

                clock_gettime(CLOCK_REALTIME, &ts_end);
                end = ts_end.tv_sec + ts_end.tv_nsec;
                elapsed = ((double)end-start) / 1e9;
            #endif

            psnr_value = psnr(lanczos, cropped);
            ssim_value = ssim(lanczos, cropped);

            psnrValues.push_back(psnr_value);
            ssimValues.push_back(ssim_value);
            perfValues.push_back(elapsed);

            if( showOutput )
            {
                std::cout << "Lanczos \n" << "Upsampling time: " << elapsed << std::endl;
                std::cout << "PSNR: " << psnr_value << std::endl;
                std::cout << "SSIM: " << ssim_value << std::endl;

                std::cout << "-----------------------------------------------" << std::endl;
            }

            if( showImg )
            {
                std::vector<Mat> imgs{ imgUpscaled, bicubic, nearest, lanczos };
                std::vector<String> titles{ sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos" };
                showBenchmark(cropped, imgs, "Benchmarking", Size(imgUpscaled.cols, imgUpscaled.rows), titles, psnrValues, ssimValues, perfValues);
            }
        }

        void DnnSuperResQuality::showBenchmark(Mat orig, std::vector<Mat> images, std::string title, Size imageSize,
                                                const std::vector<String> imageTitles,
                                                const std::vector<double> psnrValues,
                                                const std::vector<double> ssimValues,
                                                const std::vector<double> perfValues)
        {
            CV_Assert(static_cast<int>(images.size()) > 0);
            CV_Assert(!orig.empty());

            int len = static_cast<int>(images.size());

            if ( len > 9 )
            {
                std::cout << "showBenchmark() supports up to 9 images" << std::endl;
                return;
            }

            bool showTitles = false;
            bool showPSNR = false;
            bool showSSIM = false;
            bool showPerf = false;

            if( imageTitles.size() == images.size() )
                showTitles = true;

            if( psnrValues.size() == images.size() )
                showPSNR = true;

            if( psnrValues.size() == images.size() )
                showSSIM = true;

            if( psnrValues.size() == images.size() )
                showPerf = true;

            int cols, rows;

            if ( len == 1 )
            {
                cols = 1;
                rows = 1;
            }
            else if ( len == 2 )
            {
                cols = 2;
                rows = 1;
            }
            else if ( len == 3 || len == 4 )
            {
                cols = 2;
                rows = 2;
            }
            else if ( len == 5 || len == 6 )
            {
                cols = 3;
                rows = 2;
            }
            else
            {
                cols = 3;
                rows = 3;
            }

            Mat fullImage = Mat::zeros(Size((cols * 10) + imageSize.width * cols, (rows * 10) + imageSize.height * rows), images[0].type());

            std::stringstream ss;
            int h_ = -1;
            for (int i = 0; i < len; i++)
            {
                CV_Assert(!images[i].empty());

                int fontStart = 15;
                int w_ = i % cols;
                if(i % cols == 0)
                    h_++;

                Rect ROI((w_  * (10 + imageSize.width)), (h_  * (10 + imageSize.height)), imageSize.width, imageSize.height);
                Mat tmp;
                resize(images[i], tmp, Size(ROI.width, ROI.height));

                if( showTitles )
                {
                    ss << imageTitles[i];
                    putText(tmp,
                            ss.str(),
                            Point(5,fontStart),
                            fontFace,
                            fontScale,
                            fontColor,
                            1,
                            16);

                    ss.str("");
                    fontStart += 20;
                }

                if( showPSNR )
                {
                    ss << "PSNR: " << psnrValues[i];
                    putText(tmp,
                            ss.str(),
                            Point(5,fontStart),
                            fontFace,
                            fontScale,
                            fontColor,
                            1,
                            16);

                    ss.str("");
                    fontStart += 20;
                }

                if( showSSIM )
                {
                    ss << "SSIM: " << ssimValues[i];
                    putText(tmp,
                            ss.str(),
                            Point(5,fontStart),
                            fontFace,
                            fontScale,
                            fontColor,
                            1,
                            16);

                    ss.str("");
                    fontStart += 20;
                }

                if( showPerf )
                {
                    ss << "Speed: " << perfValues[i];
                    putText(tmp,
                            ss.str(),
                            Point(5,fontStart),
                            fontFace,
                            fontScale,
                            fontColor,
                            1,
                            16);

                    ss.str("");
                }

                tmp.copyTo(fullImage(ROI));
            }

            namedWindow(title, 1);
            imshow(title, fullImage);
            waitKey();
        }
    }
}