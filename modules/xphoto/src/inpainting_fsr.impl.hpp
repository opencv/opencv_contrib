// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This is not a standalone header, see inpainting.cpp

namespace cv
{
namespace xphoto
{

struct fsr_parameters
{
    // default variables
    int block_size = 16;
    double conc_weighting = 0.5;
    double rhos[4] = { 0.80, 0.70, 0.66, 0.64 };
    double threshold_stddev_Y[3] = { 0.014, 0.030, 0.090 };
    double threshold_stddev_Cx[3] = { 0.006, 0.010, 0.028 };
    // quality profile dependent variables
    int block_size_min, fft_size, max_iter, min_iter, iter_const;
    double orthogonality_correction;
    fsr_parameters(const int quality)
    {
        if (quality == xphoto::INPAINT_FSR_BEST)
        {
            block_size_min = 2;
            fft_size = 64;
            max_iter = 400;
            min_iter = 50;
            iter_const = 2000;
            orthogonality_correction = 0.2;
        }
        else if (quality == xphoto::INPAINT_FSR_FAST)
        {
            block_size_min = 4;
            fft_size = 32;
            max_iter = 100;
            min_iter = 20;
            iter_const = 1000;
            orthogonality_correction = 0.5;
        }
        else
        {
            CV_Error(Error::StsBadArg, "Unknown quality level set, supported: FAST, BEST");

        }
    }
};


static void
icvSgnMat(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.size(), CV_64F);
    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            double curr_val = src.at<double>(y,x);
            if (curr_val > 0)
            {
                dst.at<double>(y,x) = 1;
            }
            else if (curr_val)
            {
                dst.at<double>(y,x) = -1;
            }
        }
    }
}


static double
icvStandardDeviation(const Mat& distorted_block_2d, const Mat& error_mask_2d) {
    if (countNonZero(error_mask_2d) < 1)
    {
        return NAN; // block with no undistorted pixels shouldn't be chosen for processing (only if block_size_min is reached)
    }
    Scalar tmp_stddev, tmp_mean;
    Mat mask8u;
    error_mask_2d.convertTo(mask8u, CV_8U, 2.0);
    meanStdDev(distorted_block_2d, tmp_mean, tmp_stddev, mask8u);
    double sigma_n = tmp_stddev[0] / 255;
    if (sigma_n < 0)
    {
        sigma_n = 0;
    }
    else if (sigma_n > 1)
    {
        sigma_n = 1;
    }
    return sigma_n;
}

static void
icvExtrapolateBlock(Mat& distorted_block, Mat& error_mask, fsr_parameters& fsr_params, double rho, double normedStdDev, Mat& extrapolated_block)
{
    double fft_size = fsr_params.fft_size;
    double orthogonality_correction = fsr_params.orthogonality_correction;
    int M = distorted_block.rows;
    int N = distorted_block.cols;
    int fft_x_offset = cvFloor((fft_size - N) / 2);
    int fft_y_offset = cvFloor((fft_size - M) / 2);

    // weighting function
    Mat w = Mat::zeros(fsr_params.fft_size, fsr_params.fft_size, CV_64F);
    error_mask.copyTo(w(Range(fft_y_offset, fft_y_offset + M), Range(fft_x_offset, fft_x_offset + N)));
    for (int u = 0; u < fft_size; ++u)
    {
        for (int v = 0; v < fft_size; ++v)
        {
            w.at<double>(u, v) *= std::pow(rho, std::sqrt(std::pow(u + 0.5 - (fft_y_offset + M / 2), 2) + std::pow(v + 0.5 - (fft_x_offset + N / 2), 2)));
        }
    }
    Mat W;
    dft(w, W, DFT_COMPLEX_OUTPUT);
    Mat W_padded;
    hconcat(W, W, W_padded);
    vconcat(W_padded, W_padded, W_padded);

    // frequency weighting
    Mat frequency_weighting = Mat::ones(fsr_params.fft_size, fsr_params.fft_size / 2 + 1, CV_64F);
    for (int y = 0; y < fft_size; ++y)
    {
        for (int x = 0; x < (fft_size / 2 + 1); ++x)
        {
            double y2 = fft_size / 2 - std::abs(y - fft_size / 2);
            double x2 = fft_size / 2 - std::abs(x - fft_size / 2);
            frequency_weighting.at<double>(y, x) = 1 - std::sqrt(x2*x2 + y2 * y2)*std::sqrt(2) / fft_size;
        }
    }
    // pad image to fft window size
    Mat f(Size(fsr_params.fft_size, fsr_params.fft_size), CV_64F, Scalar::all(0));
    distorted_block.copyTo(f(Range(fft_y_offset, fft_y_offset + M), Range(fft_x_offset, fft_x_offset + N)));

    // create initial model
    Mat G = Mat::zeros(fsr_params.fft_size, fsr_params.fft_size, CV_64FC2); // complex

    // calculate initial residual
    Mat Rw_tmp, Rw;
    dft(f.mul(w), Rw_tmp, DFT_COMPLEX_OUTPUT);
    Rw = Rw_tmp(Range(0, fsr_params.fft_size), Range(0, fsr_params.fft_size / 2 + 1));

    // estimate ideal number of iterations (GenserIWSSIP2017)
    // calculate stddev if not available (e.g., for smallest block size)
    if (normedStdDev == 0) {
        normedStdDev = icvStandardDeviation(distorted_block, error_mask);
    }
    int num_iters = cvRound(fsr_params.iter_const * normedStdDev);
    if (num_iters < fsr_params.min_iter) {
        num_iters = fsr_params.min_iter;
    }
    else if (num_iters > fsr_params.max_iter) {
        num_iters = fsr_params.max_iter;
    }

    int iter_counter = 0;
    while (iter_counter < num_iters)
    { // Spectral Constrained FSE (GenserIWSSIP2018)
        Mat projection_distances(Rw.size(), CV_64F);
        Mat Rw_mag = Mat(Rw.size(), CV_64F);
        std::vector<Mat> channels(2);
        split(Rw, channels);
        magnitude(channels[0], channels[1], Rw_mag);
        projection_distances = Rw_mag.mul(frequency_weighting);

        double minVal, maxVal;
        int maxLocx = -1;
        int maxLocy = -1;
        minMaxLoc(projection_distances, &minVal, &maxVal);

        for (int y = 0; y < projection_distances.rows; ++y)
        { // assure that first appearance of max Value is selected
            for (int x = 0; x < projection_distances.cols; ++x)
            {
                if (std::abs(projection_distances.at<double>(y, x) - maxVal) < 0.001)
                {
                    maxLocy = y;
                    maxLocx = x;
                    break;
                }
            }
            if (maxLocy != -1)
            {
                break;
            }
        }
        int bf2select = maxLocy + maxLocx * projection_distances.rows;
        int v = static_cast<int>(std::max(0.0, std::floor(bf2select / fft_size)));
        int u = static_cast<int>(std::max(0, bf2select % fsr_params.fft_size));


        // exclude second half of first and middle col
        if ((v == 0 && u > fft_size / 2) || (v == fft_size / 2 && u > fft_size / 2))
        {
            int u_prev = u;
            u = fsr_params.fft_size - u;
            Rw.at<std::complex<double> >(u, v) = std::conj(Rw.at<std::complex<double> >(u_prev, v));
        }

        // calculate complex conjugate solution
        int u_cj = -1;
        int v_cj = -1;
        // fill first lower col (copy from first upper col)
        if (u >= 1 && u < fft_size / 2 && v == 0)
        {
            u_cj = fsr_params.fft_size - u;
            v_cj = v;
        }
        // fill middle lower col (copy from first middle col)
        if (u >= 1 && u < fft_size / 2 && v == fft_size / 2)
        {
            u_cj = fsr_params.fft_size - u;
            v_cj = v;
        }
        // fill first row right (copy from first row left)
        if (u == 0 && v >= 1 && v < fft_size / 2)
        {
            u_cj = u;
            v_cj = fsr_params.fft_size - v;
        }
        // fill middle row right (copy from middle row left)
        if (u == fft_size / 2 && v >= 1 && v < fft_size / 2)
        {
            u_cj = u;
            v_cj = fsr_params.fft_size - v;
        }
        // fill cell upper right (copy from lower cell left)
        if (u >= fft_size / 2 + 1 && v >= 1 && v < fft_size / 2)
        {
            u_cj = fsr_params.fft_size - u;
            v_cj = fsr_params.fft_size - v;
        }
        // fill cell lower right (copy from upper cell left)
        if (u >= 1 && u < fft_size / 2 && v >= 1 && v < fft_size / 2)
        {
            u_cj = fsr_params.fft_size - u;
            v_cj = fsr_params.fft_size - v;
        }

        /// add coef to model and update residual
        if (u_cj != -1 && v_cj != -1)
        {
            std::complex< double> expansion_coefficient = orthogonality_correction * Rw.at< std::complex<double> >(u, v) / W.at<std::complex<double> >(0, 0);
            G.at< std::complex<double> >(u, v) += fft_size * fft_size * expansion_coefficient;
            G.at< std::complex<double> >(u_cj, v_cj) = std::conj(G.at< std::complex<double> >(u, v));

            Mat expansion_mat(Rw.size(), CV_64FC2, Scalar(expansion_coefficient.real(), expansion_coefficient.imag()));
            Mat W_tmp1 = W_padded(Range(fsr_params.fft_size - u, fsr_params.fft_size - u + Rw.rows), Range(fsr_params.fft_size - v, fsr_params.fft_size - v + Rw.cols));
            Mat W_tmp2 = W_padded(Range(fsr_params.fft_size - u_cj, fsr_params.fft_size - u_cj + Rw.rows), Range(fsr_params.fft_size - v_cj, fsr_params.fft_size - v_cj + Rw.cols));
            Mat res_1(W_tmp1.size(), W_tmp1.type());
            mulSpectrums(expansion_mat, W_tmp1, res_1, 0);
            expansion_mat.setTo(Scalar(expansion_coefficient.real(), -expansion_coefficient.imag()));
            Mat res_2(W_tmp1.size(), W_tmp1.type());
            mulSpectrums(expansion_mat, W_tmp2, res_2, 0);
            Rw -= res_1 + res_2;

            ++iter_counter; // ... as two basis functions were added
        }
        else
        {
            std::complex<double> expansion_coefficient = orthogonality_correction * Rw.at< std::complex<double> >(u, v) / W.at< std::complex<double> >(0, 0);
            G.at< std::complex<double> >(u, v) += fft_size * fft_size * expansion_coefficient;
            Mat expansion_mat(Rw.size(), CV_64FC2, Scalar(expansion_coefficient.real(), expansion_coefficient.imag()));
            Mat W_tmp = W_padded(Range(fsr_params.fft_size - u, fsr_params.fft_size - u + Rw.rows), Range(fsr_params.fft_size - v, fsr_params.fft_size - v + Rw.cols));
            Mat res_tmp(W_tmp.size(), W_tmp.type());
            mulSpectrums(expansion_mat, W_tmp, res_tmp, 0);
            Rw -= res_tmp;

        }
        ++iter_counter;
    }

    // get pixels from model
    Mat g;
    idft(G, g, DFT_SCALE);

    // extract reconstructed pixels
    Mat g_real(M, N, CV_64F);
    for (int x = 0; x < M; ++x)
    {
        for (int y = 0; y < N; ++y)
        {
            g_real.at<double>(x, y) = g.at< std::complex<double> >(fft_y_offset + x, fft_x_offset + y).real();
        }
    }
    g_real.copyTo(extrapolated_block);
    Mat orig_samples;
    error_mask.convertTo(orig_samples, CV_8U);
    distorted_block.copyTo(extrapolated_block, orig_samples); // copy where orig_samples is nonzero
}


static void
icvGetTodoBlocks(Mat& sampled_img, Mat& sampling_mask, std::vector< std::tuple< int, int > >& set_todo, int block_size, int block_size_min, int border_width, double homo_threshold, Mat& set_process_this_block_size, std::vector< std::tuple< int, int > >& set_later, Mat& sigma_n_array)
{
    std::vector< std::tuple< int, int > > set_now;
    set_later.clear();
    size_t list_length = set_todo.size();
    int img_height = sampled_img.rows;
    int img_width = sampled_img.cols;
    Mat reconstructed_img;
    sampled_img.copyTo(reconstructed_img);

    // calculate block lists
    for (size_t entry = 0; entry < list_length; ++entry)
    {
        int xblock_counter = std::get<0>(set_todo[entry]);
        int yblock_counter = std::get<1>(set_todo[entry]);

        int left_border = std::min(xblock_counter*block_size, border_width);
        int top_border = std::min(yblock_counter*block_size, border_width);
        int right_border = std::max(0, std::min(img_width - (xblock_counter + 1)*block_size, border_width));
        int bottom_border = std::max(0, std::min(img_height - (yblock_counter + 1)*block_size, border_width));

        // extract blocks from images
        Mat distorted_block_2d = reconstructed_img(Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), Range(xblock_counter*block_size - left_border, std::min(img_width, (xblock_counter*block_size + block_size + right_border))));
        Mat error_mask_2d = sampling_mask(Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), Range(xblock_counter*block_size - left_border, std::min(img_width, (xblock_counter*block_size + block_size + right_border))));

        // determine normalized and weighted standard deviation
        if (block_size > block_size_min && xblock_counter < sigma_n_array.cols && yblock_counter < sigma_n_array.rows)
        {
            double sigma_n = icvStandardDeviation(distorted_block_2d, error_mask_2d);
            sigma_n_array.at<double>( yblock_counter, xblock_counter) = sigma_n;

            // homogeneous case
            if (sigma_n < homo_threshold)
            {
                set_now.emplace_back(xblock_counter, yblock_counter);
                set_process_this_block_size.at<double>(yblock_counter, xblock_counter) = 255;

            }
            else
            {
                int yblock_counter_quadernary = yblock_counter * 2;
                int xblock_counter_quadernary = xblock_counter * 2;
                int yblock_offset = 0;
                int xblock_offset = 0;

                for (int quader_counter = 0; quader_counter < 4; ++quader_counter)
                {
                    if (quader_counter == 0)
                    {
                        yblock_offset = 0;
                        xblock_offset = 0;
                    }
                    else if (quader_counter == 1)
                    {
                        yblock_offset = 0;
                        xblock_offset = 1;
                    }
                    else if (quader_counter == 2)
                    {
                        yblock_offset = 1;
                        xblock_offset = 0;
                    }
                    else if (quader_counter == 3)
                    {
                        yblock_offset = 1;
                        xblock_offset = 1;
                    }

                    set_later.emplace_back(xblock_counter_quadernary + xblock_offset, yblock_counter_quadernary + yblock_offset);
                }

            }
        }

    }
}


static void
icvDetermineProcessingOrder(
    const Mat& _sampled_img, const Mat& _sampling_mask,
    const int quality, const std::string& channel, Mat& reconstructed_img
)
{
    fsr_parameters fsr_params(quality);
    int block_size = fsr_params.block_size;
    int block_size_max = fsr_params.block_size;
    int block_size_min = fsr_params.block_size_min;
    double conc_weighting = fsr_params.conc_weighting;
    int fft_size = fsr_params.fft_size;
    double rho = fsr_params.rhos[0];
    Mat sampled_img, sampling_mask;
    _sampled_img.convertTo(sampled_img, CV_64F);
    reconstructed_img = sampled_img.clone();

    _sampling_mask.convertTo(sampling_mask, CV_64F);

    double threshold_stddev_LUT[3];
    if (channel == "Y")
    {
        std::copy(fsr_params.threshold_stddev_Y, fsr_params.threshold_stddev_Y + 3, threshold_stddev_LUT);
    }
    else if (channel == "Cx")
    {
        std::copy(fsr_params.threshold_stddev_Cx, fsr_params.threshold_stddev_Cx + 3, threshold_stddev_LUT);
    }
    else
    {
        CV_Error(Error::StsBadArg, "channel type unsupported!");
    }


    double threshold_stddev = threshold_stddev_LUT[0];

    std::vector< std::tuple< int, int > > set_later;
    int img_height = sampled_img.rows;
    int img_width = sampled_img.cols;

    // initial scan of distorted blocks
    std::vector< std::tuple< int, int > > set_todo;
    int blocks_column = divUp(img_height, block_size);
    int blocks_line = divUp(img_width, block_size);
    for (int y = 0; y < blocks_column; ++y)
    {
        for (int x = 0; x < blocks_line; ++x)
        {
            Mat curr_block = sampling_mask(Range(y*block_size, std::min(img_height, (y + 1)*block_size)), Range(x*block_size, std::min(img_width, (x + 1)*block_size)));
            double min_block, max_block;
            minMaxLoc(curr_block, &min_block, &max_block);
            if (min_block == 0)
            {
                set_todo.emplace_back(x, y);
            }
        }
    }

    // loop over all distorted blocks and extrapolate them depending on
    // their block size
    int border_width = 0;
    while (block_size >= block_size_min)
    {
        int blocks_per_column = cvCeil(img_height / block_size);
        int blocks_per_line = cvCeil(img_width / block_size);
        Mat nen_array = Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
        Mat proc_array = Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
        Mat sigma_n_array = Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
        Mat set_process_this_block_size = Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
        if (block_size > block_size_min)
        {
            if (block_size < block_size_max)
            {
                set_todo = set_later;
            }
            border_width = cvFloor(fft_size - block_size) / 2;
            icvGetTodoBlocks(sampled_img, sampling_mask, set_todo, block_size, block_size_min, border_width, threshold_stddev, set_process_this_block_size, set_later, sigma_n_array);
        }
        else
        {
            set_process_this_block_size.setTo(Scalar(255));
        }

        // if block to be extrapolated, increase nen of neighboring pixels
        for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
        {
            for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
            {
                Mat curr_block = sampling_mask(Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size)));
                double min_block, max_block;
                minMaxLoc(curr_block, &min_block, &max_block);
                if (min_block == 0)
                {
                    if (yblock_counter > 0 && xblock_counter > 0)
                    {
                        nen_array.at<double>(yblock_counter - 1, xblock_counter - 1)++;
                    }
                    if (yblock_counter > 0)
                    {
                        nen_array.at<double>(yblock_counter - 1, xblock_counter)++;
                    }
                    if (yblock_counter > 0 && xblock_counter < (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter - 1, xblock_counter + 1)++;
                    }
                    if (xblock_counter > 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter - 1)++;
                    }
                    if (xblock_counter < (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter + 1)++;
                    }
                    if (yblock_counter < (blocks_per_column - 1) && xblock_counter>0)
                    {
                        nen_array.at<double>(yblock_counter + 1, xblock_counter - 1)++;
                    }
                    if (yblock_counter < (blocks_per_column - 1))
                    {
                        nen_array.at<double>(yblock_counter + 1, xblock_counter)++;
                    }
                    if (yblock_counter < (blocks_per_column - 1) && xblock_counter < (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter + 1, xblock_counter + 1)++;
                    }
                }
            }
        }

        // determine if block itself has to be extrapolated
        for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
        {
            for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
            {
                Mat curr_block = sampling_mask(Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size)));
                double min_block, max_block;
                minMaxLoc(curr_block, &min_block, &max_block);
                if (min_block != 0)
                {
                    nen_array.at<double>(yblock_counter, xblock_counter) = -1;
                }
                else
                {
                // if border block, increase nen respectively
                    if (yblock_counter == 0 && xblock_counter == 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                    }
                    if (yblock_counter == 0 && xblock_counter == (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                    }
                    if (yblock_counter == (blocks_per_column - 1) && xblock_counter == 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                    }
                    if (yblock_counter == (blocks_per_column - 1) && xblock_counter == (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                    }
                    if (yblock_counter == 0 && xblock_counter != 0 && xblock_counter != (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                    }
                    if (yblock_counter == (blocks_per_column - 1) && xblock_counter != 0 && xblock_counter != (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                    }
                    if (yblock_counter != 0 && yblock_counter != (blocks_per_column - 1) && xblock_counter == 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                    }
                    if (yblock_counter != 0 && yblock_counter != (blocks_per_column - 1) && xblock_counter == (blocks_per_line - 1))
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                    }
                }
            }
        }

        // if all blocks have 8 not extrapolated neighbors, penalize nen of blocks without any known samples by one
        double min_nen_tmp, max_nen_tmp;
        minMaxLoc(nen_array, &min_nen_tmp, &max_nen_tmp);
        if (min_nen_tmp == 8) {
            for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
            {
                for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
                {
                    Mat curr_block = sampling_mask(Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size)));
                    double min_block, max_block;
                    minMaxLoc(curr_block, &min_block, &max_block);
                    if (max_block == 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter)++;
                    }
                }
            }
        }

        // do actual processing per block
        int all_blocks_finished = 0;
        while (all_blocks_finished == 0) {
            // clear proc_array
            proc_array.setTo(Scalar(1));

            // determine blocks to extrapolate
            double min_nen = 99;
            int bl_counter = 0;
            // add all homogeneous blocks that shall be processed to list
            // using same priority
            // begins with highest prioroty or lowest nen array value
            std::vector< std::tuple< int, int > > block_list;
            for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
            {
                for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
                {
            // decision if block contains errors
                    double tmp_val = nen_array.at<double>(yblock_counter, xblock_counter);
                    if (tmp_val >= 0 && tmp_val < min_nen && set_process_this_block_size.at<double>(yblock_counter, xblock_counter) == 255) {
                        bl_counter = 0;
                        block_list.clear();
                        min_nen = tmp_val;
                        proc_array.setTo(Scalar(1));
                    }
                    if (tmp_val == min_nen && proc_array.at<double>(yblock_counter, xblock_counter) != 0 && set_process_this_block_size.at<double>(yblock_counter, xblock_counter) == 0) {
                        nen_array.at<double>(yblock_counter, xblock_counter) = -1;
                    }
                    if (tmp_val == min_nen && proc_array.at<double>(yblock_counter, xblock_counter) != 0 && set_process_this_block_size.at<double>(yblock_counter, xblock_counter) != 0) {
                        block_list.emplace_back(yblock_counter, xblock_counter);
                        bl_counter++;
                        // block neighboring blocks from processing
                        if (yblock_counter > 0 && xblock_counter > 0)
                        {
                            proc_array.at<double>(yblock_counter - 1, xblock_counter - 1) = 0;
                        }
                        if (yblock_counter > 0)
                        {
                            proc_array.at<double>(yblock_counter - 1, xblock_counter) = 0;
                        }
                         if (yblock_counter > 0 && xblock_counter > 0)
                        {
                            proc_array.at<double>(yblock_counter - 1, xblock_counter - 1) = 0;
                        }
                        if (yblock_counter > 0)
                        {
                            proc_array.at<double>(yblock_counter - 1, xblock_counter) = 0;
                        }
                        if (yblock_counter > 0 && xblock_counter < (blocks_per_line - 1))
                        {
                            proc_array.at<double>(yblock_counter - 1, xblock_counter + 1) = 0;
                        }
                        if (xblock_counter > 0)
                        {
                            proc_array.at<double>(yblock_counter, xblock_counter - 1) = 0;
                        }
                        if (xblock_counter < (blocks_per_line - 1))
                        {
                            proc_array.at<double>(yblock_counter, xblock_counter + 1) = 0;
                        }
                        if (yblock_counter < (blocks_per_column - 1) && xblock_counter > 0)
                        {
                            proc_array.at<double>(yblock_counter + 1, xblock_counter - 1) = 0;
                        }
                        if (yblock_counter < (blocks_per_column - 1))
                        {
                            proc_array.at<double>(yblock_counter + 1, xblock_counter) = 0;
                        }
                        if (yblock_counter < (blocks_per_column - 1) && xblock_counter < (blocks_per_line - 1))
                        {
                            proc_array.at<double>(yblock_counter + 1, xblock_counter + 1) = 0;
                        }
                    }
                }
            }
            int max_bl_counter = bl_counter;
            block_list.emplace_back(-1, -1);
            if (bl_counter == 0)
            {
                all_blocks_finished = 1;
            }
            // blockwise extrapolation of all blocks that can be processed in parallel
            for (bl_counter = 0; bl_counter < max_bl_counter; ++bl_counter)
            {
                int yblock_counter = std::get<0>(block_list[bl_counter]);
                int xblock_counter = std::get<1>(block_list[bl_counter]);

                // calculation of the extrapolation area's borders
                int left_border = std::min(xblock_counter*block_size, border_width);
                int top_border = std::min(yblock_counter*block_size, border_width);
                int right_border = std::max(0, std::min(img_width - (xblock_counter + 1)*block_size, border_width));
                int bottom_border = std::max(0, std::min(img_height - (yblock_counter + 1)*block_size, border_width));

                // extract blocks from images
                Mat distorted_block_2d = reconstructed_img(Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), Range(xblock_counter*block_size - left_border, std::min(img_width, (xblock_counter*block_size + block_size + right_border))));
                Mat error_mask_2d = sampling_mask(Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), Range(xblock_counter*block_size - left_border, std::min(img_width, xblock_counter*block_size + block_size + right_border)));
                // get actual stddev value as it is needed to estimate the
                // best number of iterations
                double sigma_n_a = sigma_n_array.at<double>(yblock_counter, xblock_counter);

                // actual extrapolation
                Mat extrapolated_block_2d;
                icvExtrapolateBlock(distorted_block_2d, error_mask_2d, fsr_params, rho, sigma_n_a, extrapolated_block_2d);

                // update image and mask
                extrapolated_block_2d(Range(top_border, extrapolated_block_2d.rows - bottom_border), Range(left_border, extrapolated_block_2d.cols - right_border)).copyTo(reconstructed_img(Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size))));

                Mat signs;
                icvSgnMat(error_mask_2d(Range(top_border, error_mask_2d.rows - bottom_border), Range(left_border, error_mask_2d.cols - right_border)), signs);
                Mat tmp_mask = error_mask_2d(Range(top_border, error_mask_2d.rows - bottom_border), Range(left_border, error_mask_2d.cols - right_border)) + (1 - signs) *conc_weighting;
                tmp_mask.copyTo(sampling_mask(Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size))));

                // update nen-array
                nen_array.at<double>(yblock_counter, xblock_counter) = -1;
                if (yblock_counter > 0 && xblock_counter > 0)
                {
                    nen_array.at<double>(yblock_counter - 1, xblock_counter - 1)--;
                }
                if (yblock_counter > 0)
                {
                    nen_array.at<double>(yblock_counter - 1, xblock_counter)--;
                }
                if (yblock_counter > 0 && xblock_counter < blocks_per_line - 1)
                {
                    nen_array.at<double>(yblock_counter - 1, xblock_counter + 1)--;
                }
                if (xblock_counter > 0)
                {
                    nen_array.at<double>(yblock_counter, xblock_counter - 1)--;
                }
                if (xblock_counter < blocks_per_line - 1)
                {
                    nen_array.at<double>(yblock_counter, xblock_counter + 1)--;
                }
                if (yblock_counter < blocks_per_column - 1 && xblock_counter>0)
                {
                    nen_array.at<double>(yblock_counter + 1, xblock_counter - 1)--;
                }
                if (yblock_counter < blocks_per_column - 1)
                {
                    nen_array.at<double>(yblock_counter + 1, xblock_counter)--;
                }
                if (yblock_counter < blocks_per_column - 1 && xblock_counter < blocks_per_line - 1)
                {
                    nen_array.at<double>(yblock_counter + 1, xblock_counter + 1)--;
                }

            }

        }

        // set parameters for next extrapolation tasks (higher texture)
        block_size = block_size / 2;
        border_width = (fft_size - block_size) / 2;
        if (block_size == 8)
        {
            threshold_stddev = threshold_stddev_LUT[1];
            rho = fsr_params.rhos[1];
        }
        if (block_size == 4)
        {
            threshold_stddev = threshold_stddev_LUT[2];
            rho = fsr_params.rhos[2];
        }
        if (block_size == 2)
        {
            rho = fsr_params.rhos[3];
        }

        // terminate function - no heterogeneous blocks left
        if (set_later.empty())
        {
            break;
        }
    }
}


static
void inpaint_fsr(Mat src, const Mat &mask, Mat &dst, const int algorithmType)
{
    CV_Assert(algorithmType == xphoto::INPAINT_FSR_BEST || algorithmType == xphoto::INPAINT_FSR_FAST);
    CV_Check(src.channels(), src.channels() == 1 || src.channels() == 3, "");
    switch (src.type())
    {
        case CV_8UC1:
        case CV_8UC3:
            break;
        case CV_16UC1:
        case CV_16UC3:
        {
            double minRange, maxRange;
            minMaxLoc(src, &minRange, &maxRange);
            if (minRange < 0 || maxRange > 65535)
            {
                CV_Error(Error::StsUnsupportedFormat, "Unsupported source image format!");
                break;
            }
            src.convertTo(src, CV_8U, 1/256.0);
            break;
        }
        case CV_32FC1:
        case CV_64FC1:
        case CV_32FC3:
        case CV_64FC3:
        {
            double minRange, maxRange;
            minMaxLoc(src, &minRange, &maxRange);
            if (minRange < -FLT_EPSILON || maxRange > (1.0 + FLT_EPSILON))
            {
                CV_Error(Error::StsUnsupportedFormat, "Unsupported source image format!");
                break;
            }
            src.convertTo(src, CV_8U, 255.0);
            break;
        }
        default:
            CV_Error(Error::StsUnsupportedFormat, "Unsupported source image format!");
            break;
    }
    dst.create(src.size(), src.type());
    Mat mask_01;
    threshold(mask, mask_01, 0.0, 1.0, THRESH_BINARY);
    if (src.channels() == 1)
    { // grayscale image
        Mat y_reconstructed;
        icvDetermineProcessingOrder(src, mask_01, algorithmType, "Y", y_reconstructed);
        y_reconstructed.convertTo(dst, CV_8U);
    }
    else if (src.channels() == 3)
    { // RGB image
        Mat ycrcb;
        cvtColor(src, ycrcb, COLOR_BGR2YCrCb);
        std::vector<Mat> channels(3);
        split(ycrcb, channels);
        Mat y = channels[0];
        Mat cb = channels[2];
        Mat cr = channels[1];
        Mat y_reconstructed, cb_reconstructed, cr_reconstructed;
        y = y.mul(mask_01);
        cb = cb.mul(mask_01);
        cr = cr.mul(mask_01);
        icvDetermineProcessingOrder(y, mask_01, algorithmType, "Y", y_reconstructed);
        icvDetermineProcessingOrder(cb, mask_01, algorithmType, "Cx", cb_reconstructed);
        icvDetermineProcessingOrder(cr, mask_01, algorithmType, "Cx", cr_reconstructed);
        Mat ycrcb_reconstructed;
        y_reconstructed.convertTo(channels[0], CV_8U);
        cr_reconstructed.convertTo(channels[1], CV_8U);
        cb_reconstructed.convertTo(channels[2], CV_8U);
        merge(channels, ycrcb_reconstructed);
        cvtColor(ycrcb_reconstructed, dst, COLOR_YCrCb2BGR);
    }
}

}}  // namespace
