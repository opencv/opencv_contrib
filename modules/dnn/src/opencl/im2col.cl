__kernel void im2col(__global const float *im_src,
                     int channels, int height_inp, int width_inp, 
                     int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                     int height_out, int width_out,
                     __global float *im_col
                    )
{
    int index = get_global_id(0);
    int j_out = index % width_out;
    int i_out = (index / width_out) % height_out;
    int c_inp = (index / width_out) / height_out;
    
    int c_out = c_inp * kernel_h * kernel_w;
    int i_inp = i_out * stride_h - pad_h;
    int j_inp = j_out * stride_w - pad_w;

    im_col += (c_out * height_out + i_out) * width_out + j_out;
    im_src += (c_inp * height_inp + i_inp) * width_inp + j_inp;

    for (int ki = 0; ki < kernel_h; ++ki)
        for (int kj = 0; kj < kernel_w; ++kj) {
            int i = i_inp + ki;
            int j = j_inp + kj;
            *im_col = (h >= 0 && w >= 0 && h < height_inp && w < width_inp) ?
                im_src[i * width_inp + j] : 0;
            im_col += height_out * width_out;
      }
}

}