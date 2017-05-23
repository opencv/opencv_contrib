#include "precomp.hpp"

namespace cv { namespace photoeffects {

void antique(InputArray src, OutputArray dst, InputArray texture, float alpha)
{
    CV_Assert((src.type() == CV_8UC3) || (src.type() == CV_32FC3));
    CV_Assert((texture.type() == CV_8UC3) || (texture.type() == CV_32FC3));
    CV_Assert((alpha > 0.0f) && (alpha < 1.0f));
    Mat textureImg = texture.getMat();
    resize(texture,textureImg, src.size(),0,0, INTER_LINEAR);
    Mat m_sepiaKernel = (Mat_<float>(3,3)<<0.272f, 0.534f, 0.131f,
                                           0.349f, 0.686f, 0.168f,
                                           0.393f, 0.769f, 0.189f);
    Mat newSrc = src.getMat();
    transform(newSrc,newSrc,m_sepiaKernel);
    float beta = 1.0f-alpha;
    addWeighted(textureImg, alpha , newSrc, beta, 0.0f, dst);
}

}}
