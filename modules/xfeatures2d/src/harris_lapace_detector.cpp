// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace {

using namespace cv;

bool sort_func(KeyPoint kp1, KeyPoint kp2)
{
    return (kp1.response > kp2.response);
}

class Pyramid
{

protected:
    class Octave
    {
    public:
        std::vector<Mat> layers;
        Octave(std::vector<Mat> layers);
        virtual ~Octave();
        Mat getLayerAt(int i);
    };

    class DOGOctave
    {
    public:
        std::vector<Mat> layers;

        DOGOctave(std::vector<Mat> layers);
        virtual ~DOGOctave();
        Mat getLayerAt(int i);
    };

private:
    std::vector<Octave> octaves;
    std::vector<DOGOctave> DOG_octaves;
    void build(const Mat& img, bool DOG);
public:
    class Params
    {
    public:
        int octavesN;
        int layersN;
        float sigma0;
        int omin;
        float step;
        Params(int octavesN, int layersN, float sigma0, int omin);
        void clear();
    };
    Params params;

    Pyramid(const Mat& img, int octavesN, int layersN = 2, float sigma0 = 1, int omin = 0,
            bool DOG = false);
    Mat getLayer(int octave, int layer);
    Mat getDOGLayer(int octave, int layer);
    float getSigma(int layer);

    virtual ~Pyramid();
    void clear();
};

/**
 * Pyramid class constructor
 * octavesN_: number of octaves
 * layersN_: number of layers before subsampling layer
 * sigma0_: starting sigma (depends on detector's type, i.e. SIFT sigma0 = 1.6, Harris sigma0 = 1)
 * omin_: if omin<0 an octave is added before first octave. In this octave the image size is doubled
 * _DOG: if true, a DOG pyramid is build
 */
Pyramid::Pyramid(const Mat & img, int octavesN_, int layersN_, float sigma0_, int omin_, bool _DOG) :
    params(
        //Need to set the octavesN parameter globally. See issue #1513
        MIN(octavesN_, int(floor(log((double)MIN(img.size().width, img.size().height)) / log(2.0f)))),
        layersN_,
        sigma0_,
        omin_
    )
{
    build(img, _DOG);
}

/**
 * Build gaussian pyramid with layersN_ + 3 layers and 2^(1/layersN_) step between layers
 * each octave is downsampled of a factor of 2
 */

void Pyramid::build(const Mat& img, bool DOG)
{

    Size ksize(0, 0);
    int gsize;

    float sigma0 = params.sigma0;
    float sigma = sigma0;
    int layersN = params.layersN + 3;
    int omin = params.omin;
    float k = params.step;

    /*layer to downsample*/
    int down_lay = int(1 / log(k));

    int octave, layer;
    double sigmaN = 0.5;

    std::vector<Mat> layers, DOG_layers;
    /* standard deviation of current layer*/
    float sigma_curr = sigma;
    /* standard deviation of previous layer*/
    float sigma_prev = sigma;

    if (omin < 0)
    {
        omin = -1;
        Mat tmp_img;
        Mat blurred_img;
        gsize = int(ceil(sigmaN * 3)) * 2 + 1;
        GaussianBlur(img, blurred_img, Size(gsize,gsize), sigmaN);
        resize(blurred_img, tmp_img, ksize, 2, 2, INTER_AREA);
        layers.push_back(tmp_img);

        for (layer = 1; layer < layersN; layer++)
        {
            sigma_curr = getSigma(layer);
            sigma = sqrt(powf(sigma_curr, 2) - powf(sigma_prev, 2));
            Mat prev_lay = layers[layer - 1], curr_lay, DOG_lay;
            /* smoothing is applied on previous layer so sigma_curr^2 = sigma^2 + sigma_prev^2 */
            gsize = int(ceil(sigma * 3)) * 2 + 1;
            GaussianBlur(prev_lay, curr_lay, Size(gsize,gsize), sigma);
            layers.push_back(curr_lay);
            if (DOG)
            {
                absdiff(curr_lay, prev_lay, DOG_lay);
                DOG_layers.push_back(DOG_lay);
            }
            sigma_prev = sigma_curr;

        }
        Octave tmp_oct(layers);
        octaves.push_back(tmp_oct);
        layers.clear();

        if (DOG)
        {
            DOGOctave tmp_DOG_Oct(DOG_layers);
            DOG_octaves.push_back(tmp_DOG_Oct);
            DOG_layers.clear();
        }

    }

    /* Presmoothing on first layer */
    float sb = float(sigmaN) / powf(2.0f, (float) omin);
    sigma = sigma0;
    if (sigma0 > sb)
        sigma = sqrt(sigma0 * sigma0 - sb * sb);

    /*1° step on image*/
    Mat tmpImg;
    gsize = int(ceil(sigma * 3)) * 2 + 1;
    GaussianBlur(img, tmpImg, Size(gsize,gsize), sigma);
    layers.push_back(tmpImg);

    /*for every octave build layers*/
    sigma_prev = sigma;

    for (octave = 0; octave < params.octavesN; octave++)
    {
        for (layer = 1; layer < layersN; layer++)
        {
            sigma_curr = getSigma(layer);
            sigma = sqrt(powf(sigma_curr, 2) - powf(sigma_prev, 2));

            Mat prev_lay = layers[layer - 1], curr_lay, DOG_lay;
            gsize = int(ceil(sigma * 3)) * 2 + 1;
            GaussianBlur(prev_lay, curr_lay, Size(gsize,gsize), sigma);
            layers.push_back(curr_lay);

            if (DOG)
            {
                absdiff(curr_lay, prev_lay, DOG_lay);
                DOG_layers.push_back(DOG_lay);
            }
            sigma_prev = sigma_curr;
        }

        Mat resized_lay;
        resize(layers[down_lay], resized_lay, ksize, 1.0f / 2, 1.0f / 2, INTER_AREA);

        Octave tmp_oct(layers);
        octaves.push_back(tmp_oct);
        if (DOG)
        {
            DOGOctave tmp_DOG_Oct(DOG_layers);
            DOG_octaves.push_back(tmp_DOG_Oct);
            DOG_layers.clear();
        }
        sigma_curr = sigma_prev = sigma0;
        layers.clear();
        layers.push_back(resized_lay);

    }

}

/**
 * Return layer at indicated octave and layer numbers
 */
Mat Pyramid::getLayer(int octave, int layer)
{
    return octaves[octave].getLayerAt(layer);
}

/**
 * Return DOG layer at indicated octave and layer numbers
 */
Mat Pyramid::getDOGLayer(int octave, int layer)
{
    CV_Assert(!DOG_octaves.empty());
    return DOG_octaves[octave].getLayerAt(layer);
}

/**
 * Return sigma value of indicated layer
 * sigma value of layer is the same at each octave
 * i.e. sigma of first layer at each octave is sigma0
 */
float Pyramid::getSigma(int layer)
{

    return powf(params.step, float(layer)) * params.sigma0;
}

/**
 * Destructor of Pyramid class
 */
Pyramid::~Pyramid()
{
    clear();
}

/**
 * Clear octaves and params
 */
void Pyramid::clear()
{
    octaves.clear();
    params.clear();
}

/**
 * Params for Pyramid class
 *
 */
Pyramid::Params::Params(int octavesN_, int layersN_, float sigma0_, int omin_) :
    octavesN(octavesN_), layersN(layersN_), sigma0(sigma0_), omin(omin_)
{
    CV_Assert(layersN > 0 && octavesN_>0);
    step = powf(2, 1.0f / layersN);
}

/**
 * Set to zero all params
 */
void Pyramid::Params::clear()
{
    octavesN = 0;
    layersN = 0;
    sigma0 = 0;
    omin = 0;
    step = 0;
}

/**
 * Create an Octave with layers
 */
Pyramid::Octave::Octave(std::vector<Mat> _layers) : layers(_layers) {}

/**
 * Return the Octave's layer at index i
 */
Mat Pyramid::Octave::getLayerAt(int i)
{
    CV_Assert(i < (int) layers.size());
    return layers[i];
}

Pyramid::Octave::~Octave()
{
}

Pyramid::DOGOctave::DOGOctave(std::vector<Mat> _layers) : layers(_layers) {}

Pyramid::DOGOctave::~DOGOctave()
{
}

Mat Pyramid::DOGOctave::getLayerAt(int i)
{
    CV_Assert(i < (int) layers.size());
    return layers[i];
}

} // anonymous namespace

namespace cv
{
namespace xfeatures2d
{

/*
 *  HarrisLaplaceFeatureDetector_Impl
 */
class HarrisLaplaceFeatureDetector_Impl CV_FINAL : public HarrisLaplaceFeatureDetector
{
public:
    HarrisLaplaceFeatureDetector_Impl(
        int numOctaves=6,
        float corn_thresh=0.01,
        float DOG_thresh=0.01,
        int maxCorners=5000,
        int num_layers=4
    );
    void read( const FileNode& fn ) CV_OVERRIDE;
    void write( FileStorage& fs ) const CV_OVERRIDE;

    void setNumOctaves(int numOctaves_) CV_OVERRIDE {numOctaves = numOctaves_;}
    int getNumOctaves() const CV_OVERRIDE {return numOctaves;}

    void setCornThresh(float corn_thresh_) CV_OVERRIDE {corn_thresh = corn_thresh_;}
    float getCornThresh() const CV_OVERRIDE {return corn_thresh;}

    void setDOGThresh(float DOG_thresh_) CV_OVERRIDE {DOG_thresh = DOG_thresh_;}
    float getDOGThresh() const CV_OVERRIDE {return DOG_thresh;}

    void setMaxCorners(int maxCorners_) CV_OVERRIDE {maxCorners = maxCorners_;}
    int getMaxCorners() const CV_OVERRIDE {return maxCorners;}

    void setNumLayers(int num_layers_) CV_OVERRIDE {num_layers = num_layers_; CV_Assert(num_layers == 2 || num_layers==4);}
    int getNumLayers() const CV_OVERRIDE {return num_layers;}

protected:
    void detect( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) CV_OVERRIDE;

    int numOctaves;
    float corn_thresh;
    float DOG_thresh;
    int maxCorners;
    int num_layers;
};

Ptr<HarrisLaplaceFeatureDetector> HarrisLaplaceFeatureDetector::create(
    int numOctaves,
    float corn_thresh,
    float DOG_thresh,
    int maxCorners,
    int num_layers)
{
    return makePtr<HarrisLaplaceFeatureDetector_Impl>(numOctaves, corn_thresh, DOG_thresh, maxCorners, num_layers);
}

CV_WRAP String HarrisLaplaceFeatureDetector::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".HARRIS-LAPLACE");
}

HarrisLaplaceFeatureDetector_Impl::HarrisLaplaceFeatureDetector_Impl(
    int _numOctaves,
    float _corn_thresh,
    float _DOG_thresh,
    int _maxCorners,
    int _num_layers
) :
    numOctaves(_numOctaves),
    corn_thresh(_corn_thresh),
    DOG_thresh(_DOG_thresh),
    maxCorners(_maxCorners),
    num_layers(_num_layers)
{
    CV_Assert(num_layers == 2 || num_layers==4);
}

void HarrisLaplaceFeatureDetector_Impl::read (const FileNode& fn)
{
    // if node is empty, keep previous value
    if (!fn["numOctaves"].empty())
        fn["numOctaves"] >> numOctaves;
    if (!fn["corn_thresh"].empty())
        fn["corn_thresh"] >> corn_thresh;
    if (!fn["corn_thresh"].empty())
        fn["DOG_thresh"] >> DOG_thresh;
    if (!fn["maxCorners"].empty())
        fn["maxCorners"] >> maxCorners;
    if (!fn["num_layers"].empty())
        fn["num_layers"] >> num_layers;
}

void HarrisLaplaceFeatureDetector_Impl::write (FileStorage& fs) const
{
    if ( fs.isOpened() )
    {
        fs << "name" << getDefaultName();
        fs << "numOctaves" << numOctaves;
        fs << "corn_thresh" << corn_thresh;
        fs << "DOG_thresh" << DOG_thresh;
        fs << "maxCorners" << maxCorners;
        fs << "num_layers" << num_layers;
    }
}

/*
 * Detect method
 * The method detect Harris corners on scale space as described in
 * "K. Mikolajczyk and C. Schmid.
 * Scale & affine invariant interest point detectors.
 * International Journal of Computer Vision, 2004"
 */
void HarrisLaplaceFeatureDetector_Impl::detect(InputArray img, std::vector<KeyPoint>& keypoints, InputArray msk )
{
    Mat image = img.getMat();
    if( image.empty() )
    {
        keypoints.clear();
        return;
    }
    Mat mask = msk.getMat();
    if( !mask.empty() )
    {
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(mask.size == image.size);
    }
    Mat_<float> dx2, dy2, dxy;
    Mat Lx, Ly;
    float si, sd;
    int gsize;
    Mat fimage;
    image.convertTo(fimage, CV_32F, 1.f/255);
    /*Build gaussian pyramid*/
    Pyramid pyr(fimage, numOctaves, num_layers, 1, -1, true);
    keypoints = std::vector<KeyPoint> (0);

    /*Find Harris corners on each layer*/
    //Use pyr.params.octavesN instead of numOctaves. See issue #1513
    for (int octave = 0; octave <= pyr.params.octavesN; octave++)
    {
        for (int layer = 1; layer <= num_layers; layer++)
        {
            if (octave == 0)
                layer = num_layers;

            Mat Lxm2smooth, Lxmysmooth, Lym2smooth;

            si = powf(2.f, layer / (float) num_layers);
            sd = si * 0.7f;

            Mat curr_layer;
            if (num_layers == 4)
            {
                if (layer == 1)
                {
                    Mat tmp = pyr.getLayer(octave - 1, num_layers - 1);
                    resize(tmp, curr_layer, Size(0, 0), 0.5, 0.5, INTER_AREA);

                } else
                    curr_layer = pyr.getLayer(octave, layer - 2);
            } else /*if num_layer==2*/
            {

                curr_layer = pyr.getLayer(octave, layer - 1);
            }

            /*Calculates second moment matrix*/

            /*Derivatives*/
            Sobel(curr_layer, Lx, CV_32F, 1, 0, 1);
            Sobel(curr_layer, Ly, CV_32F, 0, 1, 1);

            /*Normalization*/
            Lx = Lx * sd;
            Ly = Ly * sd;

            Mat Lxm2 = Lx.mul(Lx);
            Mat Lym2 = Ly.mul(Ly);
            Mat Lxmy = Lx.mul(Ly);

            gsize = int(ceil(si * 3)) * 2 + 1;

            /*Convolution*/
            GaussianBlur(Lxm2, Lxm2smooth, Size(gsize, gsize), si, si, BORDER_REPLICATE);
            GaussianBlur(Lym2, Lym2smooth, Size(gsize, gsize), si, si, BORDER_REPLICATE);
            GaussianBlur(Lxmy, Lxmysmooth, Size(gsize, gsize), si, si, BORDER_REPLICATE);

            Mat cornern_mat(curr_layer.size(), CV_32F);

            /*Calculates cornerness in each pixel of the image*/
            for (int row = 0; row < curr_layer.rows; row++)
            {
                for (int col = 0; col < curr_layer.cols; col++)
                {
                    float dx2f = Lxm2smooth.at<float> (row, col);
                    float dy2f = Lym2smooth.at<float> (row, col);
                    float dxyf = Lxmysmooth.at<float> (row, col);
                    float det = dx2f * dy2f - dxyf * dxyf;
                    float tr = dx2f + dy2f;
                    float cornerness = det - (0.04f * tr * tr);
                    cornern_mat.at<float> (row, col) = cornerness;
                }
            }

            double maxVal = 0;
            Mat corn_dilate;

            /*Find max cornerness value and rejects all corners that are lower than a threshold*/
            minMaxLoc(cornern_mat, 0, &maxVal, 0, 0);
            threshold(cornern_mat, cornern_mat, maxVal * corn_thresh, 0, THRESH_TOZERO);
            dilate(cornern_mat, corn_dilate, Mat());

            Size imgsize = curr_layer.size();

            /*Verify for each of the initial points whether the DoG attains a maximum at the scale of the point*/
            Mat prevDOG, curDOG, succDOG;
            prevDOG = pyr.getDOGLayer(octave, layer - 1);
            curDOG = pyr.getDOGLayer(octave, layer);
            succDOG = pyr.getDOGLayer(octave, layer + 1);

            for (int y = 1; y < imgsize.height - 1; y++)
            {
                for (int x = 1; x < imgsize.width - 1; x++)
                {
                    float val = cornern_mat.at<float> (y, x);
                    if (val != 0 && val == corn_dilate.at<float> (y, x))
                    {

                        float curVal = curDOG.at<float> (y, x);
                        float prevVal =  prevDOG.at<float> (y, x);
                        float succVal = succDOG.at<float> (y, x);

                        KeyPoint kp(
                                Point2f(x * powf(2.0f, (float) octave - 1) + powf(2.0f, (float) octave - 1) / 2,
                                        y * powf(2.0f, (float) octave - 1) + powf(2.0f, (float) octave - 1) / 2),
                                3 * powf(2.0f, (float) octave - 1) * si * 2, 0, val, octave);

                        if(!mask.empty() && mask.at<unsigned char>(int(kp.pt.y), int(kp.pt.x)) == 0)
                        {
                            // ignore keypoints where mask is zero
                            continue;
                        }

                        /*Check whether keypoint size is inside the image*/
                        float start_kp_x = kp.pt.x - kp.size / 2;
                        float start_kp_y = kp.pt.y - kp.size / 2;
                        float end_kp_x = start_kp_x + kp.size;
                        float end_kp_y = start_kp_y + kp.size;

                        if (curVal > prevVal && curVal > succVal && curVal >= DOG_thresh
                                && start_kp_x > 0 && start_kp_y > 0 && end_kp_x < image.cols
                                && end_kp_y < image.rows)
                            keypoints.push_back(kp);

                    }
                }
            }

        }

    }

    /*Sort keypoints in decreasing cornerness order*/
    sort(keypoints.begin(), keypoints.end(), sort_func);
    for (size_t i = 1; i < keypoints.size(); i++)
    {
        float max_diff = powf(2, keypoints[i].octave + 1.f / 2);

        if (keypoints[i].response == keypoints[i - 1].response && norm(
                keypoints[i].pt - keypoints[i - 1].pt) <= max_diff)
        {

            float x = (keypoints[i].pt.x + keypoints[i - 1].pt.x) / 2;
            float y = (keypoints[i].pt.y + keypoints[i - 1].pt.y) / 2;

            keypoints[i].pt = Point2f(x, y);
            --i;
            keypoints.erase(keypoints.begin() + i);

        }
    }

    /*Select strongest keypoints*/
    if (maxCorners > 0 && maxCorners < (int) keypoints.size())
        keypoints.resize(maxCorners);
}


}
}
