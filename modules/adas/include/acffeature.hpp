class ACFFeature
{
public:
    /* Initialize feature with zero row and col */
    ACFFeature();

    /* Initialize feature with given row and col */
    ACFFeature(int row, int col);

private:
    /* Feature row */
    int row_;
    /* Feature col */
    int col_;
};

/* Save ACFFeature to FileStorage */
cv::FileStorage& operator<< (cv::FileStorage& out, const ACFFeature& feature);
/* Load ACFFeature from FileStorage */
cv::FileStorage& operator>> (cv::FileStorage& in, ACFFeature& feature);

/* Compute channel pyramid for acf features

    image — image, for which pyramid should be computed

    params — pyramid computing parameters

Returns computed channels in vectors N x CH,
N — number of scales (outer vector),
CH — number of channels (inner vectors)
*/
std::vector<std::vector<cv::Mat_<int>>>
computeChannels(const cv::Mat& image, const ScaleParams& params);

class ACFFeatureEvaluator
{
public:
    /* Construct evaluator, set features to evaluate */
    ACFFeatureEvaluator(const std::vector<ACFFeature>& features);

    /* Set channels for feature evaluation */
    void setChannels(const std::vector<cv::Mat_<int>>& channels);

    /* Set window position */
    void setPosition(Size position);

    /* Evaluate feature with given index for current channels
        and window position */
    int evaluate(size_t feature_ind) const;

    /* Evaluate all features for current channels and window position

    Returns matrix-column of features
    */
    cv::Mat_<int> evaluateAll() const;

private:
    /* Features to evaluate */
    std::vector<ACFFeature> features_;
    /* Channels for feature evaluation */
    std::vector<cv::Mat_<int>> channels
    /* Channels window position */
    Size position_;
};

/* Generate acf features

    window_size — size of window in which features should be evaluated

    count — number of features to generate.
    Max number of features is min(count, # possible distinct features)

    seed — random number generator initializer

Returns vector of distinct acf features
*/
std::vector<ACFFeature>
generateFeatures(Size window_size, size_t count = UINT_MAX, int seed = 0);