class Stump
{
public:
    /* Train stump for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

    Returns chosen feature index. Feature enumeration starts from 0
    */
    int train(const cv::Mat_<int>& data, const cv::Mat_<int>& labels);

    /* Predict object class given

        value — feature value. Feature must be the same as chose during training
        stump

    Returns object class from {-1, +1}
    */
    int predict(int value);

private:
    /* Stump decision threshold */
    int threshold_;
    /* Stump polarity, can be from {-1, +1} */
    int polarity_;
    /* Stump decision rule:
        h(value) = polarity * sign(value - threshold)
    */
};

/* Save Stump to FileStorage */
cv::FileStorage& operator<< (cv::FileStorage& out, const Stump& classifier);
/* Load Stump from FileStorage */
cv::FileStorage& operator>> (cv::FileStorage& in, Stump& classifier);

class WaldBoost
{
public:
    /* Initialize WaldBoost cascade with default of specified parameters */
    WaldBoost(const WaldBoostParams& = WaldBoostParams());

    /* Train WaldBoost cascade for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

    Returns feature indices chosen for cascade.
    Feature enumeration starts from 0
    */
    std::vector<int> train(const cv::Mat_<int>& data,
                           const cv::Mat_<int>& labels);

    /* Predict object class given object that can compute object features

       feature_evaluator — object that can compute features by demand

    Returns confidence_value — measure of confidense that object
    is from class +1
    */
    float predict(const ACFFeatureEvaluator& feature_evaluator);

private:
    /* Parameters for cascade training */
    WaldBoostParams params_;
    /* Stumps in cascade */
    std::vector<Stump> stumps_;
    /* Weight for stumps in cascade linear combination */
    std::vector<float> stump_weights_;
    /* Rejection thresholds for linear combination at every stump evaluation */
    std::vector<float> thresholds_;
};

/* Save WaldBoost to FileStorage */
cv::FileStorage& operator<< (cv::FileStorage& out, const WaldBoost& classifier);
/* Load WaldBoost from FileStorage */
cv::FileStorage& operator>> (cv::FileStorage& in, WaldBoost& classifier);