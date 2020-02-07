#include "perf_precomp.hpp"
namespace opencv_test { namespace {

typedef tuple<std::string, std::string, bool> ST_SR_IM_Sparse_t;
typedef TestBaseWithParam<ST_SR_IM_Sparse_t> ST_SR_IM_Sparse;
PERF_TEST_P(ST_SR_IM_Sparse, OpticalFlow_SparseRLOF,
    testing::Combine(
        testing::Values<std::string>("ST_BILINEAR", "ST_STANDART"),
        testing::Values<std::string>("SR_CROSS", "SR_FIXED"),
        testing::Values(true, false))
)
{
    Mat frame1 = imread(getDataPath("cv/optflow/RubberWhale1.png"));
    Mat frame2 = imread(getDataPath("cv/optflow/RubberWhale2.png"));
    ASSERT_FALSE(frame1.empty());
    ASSERT_FALSE(frame2.empty());
    vector<Point2f> prevPts, currPts;
    for (int r = 0; r < frame1.rows; r += 10)
    {
        for (int c = 0; c < frame1.cols; c += 10)
        {
            prevPts.push_back(Point2f(static_cast<float>(c), static_cast<float>(r)));
        }
    }
    vector<uchar> status(prevPts.size());
    vector<float> err(prevPts.size());

    Ptr<RLOFOpticalFlowParameter> param = Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter);
    if (get<0>(GetParam()) == "ST_BILINEAR")
        param->solverType = ST_BILINEAR;
    if (get<0>(GetParam()) == "ST_STANDART")
        param->solverType = ST_STANDART;
    if (get<1>(GetParam()) == "SR_CROSS")
        param->supportRegionType = SR_CROSS;
    if (get<1>(GetParam()) == "SR_FIXED")
        param->supportRegionType = SR_FIXED;
    param->useIlluminationModel = get<2>(GetParam());

    PERF_SAMPLE_BEGIN()
        calcOpticalFlowSparseRLOF(frame1, frame2, prevPts, currPts, status, err, param, 1.f);
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

typedef tuple<std::string, int> INTERP_GRID_Dense_t;
typedef TestBaseWithParam<INTERP_GRID_Dense_t> INTERP_GRID_Dense;
PERF_TEST_P(INTERP_GRID_Dense, OpticalFlow_DenseRLOF,
    testing::Combine(
        testing::Values<std::string>("INTERP_EPIC", "INTERP_GEO", "INTERP_RIC"),
        testing::Values<int>(4,10))
)
{
    Mat flow;
    Mat frame1 = imread(getDataPath("cv/optflow/RubberWhale1.png"));
    Mat frame2 = imread(getDataPath("cv/optflow/RubberWhale1.png"));
    ASSERT_FALSE(frame1.empty());
    ASSERT_FALSE(frame2.empty());
    Ptr<RLOFOpticalFlowParameter> param = Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter);;
    Ptr< DenseRLOFOpticalFlow> algo = DenseRLOFOpticalFlow::create();
    InterpolationType interp_type = INTERP_EPIC;
    if (get<0>(GetParam()) == "INTERP_EPIC")
        interp_type = INTERP_EPIC;
    if (get<0>(GetParam()) == "INTERP_GEO")
        interp_type = INTERP_GEO;
    if (get<0>(GetParam()) == "INTERP_RIC")
        interp_type = INTERP_RIC;
    PERF_SAMPLE_BEGIN()
        calcOpticalFlowDenseRLOF(frame1, frame2,flow, param, 1.0f, Size(get<1>(GetParam()), get<1>(GetParam())), interp_type);
    PERF_SAMPLE_END()
    SANITY_CHECK_NOTHING();
}

}} // namespace
