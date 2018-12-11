#include "perf_precomp.hpp"
namespace opencv_test {
    namespace {
        typedef tuple<std::string, std::string, bool> ST_SR_IM_Sparse_t;
        typedef TestBaseWithParam<ST_SR_IM_Sparse_t> ST_SR_IM_Sparse;
        PERF_TEST_P(ST_SR_IM_Sparse, OpticalFlow_SparseRLOF,
            testing::Combine(
                testing::Values<std::string>("ST_BILINEAR", "ST_STANDART"),
                testing::Values<std::string>("SR_CROSS", "SR_FIXED"),
                testing::Values(true, false))
        )
        {
            string frame1_path = TS::ptr()->get_data_path() + "/cv/optflow/RubberWhale1.png";
            string frame2_path = TS::ptr()->get_data_path() + "/cv/optflow/RubberWhale2.png";
            frame1_path.erase(std::remove_if(frame1_path.begin(), frame1_path.end(), isspace), frame1_path.end());
            frame2_path.erase(std::remove_if(frame2_path.begin(), frame2_path.end(), isspace), frame2_path.end());
            Mat frame1 = imread(frame1_path);
            Mat frame2 = imread(frame2_path);
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

            Ptr<RLOFOpticalFlowParameter> param;
            if (get<0>(GetParam()) == "ST_BILINEAR")
                param->solverType = ST_BILINEAR;
            if (get<0>(GetParam()) == "ST_STANDART")
                param->solverType = ST_STANDART;
            if (get<1>(GetParam()) == "SR_CROSS")
                param->supportRegionType = SR_CROSS;
            if (get<1>(GetParam()) == "SR_FIXED")
                param->supportRegionType = SR_FIXED;
            param->useIlluminationModel = get<2>(GetParam());

            TEST_CYCLE_N(1)
            {
                calcOpticalFlowSparseRLOF(frame1, frame2, prevPts, currPts, status, err, param, 1.f);
            }

            SANITY_CHECK_NOTHING();
        }

        typedef tuple<std::string, int> INTERP_GRID_Dense_t;
        typedef TestBaseWithParam<INTERP_GRID_Dense_t> INTERP_GRID_Dense;
        PERF_TEST_P(INTERP_GRID_Dense, OpticalFlow_DenseRLOF,
            testing::Combine(
                testing::Values<std::string>("INTERP_EPIC", "INTERP_GEO"),
                testing::Values<int>(4,10,50))
        )
        {
            Mat flow;
            string frame1_path = TS::ptr()->get_data_path() + "/cv/optflow/RubberWhale1.png";
            string frame2_path = TS::ptr()->get_data_path() + "/cv/optflow/RubberWhale2.png";
            // removing space may be an issue on windows machines
            frame1_path.erase(std::remove_if(frame1_path.begin(), frame1_path.end(), isspace), frame1_path.end());
            frame2_path.erase(std::remove_if(frame2_path.begin(), frame2_path.end(), isspace), frame2_path.end());
            Mat frame1 = imread(frame1_path);
            Mat frame2 = imread(frame2_path);
            ASSERT_FALSE(frame1.empty());
            ASSERT_FALSE(frame2.empty());
            Ptr<RLOFOpticalFlowParameter> param;
            Ptr< DenseRLOFOpticalFlow> algo = DenseRLOFOpticalFlow::create();
            InterpolationType interp_type;
            if (get<0>(GetParam()) == "INTERP_EPIC")
                interp_type = INTERP_EPIC;
            if (get<0>(GetParam()) == "INTERP_GEO")
                interp_type = INTERP_GEO;
            TEST_CYCLE_N(5)
            {
                calcOpticalFlowDenseRLOF(frame1, frame2,flow, param, 1.0f, Size(get<1>(GetParam()), get<1>(GetParam())), interp_type);
            }
            SANITY_CHECK_NOTHING();
        }
    }
} // namespace
