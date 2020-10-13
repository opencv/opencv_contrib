// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "opencv2/tbmr.hpp"

namespace cv {
    namespace tbmr {
        class TBMR_Impl CV_FINAL : public TBMR
        {
        public:
            struct Params
            {
                Params(int _min_area = 60, double _max_area_relative = 0.01)
                {
                    CV_Assert(_min_area >= 0);
                    CV_Assert(_max_area_relative >= std::numeric_limits<double>::epsilon());

                    minArea = _min_area;
                    maxAreaRelative = _max_area_relative;
                }


                uint minArea;
                double maxAreaRelative;
            };

            explicit TBMR_Impl(const Params& _params) : params(_params) {}

            virtual ~TBMR_Impl() CV_OVERRIDE {}

            void setMinArea(int minArea) CV_OVERRIDE
            {
                params.minArea = minArea;
            }
            int getMinArea() const CV_OVERRIDE
            {
                return params.minArea;
            }

            void setMaxAreaRelative(double maxAreaRelative) CV_OVERRIDE
            {
                params.maxAreaRelative = maxAreaRelative;
            }
            double getMaxAreaRelative() const CV_OVERRIDE
            {
                return params.maxAreaRelative;
            }

            void detect(InputArray image, CV_OUT std::vector<KeyPoint>& keypoints, InputArray mask = noArray()) CV_OVERRIDE;

            // radix sort images -> indexes
            template<bool sort_order_up = true>
            cv::Mat sort_indexes(const cv::Mat& input)
            {
                const size_t bucket_count = 1 << 8;
                const uint mask = bucket_count - 1;
                uint N = input.cols * input.rows;
                cv::Mat indexes_sorted(input.rows, input.cols, CV_32S); // unsigned


                const uint8_t* input_ptr = input.ptr<const uint8_t>();
                uint* lutSorted = indexes_sorted.ptr<uint>();

                uint bucket_offsets[bucket_count + 1];
                memset(&bucket_offsets, 0, sizeof(int) * (bucket_count + 1));

                // count occurences
                for (uint i = 0; i < N; ++i)
                {
                    uint8_t key = input_ptr[i];
                    uint bucket = key & mask;
                    bucket_offsets[bucket + 1]++;
                }

                // make cumulative
                for (uint i = 0; i < bucket_count - 1; ++i)
                    bucket_offsets[i + 1] += bucket_offsets[i];


                // actual step
                for (uint i = 0; i < N; ++i)
                {
                    uint8_t key = input_ptr[i];
                    uint bucket = key & mask;

                    uint pos = bucket_offsets[bucket]++;

                    if (sort_order_up)
                    {
                        lutSorted[pos] = i;
                    }
                    else
                    {
                        lutSorted[N - 1 - pos] = i; // reverse
                    }
                }

                return indexes_sorted;
            }

            inline uint zfindroot(uint* parent, uint p)
            {
                if (parent[p] == p)
                    return p;
                else
                    return parent[p] = zfindroot(parent, parent[p]);
            }


            template<bool order_up = true>
            void calc_min_max_tree(cv::Mat ima, cv::Mat& parent, cv::Mat& S, std::array<uint, 6>* imaAttribute)
            {
                int rs = ima.rows;
                int cs = ima.cols;
                uint imSize = (uint)rs * cs;

                std::array<int, 4> offsets = { -ima.cols, -1, 1, ima.cols }; // {-1,0}, {0,-1}, {0,1}, {1,0} yx
                std::array<cv::Vec2i, 4> offsetsv
                    = { cv::Vec2i(0, -1), cv::Vec2i(-1, 0), cv::Vec2i(1, 0), cv::Vec2i(0, 1) }; //  xy


                uint* zpar = (uint*)malloc(imSize * sizeof(uint));
                uint* root = (uint*)malloc(imSize * sizeof(uint));
                uint* rank = (uint*)calloc(imSize, sizeof(uint));
                parent = cv::Mat(rs, cs, CV_32S); // unsigned
                bool* dejaVu = (bool*)calloc(imSize, sizeof(bool));
                S = sort_indexes<order_up>(ima);

                const uint8_t* ima_ptr = ima.ptr<const uint8_t>();
                const uint* S_ptr = S.ptr<const uint>();
                uint* parent_ptr = parent.ptr<uint>();


                for (int i = imSize - 1; i >= 0; --i)
                {
                    uint p = S_ptr[i];

                    cv::Vec2i idx_p(p % cs, p / cs);
                    // make set
                    {
                        parent_ptr[p] = p;
                        zpar[p] = p;
                        root[p] = p;
                        dejaVu[p] = true;
                        imaAttribute[p][0] = 1;                   // area
                        imaAttribute[p][1] = idx_p[0];            // sum_x
                        imaAttribute[p][2] = idx_p[1];            // sum_y
                        imaAttribute[p][3] = idx_p[0] * idx_p[1]; // sum_xy
                        imaAttribute[p][4] = idx_p[0] * idx_p[0]; // sum_xx
                        imaAttribute[p][5] = idx_p[1] * idx_p[1]; // sum_yy
                    }

                    uint x = p; // zpar of p
                    for (unsigned k = 0; k < offsets.size(); ++k)
                    {
                        uint q = p + offsets[k];

                        cv::Vec2i q_idx = idx_p + offsetsv[k];
                        bool inBorder
                            = q_idx[0] >= 0 && q_idx[0] < ima.cols && q_idx[1] >= 0 && q_idx[1] < ima.rows; // filter out border cases

                        if (inBorder && dejaVu[q]) // remove first check obsolete
                        {
                            uint r = zfindroot(zpar, q);
                            if (r != x) // make union
                            {
                                parent_ptr[root[r]] = p;
                                // accumulate information
                                imaAttribute[p][0] += imaAttribute[root[r]][0]; // area
                                imaAttribute[p][1] += imaAttribute[root[r]][1]; // sum_x
                                imaAttribute[p][2] += imaAttribute[root[r]][2]; // sum_y
                                imaAttribute[p][3] += imaAttribute[root[r]][3]; // sum_xy
                                imaAttribute[p][4] += imaAttribute[root[r]][4]; // sum_xx
                                imaAttribute[p][5] += imaAttribute[root[r]][5]; // sum_yy

                                if (rank[x] < rank[r])
                                {
                                    // we merge p to r
                                    zpar[x] = r;
                                    root[r] = p;
                                    x = r;
                                }
                                else if (rank[r] < rank[p])
                                {
                                    // merge r to p
                                    zpar[r] = p;
                                }
                                else
                                {
                                    // same height
                                    zpar[r] = p;
                                    rank[p] += 1;
                                }
                            }
                        }
                    }
                }

                free(zpar);
                free(root);
                free(rank);
                free(dejaVu);
            }

            template<bool order_up = true>
            void calculateTBMRs(const cv::Mat& image, std::vector<KeyPoint>& tbmrs, const cv::Mat& mask)
            {

                uint imSize = image.cols * image.rows;
                uint maxArea = static_cast<uint>(params.maxAreaRelative * imSize);


                cv::Mat parentMat, SMat;

                // calculate moments during tree construction: compound type of: (area, x, y, xy, xx, yy)
                std::array<uint, 6>* imaAttribute = (std::array<uint, 6>*)malloc(imSize * sizeof(uint) * 6);

                calc_min_max_tree<order_up>(image, parentMat, SMat, imaAttribute);

                const uint8_t* ima_ptr = image.ptr<const uint8_t>();
                const uint* S = SMat.ptr<const uint>();
                uint* parent = parentMat.ptr<uint>();

                // canonization
                for (uint i = 0; i < imSize; ++i)
                {
                    uint p = S[i];
                    uint q = parent[p];
                    if (ima_ptr[parent[q]] == ima_ptr[q])
                        parent[p] = parent[q];
                }

                // TBMRs extraction
                //------------------------------------------------------------------------
                // small variant of the given algorithm in the paper. For each critical node having more than one child, we
                // check if the largest region containing this node without any change of topology is above its parent, if not,
                // discard this critical node.
                //
                // note also that we do not select the critical nodes themselves as final TBMRs
                //--------------------------------------------------------------------------


                uint* numSons = (uint*)calloc(imSize, sizeof(uint));
                uint vecNodesSize = imaAttribute[S[0]][0];                     // area
                uint* vecNodes = (uint*)calloc(vecNodesSize, sizeof(uint)); // area
                uint numNodes = 0;

                // leaf to root propagation to select the canonized nodes
                for (int i = imSize - 1; i >= 0; --i)
                {
                    uint p = S[i];
                    if (parent[p] == p || ima_ptr[p] != ima_ptr[parent[p]])
                    {
                        vecNodes[numNodes++] = p;
                        if (imaAttribute[p][0] >= params.minArea) // area
                            numSons[parent[p]]++;
                    }
                }

                bool* isSeen = (bool*)calloc(imSize, sizeof(bool));

                // parent of critical leaf node
                bool* isParentofLeaf = (bool*)calloc(imSize, sizeof(bool));

                for (uint i = 0; i < vecNodesSize; i++)
                {
                    uint p = vecNodes[i];
                    if (numSons[p] == 0 && numSons[parent[p]] == 1)
                        isParentofLeaf[parent[p]] = true;
                }

                uint numTbmrs = 0;
                uint* vecTbmrs = (uint*)malloc(numNodes * sizeof(uint));
                for (uint i = 0; i < vecNodesSize; i++)
                {
                    std::size_t p = vecNodes[i];
                    if (numSons[p] == 1 && !isSeen[p] && imaAttribute[p][0] <= maxArea)
                    {
                        uint num_ancestors = 0;
                        std::size_t pt = p;
                        std::size_t po = pt;
                        while (numSons[pt] == 1 && imaAttribute[pt][0] <= maxArea)
                        {
                            isSeen[pt] = true;
                            num_ancestors++;
                            po = pt;
                            pt = parent[pt];
                        }
                        if (!isParentofLeaf[p] || num_ancestors > 1)
                        {
                            vecTbmrs[numTbmrs++] = po;
                        }
                    }
                }
                // end of TBMRs extraction
                //------------------------------------------------------------------------

                // compute best fitting ellipses
                //------------------------------------------------------------------------
                for (int i = 0; i < numTbmrs; i++)
                {
                    uint p = vecTbmrs[i];
                    double area = static_cast<double>(imaAttribute[p][0]);
                    double sum_x = static_cast<double>(imaAttribute[p][1]);
                    double sum_y = static_cast<double>(imaAttribute[p][2]);
                    double sum_xy = static_cast<double>(imaAttribute[p][3]);
                    double sum_xx = static_cast<double>(imaAttribute[p][4]);
                    double sum_yy = static_cast<double>(imaAttribute[p][5]);

                    // Barycenter:
                    double x = sum_x / area;
                    double y = sum_y / area;
                    // Second order moments
                    //double X2 = sum_xx / area - x * x;
                    //double Y2 = sum_yy / area - y * y;
                    //double XY = sum_xy / area - x * y;

                    double i20 = sum_xx - area * x * x;
                    double i02 = sum_yy - area * y * y;
                    double i11 = sum_xy - area * x * y;
                    double n = i20 * i02 - i11 * i11;
                    if (n != 0)
                    {
                        double a = (i02 / n) * (area - 1) / 4;
                        double b = (-i11 / n) * (area - 1) / 4;
                        double c = (i20 / n) * (area - 1) / 4;

                        // filter out some non meaningful ellipses
                        double a1 = a;
                        double b1 = b;
                        double c1 = c;
                        unsigned ai = 0;
                        unsigned bi = 0;
                        unsigned ci = 0;
                        if (a > 0)
                        {
                            ai = 100000 * a;
                            if (a < 0.00005)
                                a1 = 0;
                            else if (a < 0.0001)
                            {
                                a1 = 0.0001;
                            }
                            else
                            {
                                ai = 10000 * a;
                                a1 = (double)ai / 10000;
                            }
                        }
                        else
                        {
                            if (a > -0.00005)
                                a1 = 0;
                            else if (a > -0.0001)
                                a1 = -0.0001;
                            else
                            {
                                ai = 10000 * (-a);
                                a1 = -(double)ai / 10000;
                            }
                        }

                        if (b > 0)
                        {
                            bi = 100000 * b;
                            if (b < 0.00005)
                                b1 = 0;
                            else if (b < 0.0001)
                            {
                                b1 = 0.0001;
                            }
                            else
                            {
                                bi = 10000 * b;
                                b1 = (double)bi / 10000;
                            }
                        }
                        else
                        {
                            if (b > -0.00005)
                                b1 = 0;
                            else if (b > -0.0001)
                                b1 = -0.0001;
                            else
                            {
                                bi = 10000 * (-b);
                                b1 = -(double)bi / 10000;
                            }
                        }

                        if (c > 0)
                        {
                            ci = 100000 * c;
                            if (c < 0.00005)
                                c1 = 0;
                            else if (c < 0.0001)
                            {
                                c1 = 0.0001;
                            }
                            else
                            {
                                ci = 10000 * c;
                                c1 = (double)ci / 10000;
                            }
                        }
                        else
                        {
                            if (c > -0.00005)
                                c1 = 0;
                            else if (c > -0.0001)
                                c1 = -0.0001;
                            else
                            {
                                ci = 10000 * (-c);
                                c1 = -(double)ci / 10000;
                            }
                        }
                        double v = (a1 + c1 - std::sqrt(a1 * a1 + c1 * c1 + 4 * b1 * b1 - 2 * a1 * c1)) / 2;

                        double l1 = 1. / std::sqrt((a + c + std::sqrt(a * a + c * c + 4 * b * b - 2 * a * c)) / 2);
                        double l2 = 1. / std::sqrt((a + c - std::sqrt(a * a + c * c + 4 * b * b - 2 * a * c)) / 2);
                        double l = std::min(l1, l2);

                        if (l >= 1.5 && v != 0 && (mask.empty() || mask.at<uchar>(cvRound(y), cvRound(x)) != 0))
                        {
                            float diam = 2 * a; // Major axis
                            //    float angle = std::atan((a / b) * (y / x));
                            tbmrs.push_back(cv::KeyPoint(cv::Point2f(x, y), diam));
                        }
                    }
                }

                free(imaAttribute);
                free(numSons);
                free(vecNodes);
                free(isSeen);
                free(isParentofLeaf);
                free(vecTbmrs);
                //---------------------------------------------
            }

            Mat tempsrc;

            Params params;
        };

        void TBMR_Impl::detect(InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask)
        {
            Mat mask = _mask.getMat();
            Mat src = _image.getMat();

            keypoints.clear();

            CV_Assert(!src.empty());
            CV_Assert(src.type() == CV_8UC1);

            if (!src.isContinuous())
            {
                src.copyTo(tempsrc);
                src = tempsrc;
            }

            // append max-tree tbmrs
            calculateTBMRs<true>(src, keypoints, mask);
            // append min-tree tbmrs
            calculateTBMRs<false>(src, keypoints, mask);
        }

        CV_WRAP Ptr<TBMR> TBMR::create(int _min_area, double _max_area_relative)
        {
            return cv::makePtr<TBMR_Impl>(TBMR_Impl::Params(_min_area, _max_area_relative));
        }

        String TBMR::getDefaultName() const
        {
            return (Feature2D::getDefaultName() + ".TBMR");
        }
    }
}
