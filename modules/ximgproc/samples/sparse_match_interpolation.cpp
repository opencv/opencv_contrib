#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/optflow.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

#define EPS 1e-43f

const String keys =
    "{help h usage ? |      | print this message                }"
    "{src_video      |None  | path to the folder with src frames (either this of pass two frames) }"
    "{prev_frame     |None  | path to the previous frame }"
    "{cur_frame      |None  | path to the current frame }"
    "{GT_path        |None  | path to the folder with flo files }"
    "{dst_path       |None  | folder to store the dense flow    }"
    "{dst_raw_path   |None  | folder to store the sparse flow   }"
    "{use-grid       |      | disable explicit point-tracking   }"
    ;

void VisualizeFlow(Mat& flow, Mat& dst_vis, double scale_contrast=0.05)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = min(scale_contrast*magnitude,1.0);
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    rgb.convertTo(dst_vis,CV_8UC3,255.0);
}

void VisualizeSparseFlow(Mat& src,vector<Point2f>& src_points,vector<Point2f>& dst_points,vector<unsigned char> status, Mat& dst_vis)
{
    Mat dense_flow(src.rows,src.cols,CV_32FC2);
    dense_flow = Scalar(0.0,0.0);
    for(int i=0;i<src_points.size();i++)
    {
        if(status[i]!=0)
            dense_flow.at<Point2f>(src_points[i].y,src_points[i].x) = Point2f(dst_points[i].x-src_points[i].x,
                                                                              dst_points[i].y-src_points[i].y);
    }
    VisualizeFlow(dense_flow,dst_vis);
    dilate(dst_vis,dst_vis,getStructuringElement(MORPH_RECT,Size(3,3)));
}

void mergePoints(vector<Point2f>& prev_points, vector<uchar>& prev_status, vector<Point2f>& cur_points, int w, int h, vector<Point2f>& dst_points, float eps = 5.0f)
{
    dst_points.clear();
    for(int i=0;i<prev_points.size();i++)
    {
        if(prev_status[i]!=0 && prev_points[i].x>=0 && prev_points[i].x<w && prev_points[i].y>=0 && prev_points[i].y<h)
            dst_points.push_back(prev_points[i]);
    }
    int sz = dst_points.size();
    bool has_close_neighbor;
    for(int i=0;i<cur_points.size();i++)
    {
        has_close_neighbor = false;
        for(int j=0;j<sz;j++)
        {
            if(abs(cur_points[i].x - dst_points[j].x)<eps &&
               abs(cur_points[i].y - dst_points[j].y)<eps )
            {
                has_close_neighbor = true;
                break;
            }
        }
        if(!has_close_neighbor && dst_points.size()<SHRT_MAX)
            dst_points.push_back(cur_points[i]);
    }
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv,keys);
    parser.about("Match Densifier Demo");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String src_video_folder = parser.get<String>("src_video");
    String prev_frame_path  = parser.get<String>("prev_frame");
    String cur_frame_path   = parser.get<String>("cur_frame");

    String dst_path = parser.get<String>("dst_path");
    String dst_raw_path = parser.get<String>("dst_raw_path");
    String GT_path = parser.get<String>("GT_path");
    bool use_grid  = parser.has("use-grid");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    vector<Point2f> points;
    vector<Point2f> cur_points;

    int max_num_matches = SHRT_MAX;
    vector<Point2f> dst_points;
    vector<unsigned char> status;
    vector<float> err;

    Mat prev,           cur;
    Mat prev_grayscale, cur_grayscale, prev_grayscale_CLAHE;

    if(src_video_folder!="None")
    {
        int idx = 1;
        char idx_string[10];
        String frame_file_name;
        Ptr<CLAHE> equalizer = createCLAHE(10.0);

        sprintf(idx_string,"%04d",idx);
        frame_file_name = src_video_folder + "/frame_" + String(idx_string) + ".png";
        cur = imread(frame_file_name,IMREAD_COLOR);

        while(!cur.empty())
        {
            if(!prev.empty())
            {
                if(use_grid)
                {
                    points.clear();
                    for(int i=0;i<prev.rows;i+=8)
                        for(int j=0;j<prev.cols;j+=8)
                            points.push_back(Point2f(j,i));
                    dst_points.clear();
                    status.clear();
                    err.clear();
                    cvtColor(prev,prev_grayscale,COLOR_BGR2GRAY);
                    cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
                    calcOpticalFlowPyrLK(prev_grayscale,cur_grayscale,points,dst_points,status,err,Size(21,21));
                }
                else
                {
                    cur_points.clear();
                    cvtColor(prev,prev_grayscale,COLOR_BGR2GRAY);
                    equalizer->apply(prev_grayscale,prev_grayscale_CLAHE);
                    goodFeaturesToTrack(prev_grayscale_CLAHE,cur_points,max_num_matches,0.01,0.0);

                    if(dst_points.empty())
                        points = cur_points;
                    else
                        mergePoints(dst_points,status,cur_points,prev.cols,prev.rows,points);

                    dst_points.clear();
                    status.clear();
                    err.clear();
                    cvtColor(cur,cur_grayscale,COLOR_BGR2GRAY);
                    calcOpticalFlowPyrLK(prev_grayscale,cur_grayscale,points,dst_points,status,err,Size(21,21));
                }

                if(dst_raw_path!="None")
                {
                    //visualize sparse flow:
                    Mat vis;
                    VisualizeSparseFlow(prev_grayscale,points,dst_points,status,vis);
                    sprintf(idx_string,"%04d",idx-1);
                    imwrite(dst_raw_path + "/frame_" + String(idx_string) + ".png",vis);
                }

                if(dst_path!="None")
                {
                    Mat dense_flow;
                    vector<SparseMatch> matches;
                    for(int i=0;i<points.size();i++)
                    {
                        if(status[i]!=0)
                            matches.push_back(SparseMatch(points[i],dst_points[i]));
                    }

                    Ptr<EdgeAwareInterpolator> gd = createEdgeAwareInterpolator(false);
                    double filtering_time = (double)getTickCount();
                    gd->interpolate(prev,cur,matches,dense_flow);
                    filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
                    cout.precision(2);
                    cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
                    cout<<endl;

                    Mat vis;
                    VisualizeFlow(dense_flow,vis);
                    sprintf(idx_string,"%04d",idx-1);
                    imwrite(dst_path + "/frame_" + String(idx_string) + ".png",vis);
                }

                if(GT_path!="None" && dst_path!="None")
                {
                    sprintf(idx_string,"%04d",idx-1);
                    Mat GT_flow = optflow::readOpticalFlow(GT_path + "/frame_" + String(idx_string) + ".flo");
                    Mat vis;
                    VisualizeFlow(GT_flow,vis);
                    imwrite(dst_path + "/frame_" + String(idx_string) + "_GT.png",vis);
                }
            }

            cur.copyTo(prev);
            idx++;
            sprintf(idx_string,"%04d",idx);
            frame_file_name = src_video_folder + "/frame_" + String(idx_string) + ".png";
            cur = imread(frame_file_name,IMREAD_COLOR);
            cout<<"Frame done"<<endl;
        }
    }
    else
    {
        prev = imread(prev_frame_path ,IMREAD_COLOR);
        if ( prev.empty() )
        {
            cout<<"Cannot read image file: "<<prev;
            return -1;
        }
        cur = imread(cur_frame_path,IMREAD_COLOR);
        if ( cur.empty() )
        {
            cout<<"Cannot read image file: "<<cur;
            return -1;
        }

        for(int i=0;i<prev.rows;i+=8)
            for(int j=0;j<prev.cols;j+=8)
                points.push_back(Point2f(j,i));
        cvtColor(prev,prev_grayscale,COLOR_BGR2GRAY);
        cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
        calcOpticalFlowPyrLK(prev_grayscale,cur_grayscale,points,dst_points,status,err,Size(21,21));

        if(dst_path!="None")
        {
            Mat dense_flow;
            vector<SparseMatch> matches;
            for(int i=0;i<points.size();i++)
            {
                if(status[i]!=0)
                    matches.push_back(SparseMatch(points[i],dst_points[i]));
            }

            Ptr<EdgeAwareInterpolator> gd = createEdgeAwareInterpolator(false);
            double filtering_time = (double)getTickCount();
            gd->interpolate(prev,cur,matches,dense_flow);
            filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
            cout.precision(2);
            cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
            cout<<endl;

            Mat vis;
            VisualizeFlow(dense_flow,vis);
            imwrite(dst_path,vis);
        }
    }

    return 0;
}
