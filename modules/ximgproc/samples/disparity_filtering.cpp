#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <map>

#if defined(_WIN32)
#include <direct.h>
#else 
#include <sys/stat.h>
#endif

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

#define UNKNOWN_DISPARITY 16320
 
static void print_help()
{
    printf("\nDemo for disparity filtering, evaluating speed and performance of different filters\n");
    printf("\nUsage: disparity_filtering.exe <path_to_dataset_folder> <path_to_results_folder>\n");
}

struct dataset_entry
{
    string name;
    string dataset_folder;
    string left_file,right_file,GT_file;
    dataset_entry(string _dataset_folder): dataset_folder(_dataset_folder){}
    void readEntry(Mat& dst_left,Mat& dst_right,Mat& dst_GT)
    {
        dst_left  = imread(dataset_folder+"/"+left_file,  IMREAD_COLOR);
        dst_right = imread(dataset_folder+"/"+right_file, IMREAD_COLOR);
        Mat raw_disp = imread(dataset_folder+"/"+GT_file, IMREAD_COLOR);
        dst_GT = Mat(raw_disp.rows,raw_disp.cols,CV_16S);
        for(int i=0;i<raw_disp.rows;i++)
            for(int j=0;j<raw_disp.cols;j++)
            {
                Vec3b bgrPixel = raw_disp.at<Vec3b>(i, j);
                dst_GT.at<short>(i,j) = 64*bgrPixel.val[2]+bgrPixel.val[1]/4; //16-multiplied disparity
            }
    }
};

struct config
{
    Ptr<StereoMatcher> matcher_instance;
    Ptr<DisparityFilter> filter_instance;
    config(Ptr<StereoMatcher> _matcher_instance,Ptr<DisparityFilter> _filter_instance)
    {
        matcher_instance = _matcher_instance;
        filter_instance = _filter_instance;
    }
    config() {}
};

void operator>>(const FileNode& node,dataset_entry& entry);
double computeMSE(Mat& GT, Mat& src, Rect ROI);
double computeBadPixelPercent(Mat& GT, Mat& src, Rect ROI, int thresh=24/*1.5 pixels*/);
void getDisparityVis(Mat& disparity_map,Mat& dst);
Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance);
void setConfigsForTesting(map<string,config>& cfgs);
void CreateDir(string path);

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        print_help();
        return 0;
    }
    string dataset_folder(argv[1]);
    string res_folder(argv[2]);

    map<string,config> configs_for_testing;
    setConfigsForTesting(configs_for_testing);
    CreateDir(res_folder);

    for (map<string,config>::iterator cfg = configs_for_testing.begin(); cfg != configs_for_testing.end(); cfg++)
    {
        string vis_folder = res_folder+"/vis_"+cfg->first;
        CreateDir(vis_folder);

        string cfg_file_name = res_folder+"/"+cfg->first+"_res.csv";
        FILE* cur_cfg_res_file = fopen(cfg_file_name.c_str(),"w");
        fprintf(cur_cfg_res_file,"Name,MSE,MSE after postfiltering,Percent bad,Percent bad after postfiltering,Matcher Execution Time(s),Filter Execution Time(s)\n");

        printf("Processing configuration: %s\n",cfg->first.c_str());

        FileStorage fs(dataset_folder + "/_dataset.xml", FileStorage::READ);
        FileNode n = fs["data_set"];
        double MSE_pre,percent_pre,MSE_post,percent_post,matching_time,filtering_time;
        double average_MSE_pre=0,average_percent_pre=0,average_MSE_post=0,
            average_percent_post=0,average_matching_time=0,average_filtering_time=0;
        int cnt = 0;
        for (FileNodeIterator it = n.begin(); it != n.end(); it++)
        {
            dataset_entry entry(dataset_folder);
            (*it)>>entry;
            printf("%s ",entry.name.c_str());
            Mat left,right,GT;
            entry.readEntry(left,right,GT);
            Mat raw_disp;
            Mat left_gray; cvtColor(left, left_gray, COLOR_BGR2GRAY );
            Mat right_gray; cvtColor(right, right_gray, COLOR_BGR2GRAY );
            matching_time = (double)getTickCount();
            cfg->second.matcher_instance->compute(left_gray,right_gray,raw_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();

            Rect ROI = computeROI(left.size(),cfg->second.matcher_instance);
            Mat filtered_disp;
            filtering_time = (double)getTickCount();
            cfg->second.filter_instance->filter(raw_disp,left,filtered_disp,ROI);
            filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();


            MSE_pre = computeMSE(GT,raw_disp,ROI);
            percent_pre = computeBadPixelPercent(GT,raw_disp,ROI);
            MSE_post = computeMSE(GT,filtered_disp,ROI);
            percent_post = computeBadPixelPercent(GT,filtered_disp,ROI);

            fprintf(cur_cfg_res_file,"%s,%.1f,%.1f,%.1f,%.1f,%.3f,%.3f\n",entry.name.c_str(),MSE_pre,MSE_post,
                percent_pre,percent_post,matching_time,filtering_time);

            average_matching_time+=matching_time; average_filtering_time+=filtering_time;
            average_MSE_pre+=MSE_pre; average_percent_pre+=percent_pre;
            average_MSE_post+=MSE_post; average_percent_post+=percent_post;
            cnt++;

            // dump visualizations:
            imwrite(vis_folder + "/" + entry.name + "_left.png",left);
            Mat GT_vis,raw_disp_vis,filtered_disp_vis;
            getDisparityVis(GT,GT_vis);
            getDisparityVis(raw_disp,raw_disp_vis);
            getDisparityVis(filtered_disp,filtered_disp_vis);
            imwrite(vis_folder + "/" + entry.name + "_disparity_GT.png",GT_vis);
            imwrite(vis_folder + "/" + entry.name + "_disparity_raw.png",raw_disp_vis);
            imwrite(vis_folder + "/" + entry.name + "_disparity_filtered.png",filtered_disp_vis);

            printf("- Done\n");

        }
        fprintf(cur_cfg_res_file,"%s,%.1f,%.1f,%.1f,%.1f,%.3f,%.3f\n","average",average_MSE_pre/cnt,
            average_MSE_post/cnt,average_percent_pre/cnt,average_percent_post/cnt,
            average_matching_time/cnt,average_filtering_time/cnt);
        fclose(cur_cfg_res_file);
    }
    return 0;
}

void operator>>(const FileNode& node,dataset_entry& entry) 
{
    node["name"] >> entry.name;
    node["left_file"] >> entry.left_file;
    node["right_file"] >> entry.right_file;
    node["GT_file"] >> entry.GT_file;
}

double computeMSE(Mat& GT, Mat& src, Rect ROI)
{
    double res = 0;
    Mat GT_ROI(GT,ROI);
    Mat src_ROI(src,ROI);
    int cnt=0;
    for(int i=0;i<src_ROI.rows;i++)
        for(int j=0;j<src_ROI.cols;j++)
        {
            if(GT_ROI.at<short>(i,j)!=UNKNOWN_DISPARITY)
            {
                res += (GT_ROI.at<short>(i,j) - src_ROI.at<short>(i,j))*(GT_ROI.at<short>(i,j) - src_ROI.at<short>(i,j));
                cnt++;
            }
        }
    res /= cnt*256;
    return res;
}

double computeBadPixelPercent(Mat& GT, Mat& src, Rect ROI, int thresh)
{
    int bad_pixel_num = 0;
    Mat GT_ROI(GT,ROI);
    Mat src_ROI(src,ROI);
    int cnt=0;
    for(int i=0;i<src_ROI.rows;i++)
        for(int j=0;j<src_ROI.cols;j++)
        {
            if(GT_ROI.at<short>(i,j)!=UNKNOWN_DISPARITY)
            {
                if( abs(GT_ROI.at<short>(i,j) - src_ROI.at<short>(i,j))>=thresh )
                    bad_pixel_num++;
                cnt++;
            }
        }
    return (100.0*bad_pixel_num)/cnt;
}

void getDisparityVis(Mat& disparity_map,Mat& dst)
{
    dst = Mat(disparity_map.rows,disparity_map.cols,CV_8UC3);
    for(int i=0;i<dst.rows;i++)
        for(int j=0;j<dst.cols;j++)
        {
            if(disparity_map.at<short>(i,j)==UNKNOWN_DISPARITY)
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                dst.at<Vec3b>(i,j) = Vec3b(saturate_cast<unsigned char>(disparity_map.at<short>(i,j)/8),
                                           saturate_cast<unsigned char>(disparity_map.at<short>(i,j)/8),
                                           saturate_cast<unsigned char>(disparity_map.at<short>(i,j)/8));
        }
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width - minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

void setConfigsForTesting(map<string,config>& cfgs)
{
    Ptr<StereoBM> stereobm_matcher = StereoBM::create(128,21); 
    stereobm_matcher->setTextureThreshold(0); 
    stereobm_matcher->setUniquenessRatio(0);

    Ptr<DisparityFilter> wls_filter = createDisparityWLSFilter();
    Ptr<DisparityFilter> dt_filter = createDisparityDTFilter();
    Ptr<DisparityFilter> guided_filter = createDisparityGuidedFilter();

    cfgs["stereobm_wls"] = config(stereobm_matcher,wls_filter);
    cfgs["stereobm_dtf"] = config(stereobm_matcher,dt_filter);
    cfgs["stereobm_gf"]  = config(stereobm_matcher,guided_filter);
}

void CreateDir(string path)
{
#if defined(_WIN32)
    _mkdir(path.c_str());
#else 
    mkdir(path.c_str(), 0777);
#endif
}
