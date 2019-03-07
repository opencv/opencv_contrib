/*
 * textdetection_2.cpp
 *
 * A demo program based on the code of Lluis Gomez i Bigorda provide by opencv_contrib
 *
 * Created on: August, 2018
 *     Author: UNICAMP-Samsung group - Contact: mpereira at ic.unicamp.br
 */

#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <omp.h>
#include <iomanip>
#define MIN_HEIGHT_RATIO 0.5

using namespace std;
using namespace cv;
using namespace cv::text;

bool cmp(ERStat &a,ERStat &b);
vector< vector<ERStat> > Sort(Mat &src,vector<vector<ERStat> > &regions);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);


int main(int argc, const char * argv[]){
	
	if(argc != 2){
		cout << "Error: argv(1) = ImagePath" << endl;
		return 0;
	}
	
	Mat src = imread(argv[1]);
    Mat gray;
       
	
    // Extract channels to be processed individually
    vector<Mat> channels;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    
    // Append gray and negative channel to detect ER- (bright regions over dark background)
    channels.push_back(gray);
    channels.push_back(255-gray);
    
    int cn = (int)channels.size();
    
	//Text Spotting
	
    // Create ERFilter objects with the 1st and 2nd stage default classifiers for each channel
    Ptr<ERFilter> er_filter1[2];
    Ptr<ERFilter> er_filter2[2];
    
    
    for (int c=0; c < cn ; c++)
    {
        er_filter1[c] = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),16,0.00015f,0.13f,0.2f,true,0.1f);
        er_filter2[c] = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);
    }
    
    //er_filter1[0] = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),16,0.00015f,0.13f,0.2f,true,0.1f);
    //Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);
	//Ptr<M_ERFilter> er_filter2 = M_createERFilterNM2(M_loadClassifierNM2("trained_classifierNM2.xml"),0.5);
	
	
    vector<vector<ERStat> > regions(cn);
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    //cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << endl;
    //cout << "    (...) this may take a while (...)" << endl << endl;
    //Parallelizing the process of region extraction
    #pragma omp parallel for num_threads(2)
    for (int c=0; c < cn ; c++)
    {
        er_filter1[c]->run(channels[c], regions[c]);
        er_filter2[c]->run(channels[c], regions[c]);
    }
    
    vector< vector<ERStat> > sort_regions = Sort( src , regions );
	
    // Detect character groups
    //cout << "Grouping extracted ERs ... ";
    vector< vector<Vec2i> > region_groups;
    vector<Rect> groups_boxes;
    erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
    //erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);
    //End Text Spotting
    
    Mat out_img;
    Mat out_img_detection;
    Mat out_img_segmentation = Mat::zeros(src.rows+2, src.cols+2, CV_8UC1);
    src.copyTo(out_img);
    src.copyTo(out_img_detection);
    string output;
    
    //Text Recognition
    Ptr<OCRTesseract> ocr = OCRTesseract::create();
  	 
    for(int i = 0 ; i < groups_boxes.size(); i++){
    	rectangle(out_img_detection, groups_boxes[i].tl(), groups_boxes[i].br(), Scalar(0,255,255), 3);
    	Mat group_img = Mat::zeros(src.rows+2, src.cols+2, CV_8UC1);
        er_draw(channels, regions, region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        //image(nm_boxes[i]).copyTo(group_img);
        group_img(groups_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));
        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);
        printf("x: %d - y: %d - width: %d - height: %d ",\
        groups_boxes[i].x, groups_boxes[i].y, groups_boxes[i].width, groups_boxes[i].height);
        printf("- text: ");
		for(int j = 0; j < output.size(); j++){
			if( output[j] != '\n' )
				printf("%c",output[j]);
			else
				printf(" ");
		}
		printf("\n");
    }
    
    // memory clean-up
    er_filter1[0].release();
    er_filter1[1].release(); 
    regions.clear();
    if (!groups_boxes.empty())
    {
        groups_boxes.clear();
    }
    
    return 0;
}

void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

vector<vector<ERStat>> Sort(Mat &src,vector<vector<ERStat> > &regions){
	vector< vector<ERStat> > ans;
	for(int c = 0 ; c < regions.size() ; c++){
		vector<ERStat> tmp;
		for(int i = 1 ; i < regions[c].size() ; i++){
			bool inside = 0;
			for(int j = 0 ; j < tmp.size() ; j++){
				Rect area = regions[c][i].rect | tmp[j].rect;
				
				if( area == regions[c][i].rect or area == tmp[j].rect ){
					inside = 1;
					break;
				}
			}
			if( inside )
				continue;
				
			tmp.push_back( regions[c][i] );
		
			for(int j = tmp.size() - 2 ; j >= 0 ; j--){
				if( cmp( tmp[j] , tmp[j+1] ) ){
					swap( tmp[j] , tmp[j+1] );
				}
			}
		
		}
		ans.push_back( tmp );
	}
	return ans;
}

bool cmp(ERStat &region_1,ERStat &region_2){
	
	Point center_i(region_1.rect.x + region_1.rect.width/2, region_1.rect.y + region_1.rect.height/2);
    Point center_j(region_2.rect.x + region_2.rect.width/2, region_2.rect.y + region_2.rect.height/2);
    float centroid_angle = (float)atan2((float)(center_j.y - center_i.y), (float)(center_j.x - center_i.x));
  	
	float height_ratio = (float)min(region_1.rect.height,region_2.rect.height) \
	 					 / max(region_1.rect.height,region_2.rect.height);
	
	if( height_ratio < MIN_HEIGHT_RATIO  ){
		return region_2.rect.y < region_1.rect.y;
	}
	
	if( region_2.rect.y + region_2.rect.height * .95 <= region_1.rect.y )
		return 1;
	
	if( (region_1.rect.y <= region_2.rect.y and region_1.rect.y + region_1.rect.height >= region_2.rect.y \
	 	+ region_2.rect.height) or (region_1.rect.y - 0.5 * region_1.rect.height <= region_2.rect.y \
	 	and region_1.rect.y + 0.5 * region_2.rect.height >= region_2.rect.y) ) 
		return region_2.rect.x < region_1.rect.x;
	
	if( region_1.rect.y > region_2.rect.y )
		return 1;
	
	return 0;
}
