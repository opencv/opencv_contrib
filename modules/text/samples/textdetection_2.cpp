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
#include "opencv2/ml.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <queue>
#include <sys/time.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#define MIN_HEIGHT_RATIO 0.5

using namespace std;
using namespace cv;
using namespace cv::text;
using namespace cv::ml;

bool cmp(ERStat &a,ERStat &b);
vector< vector<ERStat> > Sort(Mat &src,vector<vector<ERStat> > &regions);

int main(int argc, const char * argv[]){
	
    Mat gray, src = imread(argv[1]);
	
    // Extract channels to be processed individually
    vector<Mat> channels;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    
    // Append negative channels to detect ER- (bright regions over dark background)
    channels.push_back(gray);
    channels.push_back(255-gray);
    
    int cn = (int)channels.size();
    
	//Text Spotting
	
    // Create ERFilter objects with the 1st and 2nd stage default classifiers
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
    //#pragma omp parallel for num_threads(2)
    for (int c=0; c < cn ; c++)
    {
        er_filter1[c]->run(channels[c], regions[c]);
        er_filter2[c]->run(channels[c], regions[c]);
    }
    
    //vector< vector<ERStat> > sort_regions = Sort( src , regions );
	
    // Detect character groups
    //cout << "Grouping extracted ERs ... ";
    vector< vector<Vec2i> > region_groups;
    vector<Rect> groups_boxes;
    erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
    //erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);
    //End Text Spotting
    
    //Text Recognition
    Pix *image = pixRead(argv[1]);
  	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
  	api->Init(NULL, "eng");
  	api->SetImage(image);
  	
  	int i;
    for(i = 0 ; i < groups_boxes.size(); i++){
    	api->SetRectangle(groups_boxes[i].x, groups_boxes[i].y, groups_boxes[i].width, groups_boxes[i].height);
    	char* text = api->GetUTF8Text();
    	int conf = api->MeanTextConf();
    			
		printf("%d %d %d %d ",groups_boxes[i].x, groups_boxes[i].y, groups_boxes[i].width, groups_boxes[i].height);
		if( strlen(text) == 0 or text == NULL )
			printf("\n");
		else{
			int j;
			for(j = 0; j < strlen(text); j++)
				if( text[j] != '\n' )
					printf("%c",text[j]);
				else
					printf(" ");
			printf("\n");
		}
    }
    
    api->End();
    pixDestroy(&image);
    
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
