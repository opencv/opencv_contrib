/*
 * webcam-demo.cpp
 *
 * A demo program of End-to-end Scene Text Detection and Recognition.
 *
 * Created on: Jul 31, 2014
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::text;

//ERStat extraction is done in parallel for different channels
class Parallel_extractCSER: public cv::ParallelLoopBody
{
private:
    vector<Mat> &channels;
    vector< vector<ERStat> > &regions;
    vector< Ptr<ERFilter> > er_filter1;
    vector< Ptr<ERFilter> > er_filter2;

public:
    Parallel_extractCSER(vector<Mat> &_channels, vector< vector<ERStat> > &_regions,
                         vector<Ptr<ERFilter> >_er_filter1, vector<Ptr<ERFilter> >_er_filter2)
        : channels(_channels),regions(_regions),er_filter1(_er_filter1),er_filter2(_er_filter2){}

    virtual void operator()( const cv::Range &r ) const
    {
        for (int c=r.start; c < r.end; c++)
        {
            er_filter1[c]->run(channels[c], regions[c]);
            er_filter2[c]->run(channels[c], regions[c]);
        }
    }
    Parallel_extractCSER & operator=(const Parallel_extractCSER &a);
};

//OCR recognition is done in parallel for different detections
template <class T>
class Parallel_OCR: public cv::ParallelLoopBody
{
private:
    vector<Mat> &detections;
    vector<string> &outputs;
    vector< vector<Rect> > &boxes;
    vector< vector<string> > &words;
    vector< vector<float> > &confidences;
    vector< Ptr<T> > &ocrs;

public:
    Parallel_OCR(vector<Mat> &_detections, vector<string> &_outputs, vector< vector<Rect> > &_boxes,
                 vector< vector<string> > &_words, vector< vector<float> > &_confidences,
                 vector< Ptr<T> > &_ocrs)
        : detections(_detections), outputs(_outputs), boxes(_boxes), words(_words),
          confidences(_confidences), ocrs(_ocrs)
    {}

    virtual void operator()( const cv::Range &r ) const
    {
        for (int c=r.start; c < r.end; c++)
        {
            ocrs[c%ocrs.size()]->run(detections[c], outputs[c], &boxes[c], &words[c], &confidences[c], OCR_LEVEL_WORD);
        }
    }
    Parallel_OCR & operator=(const Parallel_OCR &a);
};


//Discard wrongly recognised strings
bool   isRepetitive(const string& s);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);

//Perform text detection and recognition from webcam
int main(int argc, char* argv[])
{
    cout << endl << argv[0] << endl << endl;
    cout << "A demo program of End-to-end Scene Text Detection and Recognition using webcam." << endl << endl;
    cout << "  Usage:  " << argv[0] << " [camera_index]" << endl << endl;
    cout << "  Press 'r' to switch between MSER/CSER regions." << endl;
    cout << "  Press 'g' to switch between Horizontal and Arbitrary oriented grouping." << endl;
    cout << "  Press 'o' to switch between OCRTesseract/OCRHMMDecoder recognition." << endl;
    cout << "  Press 's' to scale down frame size to 320x240." << endl;
    cout << "  Press 'ESC' to exit." << endl << endl;

    namedWindow("recognition",WINDOW_NORMAL);
    bool downsize = false;
    int  REGION_TYPE = 1;
    int  GROUPING_ALGORITHM = 0;
    int  RECOGNITION = 0;
    char *region_types_str[2] = {const_cast<char *>("ERStats"), const_cast<char *>("MSER")};
    char *grouping_algorithms_str[2] = {const_cast<char *>("exhaustive_search"), const_cast<char *>("multioriented")};
    char *recognitions_str[2] = {const_cast<char *>("Tesseract"), const_cast<char *>("NM_chain_features + KNN")};

    Mat frame,grey,orig_grey,out_img;
    vector<Mat> channels;
    vector<vector<ERStat> > regions(2); //two channels

    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    // since er algorithm is not reentrant we need one filter for channel
    vector< Ptr<ERFilter> > er_filters1;
    vector< Ptr<ERFilter> > er_filters2;
    for (int i=0; i<2; i++)
    {
        Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
        Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);
        er_filters1.push_back(er_filter1);
        er_filters2.push_back(er_filter2);
    }

    //double t_r = getTickCount();

    //Initialize OCR engine (we initialize 10 instances in order to work several recognitions in parallel)
    cout << "Initializing OCR engines ..." << endl;
    int num_ocrs = 10;
    vector< Ptr<OCRTesseract> > ocrs;
    for (int o=0; o<num_ocrs; o++)
    {
      ocrs.push_back(OCRTesseract::create());
    }

    Mat transition_p;
    string filename = "OCRHMM_transitions_table.xml";
    FileStorage fs(filename, FileStorage::READ);
    fs["transition_probabilities"] >> transition_p;
    fs.release();
    Mat emission_p = Mat::eye(62,62,CV_64FC1);
    string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    vector< Ptr<OCRHMMDecoder> > decoders;
    for (int o=0; o<num_ocrs; o++)
    {
      decoders.push_back(OCRHMMDecoder::create(loadOCRHMMClassifierNM("OCRHMM_knn_model_data.xml.gz"),
                                               voc, transition_p, emission_p));
    }
    cout << " Done!" << endl;

    //cout << "TIME_OCR_INITIALIZATION_ALT = "<< ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;


    int cam_idx = 0;
    if (argc > 1)
        cam_idx = atoi(argv[1]);

    VideoCapture cap(cam_idx);
    if(!cap.isOpened())
    {
        cout << "ERROR: Cannot open default camera (0)." << endl;
        return -1;
    }

    while (cap.read(frame))
    {
        double t_all = (double)getTickCount();

        if (downsize)
            resize(frame,frame,Size(320,240));

        /*Text Detection*/

        cvtColor(frame,grey,COLOR_RGB2GRAY);
        grey.copyTo(orig_grey);
        // Extract channels to be processed individually
        channels.clear();
        channels.push_back(grey);
        channels.push_back(255-grey);


        regions[0].clear();
        regions[1].clear();
        //double t_d = (double)getTickCount();

        switch (REGION_TYPE)
        {
        case 0:
        {
            parallel_for_(cv::Range(0,(int)channels.size()), Parallel_extractCSER(channels,regions,er_filters1,er_filters2));
            break;
        }
        case 1:
        {
            //Extract MSER
            vector<vector<Point> > contours;
            vector<Rect> bboxes;
            Ptr<MSER> mser = MSER::create(21,(int)(0.00002*grey.cols*grey.rows),(int)(0.05*grey.cols*grey.rows),1,0.7);
            mser->detectRegions(grey, contours, bboxes);

            //Convert the output of MSER to suitable input for the grouping/recognition algorithms
            if (contours.size() > 0)
                MSERsToERStats(grey, contours, regions);

            break;
        }
        case 2:
        {
            break;
        }
        }
        //cout << "TIME_REGION_DETECTION_ALT = " << ((double)getTickCount() - t_d)*1000/getTickFrequency() << endl;

        // Detect character groups
        //double t_g = getTickCount();
        vector< vector<Vec2i> > nm_region_groups;
        vector<Rect> nm_boxes;
        switch (GROUPING_ALGORITHM)
        {
        case 0:
        {
            erGrouping(frame, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);
            break;
        }
        case 1:
        {
            erGrouping(frame, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);
            break;
        }
        }
        //cout << "TIME_GROUPING_ALT = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;




        /*Text Recognition (OCR)*/


        frame.copyTo(out_img);
        int scale = downsize ? 2 : 1;
        float scale_img  = (float)((600.f/frame.rows)/scale);
        float scale_font = (float)(2-scale_img)/1.4f;
        vector<string> words_detection;
        float min_confidence1 = 0.f, min_confidence2 = 0.f;

        if (RECOGNITION == 0)
        {
          min_confidence1 = 51.f; min_confidence2 = 60.f;
        }

        vector<Mat> detections;

        //t_r = getTickCount();

        for (int i=0; i<(int)nm_boxes.size(); i++)
        {
            rectangle(out_img, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(255,255,0),3);


            Mat group_img = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);
            er_draw(channels, regions, nm_region_groups[i], group_img);
            group_img(nm_boxes[i]).copyTo(group_img);
            copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));
            detections.push_back(group_img);
        }
        vector<string> outputs((int)detections.size());
        vector< vector<Rect> > boxes((int)detections.size());
        vector< vector<string> > words((int)detections.size());
        vector< vector<float> > confidences((int)detections.size());

        // parallel process detections in batches of ocrs.size() (== num_ocrs)
        for (int i=0; i<(int)detections.size(); i=i+(int)num_ocrs)
        {
          Range r;
          if (i+(int)num_ocrs <= (int)detections.size())
            r = Range(i,i+(int)num_ocrs);
          else
            r = Range(i,(int)detections.size());

          switch(RECOGNITION)
          {
            case 0:
              parallel_for_(r, Parallel_OCR<OCRTesseract>(detections, outputs, boxes, words, confidences, ocrs));
              break;
            case 1:
              parallel_for_(r, Parallel_OCR<OCRHMMDecoder>(detections, outputs, boxes, words, confidences, decoders));
              break;
          }
        }


        for (int i=0; i<(int)detections.size(); i++)
        {

            outputs[i].erase(remove(outputs[i].begin(), outputs[i].end(), '\n'), outputs[i].end());
            //cout << "OCR output = \"" << outputs[i] << "\" lenght = " << outputs[i].size() << endl;
            if (outputs[i].size() < 3)
                continue;

            for (int j=0; j<(int)boxes[i].size(); j++)
            {
                boxes[i][j].x += nm_boxes[i].x-15;
                boxes[i][j].y += nm_boxes[i].y-15;

                //cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
                if ((words[i][j].size() < 2) || (confidences[i][j] < min_confidence1) ||
                        ((words[i][j].size()==2) && (words[i][j][0] == words[i][j][1])) ||
                        ((words[i][j].size()< 4) && (confidences[i][j] < min_confidence2)) ||
                        isRepetitive(words[i][j]))
                    continue;
                words_detection.push_back(words[i][j]);
                rectangle(out_img, boxes[i][j].tl(), boxes[i][j].br(), Scalar(255,0,255),3);
                Size word_size = getTextSize(words[i][j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3*scale_font), NULL);
                rectangle(out_img, boxes[i][j].tl()-Point(3,word_size.height+3), boxes[i][j].tl()+Point(word_size.width,0), Scalar(255,0,255),-1);
                putText(out_img, words[i][j], boxes[i][j].tl()-Point(1,1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255,255,255),(int)(3*scale_font));
            }

        }

        //cout << "TIME_OCR_ALT = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;


        t_all = ((double)getTickCount() - t_all)*1000/getTickFrequency();
        char buff[100];
        sprintf(buff, "%2.1f Fps. @ %dx%d", (float)(1000/t_all), out_img.cols, out_img.rows);
        string fps_info = buff;
        rectangle(out_img, Point( out_img.rows-(160/scale),out_img.rows-(70/scale) ), Point(out_img.cols,out_img.rows), Scalar(255,255,255),-1);
        putText(out_img, fps_info, Point( 10,out_img.rows-(10/scale) ), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255,0,0));
        putText(out_img, region_types_str[REGION_TYPE], Point( out_img.rows-(150/scale),out_img.rows-(50/scale) ), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255,0,0));
        putText(out_img, grouping_algorithms_str[GROUPING_ALGORITHM], Point( out_img.rows-(150/scale),out_img.rows-(30/scale) ), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255,0,0));
        putText(out_img, recognitions_str[RECOGNITION], Point( out_img.rows-(150/scale),out_img.rows-(10/scale) ), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255,0,0));


        imshow("recognition", out_img);
        //imwrite("recognition_alt.jpg", out_img);
        int key = waitKey(30);
        if (key == 27) //wait for key
        {
            cout << "esc key pressed" << endl;
            break;
        }
        else
        {
            switch (key)
            {
            case 103: //g
                GROUPING_ALGORITHM = (GROUPING_ALGORITHM+1)%2;
                cout << "Grouping switched to " << grouping_algorithms_str[GROUPING_ALGORITHM] << endl;
                break;
            case 111: //o
                RECOGNITION = (RECOGNITION+1)%2;
                cout << "OCR switched to " << recognitions_str[RECOGNITION] << endl;
                break;
            case 114: //r
                REGION_TYPE = (REGION_TYPE+1)%2;
                cout << "Regions switched to " << region_types_str[REGION_TYPE] << endl;
                break;
            case 115: //s
                downsize = !downsize;
                break;
            default:
                break;

            }
        }

    }

    return 0;
}

bool isRepetitive(const string& s)
{
    int count  = 0;
    int count2 = 0;
    int count3 = 0;
    int first=(int)s[0];
    int last=(int)s[(int)s.size()-1];
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
        if((int)s[i]==first)
            count2++;
        if((int)s[i]==last)
            count3++;
    }
    if ((count > ((int)s.size()+1)/2) || (count2 == (int)s.size()) || (count3 > ((int)s.size()*2)/3))
    {
        return true;
    }


    return false;
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
