
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking/tracking_by_matching.hpp>
#include <iostream>
#include <cstring>
#include "samples_utility.hpp"

//TODO: ifdef HAVE_DNN!!!!!!!!!!!!!!!!!!!!!!!!!

using namespace std;
using namespace cv;
using namespace cv::tbm;

static const char* keys =
{   "{video_name       | | video name                       }"
    "{start_frame      |0| Start frame                      }"
    "{frame_step       |5| Frame step                       }"
    "{detector_model   | | Path to detector's Caffe model   }"
    "{detector_weights | | Path to detector's Caffe weights }"
};

static void help()
{
  cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
       "-- pause video [p] and draw a bounding box around the target to start the tracker\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./tracker <tracker_algorithm> <video_name> <start_frame> [<bounding_frame>]\n"
       "tracker_algorithm can be: MIL, BOOSTING, MEDIANFLOW, TLD, KCF, GOTURN, MOSSE.\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
}

#ifdef HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
class DnnObjectDetector
{
public:
    DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
                      int desired_class_id=-1,
                      float confidence_threshold = 0.2,
                      //the following parameters are default for popular MobileNet_SSD caffe model
                      const String& net_input_name="data",
                      const String& net_output_name="detection_out",
                      double net_scalefactor=0.007843,
                      const Size& net_size = Size(300,300),
                      const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
                      bool net_swapRB=false)
        :desired_class_id(desired_class_id),
        confidence_threshold(confidence_threshold),
        net_input_name(net_input_name),
        net_output_name(net_output_name),
        net_scalefactor(net_scalefactor),
        net_size(net_size),
        net_mean(net_mean),
        net_swapRB(net_swapRB)
    {
        net = dnn::readNetFromCaffe(net_caffe_model_path, net_caffe_weights_path);
        if (net.empty())
            CV_Error(Error::StsError, "Cannot read Caffe net");
    }
    TrackedObjects detect(const cv::Mat& frame, int frame_idx)
    {
        Mat resized_frame;
        resize(frame, resized_frame, net_size);
        Mat inputBlob = cv::dnn::blobFromImage(resized_frame, net_scalefactor, net_size, net_mean, net_swapRB);

        net.setInput(inputBlob, net_input_name);
        Mat detection = net.forward(net_output_name);
        Mat detection_as_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        TrackedObjects res;
        for (int i = 0; i < detection_as_mat.rows; i++)
        {
            float cur_confidence = detection_as_mat.at<float>(i, 2);
            if (cur_confidence < confidence_threshold)
                continue;
            int cur_idx = static_cast<int>(detection_as_mat.at<float>(i, 1));
            if ((desired_class_id >= 0) && (cur_idx != desired_class_id))
                continue;

            int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
            int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
            int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
            int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);

            Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));
            //clipping by frame size
            cur_rect = cur_rect & Rect(Point(), frame.size());
            if (cur_rect.empty())
                continue;

            //TODO: remove it
            cout << "detectObjectsByNet: " << cur_rect << " " << cur_confidence << endl;

            TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1);
            res.push_back(cur_obj);
        }
        return res;
    }
private:
    cv::dnn::Net net;
    int desired_class_id;
    float confidence_threshold;
    String net_input_name;
    String net_output_name;
    double net_scalefactor;
    Size net_size;
    Scalar net_mean;
    bool net_swapRB;
};
#endif

std::unique_ptr<PedestrianTracker>
CreatePedestrianTracker(bool should_keep_tracking_info = false) {
    cv::tbm::TrackerParams params;

    if (should_keep_tracking_info) {
        params.drop_forgotten_tracks = false;
        params.max_num_objects_in_track = -1;
    }

    std::unique_ptr<PedestrianTracker> tracker(new PedestrianTracker(params));

    // Load reid-model.
    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(
            cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast =
        std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    return tracker;
}
int main( int argc, char** argv ){
    CommandLineParser parser( argc, argv, keys );
    std::unique_ptr<PedestrianTracker> tracker = CreatePedestrianTracker();

    String video_name = parser.get<String>("video_name");
    int start_frame = parser.get<int>("start_frame");
    int frame_step = parser.get<int>("frame_step");
    String detector_model = parser.get<String>("detector_model");
    String detector_weights = parser.get<String>("detector_weights");
    cout << "video_name + " << video_name << endl;

    if( video_name.empty() || detector_model.empty() || detector_weights.empty() )
    {
        help();
        return -1;
    }


    //open the capture
    VideoCapture cap;
    cap.open( video_name );
    cap.set( CAP_PROP_POS_FRAMES, start_frame );

    if( !cap.isOpened() )
    {
        help();
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        parser.printMessage();
        return -1;
    }

    // If you use the popular MobileNet_SSD detector, the default parameters may be used.
    // Otherwise, set your own parameters (net_mean, net_scalefactor, etc).
    DnnObjectDetector detector(detector_model, detector_weights);

    Mat frame;
    namedWindow( "Tracking by Matching", 1 );

    int frame_counter = 0;
    int64 time_total = 0;
    bool paused = false;
    for ( ;; )
    {
        if( paused )
        {
            char c = (char) waitKey( 2 );
            if( c == 'p' )
                paused = !paused;
            continue;
        }

        cap >> frame;
        if(frame.empty()){
            break;
        }


        int64 frame_time = getTickCount();

        TrackedObjects detections = detector.detect(frame, frame_counter);

        // timestamp in milliseconds
        uint64_t cur_timestamp = 1000.0 / 30 * frame_counter;
        tracker->Process(frame, detections, cur_timestamp);

        frame_time = getTickCount() - frame_time;
        time_total += frame_time;

        // Drawing colored "worms" (tracks).
        frame = tracker->DrawActiveTracks(frame);


        // Drawing all detected objects on a frame by BLUE COLOR
        for (const auto &detection : detections) {
            cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
        }

        // Drawing tracked detections only by RED color and print ID and detection
        // confidence level.
        for (const auto &detection : tracker->TrackedDetections()) {
            cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
            std::string text = std::to_string(detection.object_id) +
                " conf: " + std::to_string(detection.confidence);
            cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
                        1.0, cv::Scalar(0, 0, 255), 3);
        }

        imshow( "Tracking by Matching", frame );
        frame_counter++;

        char c = (char) waitKey( 2 );
        if( c == 'q' )
            break;
        if( c == 'p' )
            paused = !paused;

        double s = frame_counter / (time_total / getTickFrequency());
        printf("FPS: %f\n", s);
    }

return 0;
}
