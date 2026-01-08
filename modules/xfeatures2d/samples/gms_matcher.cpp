#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

////////////////////////////////////////////////////
// This program demonstrates the GMS matching strategy.
int main(int argc, char* argv[])
{
    const char* keys =
        "{ h help        |                  | print help message  }"
        "{ l left        |                  | specify left (reference) image  }"
        "{ r right       |                  | specify right (query) image }"
        "{ camera        | 0                | specify the camera device number }"
        "{ nfeatures     | 10000            | specify the maximum number of ORB features }"
        "{ fastThreshold | 20               | specify the FAST threshold }"
        "{ drawSimple    | true             | do not draw not matched keypoints }"
        "{ withRotation  | false            | take rotation into account }"
        "{ withScale     | false            | take scale into account }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        std::cout << "Usage: gms_matcher [options]" << std::endl;
        std::cout << "Available options:" << std::endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    Ptr<Feature2D> orb = ORB::create(cmd.get<int>("nfeatures"));
    orb.dynamicCast<cv::ORB>()->setFastThreshold(cmd.get<int>("fastThreshold"));
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    if (!cmd.get<String>("left").empty() && !cmd.get<String>("right").empty())
    {
        Mat imgL = imread(cmd.get<String>("left"));
        Mat imgR = imread(cmd.get<String>("right"));

        std::vector<KeyPoint> kpRef, kpCur;
        Mat descRef, descCur;
        orb->detectAndCompute(imgL, noArray(), kpRef, descRef);
        orb->detectAndCompute(imgR, noArray(), kpCur, descCur);

        std::vector<DMatch> matchesAll, matchesGMS;
        matcher->match(descCur, descRef, matchesAll);

        matchGMS(imgR.size(), imgL.size(), kpCur, kpRef, matchesAll, matchesGMS, cmd.get<bool>("withRotation"), cmd.get<bool>("withScale"));
        std::cout << "matchesGMS: " << matchesGMS.size() << std::endl;

        Mat frameMatches;
        if (cmd.get<bool>("drawSimple"))
            drawMatches(imgR, kpCur, imgL, kpRef, matchesGMS, frameMatches, Scalar::all(-1), Scalar::all(-1),
                        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        else
            drawMatches(imgR, kpCur, imgL, kpRef, matchesGMS, frameMatches);
        imshow("Matches GMS", frameMatches);
        waitKey();
    }
    else
    {
        std::vector<KeyPoint> kpRef;
        Mat descRef;

        VideoCapture capture(cmd.get<int>("camera"));
        //Camera warm-up
        for (int i = 0; i < 10; i++)
        {
            Mat frame;
            capture >> frame;
        }

        Mat frameRef;
        for (;;)
        {
            Mat frame;
            capture >> frame;

            if (frameRef.empty())
            {
                frame.copyTo(frameRef);
                orb->detectAndCompute(frameRef, noArray(), kpRef, descRef);
            }

            TickMeter tm;
            tm.start();
            std::vector<KeyPoint> kp;
            Mat desc;
            orb->detectAndCompute(frame, noArray(), kp, desc);
            tm.stop();
            double t_orb = tm.getTimeMilli();

            tm.reset();
            tm.start();
            std::vector<DMatch> matchesAll, matchesGMS;
            matcher->match(desc, descRef, matchesAll);
            tm.stop();
            double t_match = tm.getTimeMilli();

            matchGMS(frame.size(), frameRef.size(), kp, kpRef, matchesAll, matchesGMS, cmd.get<bool>("withRotation"), cmd.get<bool>("withScale"));
            tm.stop();
            Mat frameMatches;
            if (cmd.get<bool>("drawSimple"))
                 drawMatches(frame, kp, frameRef, kpRef, matchesGMS, frameMatches, Scalar::all(-1), Scalar::all(-1),
                            std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            else
                 drawMatches(frame, kp, frameRef, kpRef, matchesGMS, frameMatches);

            String label = format("ORB: %.2f ms", t_orb);
            putText(frameMatches, label, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
            label = format("Matching: %.2f ms", t_match);
            putText(frameMatches, label, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
            label = format("GMS matching: %.2f ms", tm.getTimeMilli());
            putText(frameMatches, label, Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
            putText(frameMatches, "Press r to reinitialize the reference image.", Point(frameMatches.cols-380, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
            putText(frameMatches, "Press esc to quit.", Point(frameMatches.cols-180, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));

            imshow("Matches GMS", frameMatches);
            int c = waitKey(30);
            if (c == 27)
                break;
            else if (c == 'r')
            {
                frame.copyTo(frameRef);
                orb->detectAndCompute(frameRef, noArray(), kpRef, descRef);
            }
        }

    }

    return EXIT_SUCCESS;
}
