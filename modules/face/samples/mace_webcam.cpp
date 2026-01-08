// This file is part of the OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/face/mace.hpp"
#include <iostream>
using namespace cv;
using namespace cv::face;
using namespace std;


enum STATE {
    NEUTRAL,
    RECORD,
    PREDICT
};

const char *help =
        "press 'r' to record images. once N trainimages were recorded, train the mace filter\n"
        "press 'p' to predict (twofactor mode will switch back to neutral after each prediction attempt)\n"
        "press 's' to save a trained model\n"
        "press 'esc' to return\n"
        "any other key will reset to neutral state\n";

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv,
        "{ help h usage ? ||     show this help message }"
        "{ cascade c      ||     (required) path to a cascade file for face detection }"
        "{ pre p          ||     load a pretrained mace filter file, saved from previous session  (e.g. my.xml.gz) }"
        "{ num n          |50|   num train images }"
        "{ size s         |64|   image size }"
        "{ twofactor t    ||     pass phrase(text) for 2 factor authentification.\n"
        "                     (random convolute images seeded with the crc of this)\n"
        "                     users will get prompted to guess the secrect, additional to the image. }"
    );
    String cascade = parser.get<String>("cascade");
    if (parser.has("help") || cascade.empty()) {
        parser.printMessage();
        return 1;
    } else {
        cout << help << endl;
    }
    String defname = "mace.xml.gz";
    String pre = parser.get<String>("pre");
    String two = parser.get<String>("twofactor");
    int N = parser.get<int>("num");
    int Z = parser.get<int>("size");
    int state = NEUTRAL;

    Ptr<MACE> mace;
    if (! pre.empty()) { // load pretrained model, if available
        mace = MACE::load(pre);
        if (mace->empty()) {
            cerr << "loading the MACE failed !" << endl;
            return -1;
        }
        state = PREDICT;
    } else {
        mace  = MACE::create(Z);
        if (! two.empty()) {
            cout << "'" << two << "' initial passphrase" << endl;
            mace->salt(two);
        }
    }


    CascadeClassifier head(cascade);
    if (head.empty()) {
        cerr << "loading the cascade failed !" << endl;
        return -2;
    }

    VideoCapture cap(0);
    if (! cap.isOpened()) {
        cerr << "VideoCapture could not be opened !" << endl;
        return -3;
    }

    vector<Mat> train_img;
    while(1) {
        Mat frame;
        cap >> frame;

        vector<Rect> rects;
        head.detectMultiScale(frame,rects);
        if (rects.size()>0) {
            Scalar col = Scalar(0,120,0);

            if (state == RECORD) {
                if (train_img.size() >= size_t(N)) {
                    mace->train(train_img);
                    train_img.clear();
                    state = PREDICT;
                } else {
                    train_img.push_back(frame(rects[0]).clone());
                }
                col = Scalar(200,0,0);
            }

            if (state == PREDICT) {
                if (! two.empty()) { // prompt for secret on console
                    cout << "enter passphrase: ";
                    string pass;
                    getline(cin, pass);
                    mace->salt(pass);
                    state = NEUTRAL;
                    cout << "'" << pass << "' : ";
                }
                bool same = mace->same(frame(rects[0]));
                if (same) col = Scalar(0,220,220);
                else      col = Scalar(60,60,60);
                if (! two.empty()) {
                    cout << (same ? "accepted." : "denied.") << endl;
                }
            }

            rectangle(frame, rects[0], col, 2);
        }

        imshow("MACE",frame);
        int k = waitKey(10);
        switch (k) {
            case -1 : break;
            case 27 : return 0;
            default : state = NEUTRAL; break;
            case 'r': state = RECORD;  break;
            case 'p': state = PREDICT; break;
            case 's': mace->save(defname); break;
        }
    }

    return 0;
}
