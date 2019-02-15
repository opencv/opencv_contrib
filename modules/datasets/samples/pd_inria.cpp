#include "opencv2/datasets/pd_inria.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset }";

    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<PD_inria> dataset = PD_inria::create();
    dataset->load(path);

    size_t train_size = dataset->getTrain().size();
    size_t test_size = dataset->getTest().size();
    cout << "train size: " << train_size << endl;
    cout << "test size: " << test_size << endl;

    for( size_t i = 0; i < train_size; i++ )
    {
        PD_inriaObj *example = static_cast<PD_inriaObj *>(dataset->getTrain()[i].get());
        cout << "\ntrain object index: " << i << endl;
        cout << "file name: " << example->filename << endl;

        // image size
        cout << "image size: " << endl;
        cout << "  - width: " << example->width << endl;
        cout << "  - height: " << example->height << endl;
        cout << "  - depth: " << example->depth << endl;

        Mat img = imread( example->filename );

        // bounding boxes
        for ( size_t j = 0; j < example->bndboxes.size(); j++ )
        {
            cout << "object " << j << endl;
            int x = example->bndboxes[j].x;
            int y = example->bndboxes[j].y;
            cout << "  - xmin = " << x << endl;
            cout << "  - ymin = " << y << endl;
            cout << "  - xmax = " << example->bndboxes[j].width + x << endl;
            cout << "  - ymax = " << example->bndboxes[j].height + y << endl;
            rectangle( img, example->bndboxes[j], Scalar( 0, 0, 255 ), 2 );
        }

        imshow("INRIAPerson Dataset Train Images", img);

        cout << "\nPress a key to continue or ESC to exit." << endl;
        int key = waitKey();
        if( key == 27 ) break;
    }

    return 0;
}
