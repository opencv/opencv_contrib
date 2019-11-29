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
            "{ help h usage ? |     | show this message }"
            "{ path p         |true | path to dataset }"
            "{ save s         |false| save resized positive images }"
            "{ rwidth rw      |64   | width of resized positive images }"
            "{ rheight rh     |128  | height of resized positive images }"
            "{ padding        |8    | vertical padding of resized positive images }";

    CommandLineParser parser(argc, argv, keys);
    bool savebbox = parser.get<bool>("save");
    int rwidth = parser.get<int>("rwidth");
    int rheight = parser.get<int>("rheight");
    int padding = parser.get<int>("padding");
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

    int bbox_count = 0;

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
            Rect obj_bndbox = example->bndboxes[j]; // bounding box of object
            cout << "  - bounding box: " << j << " - " << obj_bndbox << endl;

            int vpadding, hpadding;
            Rect ex_bndbox; // variable used for calculating expanded bounding box

            vpadding = cvRound(padding * obj_bndbox.height / rheight); // calculate vertical padding
            ex_bndbox.y = obj_bndbox.y - vpadding;
            ex_bndbox.height = 2 * vpadding + obj_bndbox.height;
            ex_bndbox.x = obj_bndbox.x + (obj_bndbox.width / 2);
            ex_bndbox.width = ex_bndbox.height * rwidth / rheight;
            ex_bndbox.x -= (ex_bndbox.width + 1) / 2;

            if (obj_bndbox.width > ex_bndbox.width)
            {
                obj_bndbox.x += (obj_bndbox.width - ex_bndbox.width + 1) / 2;
                obj_bndbox.width = ex_bndbox.width;
            }

            hpadding = obj_bndbox.x - ex_bndbox.x; // calculate horizontal padding

            if(savebbox)
            {
                Mat dst;
                copyMakeBorder(img(obj_bndbox), dst, vpadding, vpadding, hpadding, hpadding, BORDER_REFLECT);
                resize(dst, dst, Size(rwidth, rheight), 0, 0, INTER_AREA);
                imwrite(path + format("person_%04d.png", bbox_count++), dst);
            }
            else
                rectangle(img, obj_bndbox, Scalar(0, 0, 255), 2);
        }

        if (savebbox)
            continue; // skip UI updates

        imshow("INRIAPerson Dataset Train Images", img);
        cout << "\nPress a key to continue or ESC to exit." << endl;
        int key = waitKey();
        if( key == 27 ) break;
    }

    return 0;
}
