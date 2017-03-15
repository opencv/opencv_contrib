#ifndef _ROISELECTOR_HPP_
#define _ROISELECTOR_HPP_

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

cv::Rect2d selectROI(cv::Mat img, bool fromCenter = true);
cv::Rect2d selectROI(const cv::String& windowName, cv::Mat img, bool showCrossair = true, bool fromCenter = true);
void selectROI(const cv::String& windowName, cv::Mat img, std::vector<cv::Rect2d> & boundingBox, bool fromCenter = true);

//==================================================================================================

class ROISelector
{
  public:
    cv::Rect2d select(cv::Mat img, bool fromCenter = true)
    {
        return select("ROI selector", img, fromCenter);
    }

    cv::Rect2d select(const cv::String &windowName, cv::Mat img, bool showCrossair = true, bool fromCenter = true)
    {

        key = 0;

        // set the drawing mode
        selectorParams.drawFromCenter = fromCenter;

        // show the image and give feedback to user
        cv::imshow(windowName, img);

        // copy the data, rectangle should be drawn in the fresh image
        selectorParams.image = img.clone();

        // select the object
        cv::setMouseCallback(windowName, mouseHandler, (void *)&selectorParams);

        // end selection process on SPACE (32) ESC (27) or ENTER (13)
        while (!(key == 32 || key == 27 || key == 13))
        {
            // draw the selected object
            cv::rectangle(selectorParams.image, selectorParams.box, cv::Scalar(255, 0, 0), 2, 1);

            // draw cross air in the middle of bounding box
            if (showCrossair)
            {
                // horizontal line
                cv::line(selectorParams.image,
                     cv::Point((int)selectorParams.box.x,
                           (int)(selectorParams.box.y + selectorParams.box.height / 2)),
                     cv::Point((int)(selectorParams.box.x + selectorParams.box.width),
                           (int)(selectorParams.box.y + selectorParams.box.height / 2)),
                     cv::Scalar(255, 0, 0), 2, 1);

                // vertical line
                cv::line(selectorParams.image,
                     cv::Point((int)(selectorParams.box.x + selectorParams.box.width / 2),
                           (int)selectorParams.box.y),
                     cv::Point((int)(selectorParams.box.x + selectorParams.box.width / 2),
                           (int)(selectorParams.box.y + selectorParams.box.height)),
                     cv::Scalar(255, 0, 0), 2, 1);
            }

            // show the image bouding box
            cv::imshow(windowName, selectorParams.image);

            // reset the image
            selectorParams.image = img.clone();

            // get keyboard event, extract lower 8 bits for scancode comparison
            key = cv::waitKey(1) & 0xFF;
        }

        return selectorParams.box;
    }

    void select(const cv::String &windowName, cv::Mat img, std::vector<cv::Rect2d> &boundingBox, bool fromCenter = true)
    {
        std::vector<cv::Rect2d> box;
        cv::Rect2d temp;
        key = 0;

        // show notice to user
        printf("Select an object to track and then press SPACE or ENTER button!\n");
        printf("Finish the selection process by pressing ESC button!\n");

        // while key is not ESC (27)
        for (;;)
        {
            temp = select(windowName, img, true, fromCenter);
            if (key == 27)
                break;
            if (temp.width > 0 && temp.height > 0)
                box.push_back(temp);
        }
        boundingBox = box;
    }

    struct handlerT
    {
        // basic parameters
        bool isDrawing;
        cv::Rect2d box;
        cv::Mat image;

        // parameters for drawing from the center
        bool drawFromCenter;
        cv::Point2f center;

        // initializer list
        handlerT() : isDrawing(false), drawFromCenter(true){};
    } selectorParams;

    // to store the tracked objects
    std::vector<handlerT> objects;

  private:
    static void mouseHandler(int event, int x, int y, int flags, void *param)
    {
        ROISelector *self = static_cast<ROISelector *>(param);
        self->opencv_mouse_callback(event, x, y, flags, param);
    }

    void opencv_mouse_callback(int event, int x, int y, int, void *param)
    {
        handlerT *data = (handlerT *)param;
        switch (event)
        {
        // update the selected bounding box
        case cv::EVENT_MOUSEMOVE:
            if (data->isDrawing)
            {
                if (data->drawFromCenter)
                {
                    data->box.width = 2 * (x - data->center.x) /*data->box.x*/;
                    data->box.height = 2 * (y - data->center.y) /*data->box.y*/;
                    data->box.x = data->center.x - data->box.width / 2.0;
                    data->box.y = data->center.y - data->box.height / 2.0;
                }
                else
                {
                    data->box.width = x - data->box.x;
                    data->box.height = y - data->box.y;
                }
            }
            break;

        // start to select the bounding box
        case cv::EVENT_LBUTTONDOWN:
            data->isDrawing = true;
            data->box = cv::Rect2d(x, y, 0, 0);
            data->center = cv::Point2f((float)x, (float)y);
            break;

        // cleaning up the selected bounding box
        case cv::EVENT_LBUTTONUP:
            data->isDrawing = false;
            if (data->box.width < 0)
            {
                data->box.x += data->box.width;
                data->box.width *= -1;
            }
            if (data->box.height < 0)
            {
                data->box.y += data->box.height;
                data->box.height *= -1;
            }
            break;
        }
    }

    // save the keypressed characted
    int key;
};

//==================================================================================================

static ROISelector _selector;

cv::Rect2d selectROI(cv::Mat img, bool fromCenter)
{
    return _selector.select("ROI selector", img, true, fromCenter);
};

cv::Rect2d selectROI(const cv::String &windowName, cv::Mat img, bool showCrossair, bool fromCenter)
{
    printf("Select an object to track and then press SPACE or ENTER button!\n");
    return _selector.select(windowName, img, showCrossair, fromCenter);
};

void selectROI(const cv::String &windowName, cv::Mat img, std::vector<cv::Rect2d> &boundingBox, bool fromCenter)
{
    return _selector.select(windowName, img, boundingBox, fromCenter);
}

#endif // _ROISELECTOR_HPP_
