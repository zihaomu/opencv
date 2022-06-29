#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
int main(int argc, char *argv[])
{
    VideoCapture obsensorCapture(0, CAP_OB_SENSOR);

    Mat image;
    Mat depthMap;
    Mat adjDepthMap;
    Mat irImage;
    Mat adjIrImage;

    while (true)
    {
        // obsensorCapture >> depthMap;
        // if (!depthMap.empty())
        // {
        //     normalize(depthMap, adjDepthMap, 0, 255, NORM_MINMAX, CV_8UC1);
        //     applyColorMap(adjDepthMap, adjDepthMap, COLORMAP_JET);
        //     imshow("DEPTH", adjDepthMap);
        // }

        if(obsensorCapture.grab()){
            if (obsensorCapture.retrieve(image, CAP_OB_SENSOR_BGR_IMAGE))
            {
                imshow("RGB", image);
            }

            if (obsensorCapture.retrieve(depthMap, CAP_OB_SENSOR_DEPTH_MAP))
            {
                normalize(depthMap, adjDepthMap, 0, 255, NORM_MINMAX, CV_8UC1);
                applyColorMap(adjDepthMap, adjDepthMap, COLORMAP_JET);
                imshow("DEPTH", adjDepthMap);
            }

            if (obsensorCapture.retrieve(irImage, CAP_OB_SENSOR_IR_IMAGE))
            {
                normalize(irImage, adjIrImage, 0, 255, NORM_MINMAX, CV_8UC1);
                imshow("IR", adjIrImage);
            }
        }

        if (waitKey(30) >= 0)
            break;
    }
    return 0;
}
