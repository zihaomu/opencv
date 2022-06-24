#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
int main(int argc, char *argv[])
{
    VideoCapture ob3dCapture(0, CAP_OB3D);
    while (true)
    {
        Mat image;
        Mat depthMap;
        Mat adjDepthMap;
        Mat irImage;
        Mat adjIrImage;

        namedWindow("DEPTH");
        resizeWindow("DEPTH", 640, 480);
        namedWindow("IR");
        resizeWindow("IR", 640, 480);
        namedWindow("RGB");
        resizeWindow("RGB", 640, 480);

        ob3dCapture.grab();
        if (ob3dCapture.retrieve(depthMap, CAP_OB3D_DEPTH_MAP))
        {
            normalize(depthMap, adjDepthMap, 0, 255, NORM_MINMAX, CV_8UC1);
            applyColorMap(adjDepthMap, adjDepthMap, COLORMAP_JET);
            imshow("DEPTH", adjDepthMap);
        }

        if (ob3dCapture.retrieve(irImage, CAP_OB3D_IR_IMAGE))
        {
            normalize(irImage, adjIrImage, 0, 255, NORM_MINMAX, CV_8UC1);
            imshow("IR", adjIrImage);
        }

        if (ob3dCapture.retrieve(image, CAP_OB3D_BGR_IMAGE))
        {
            imshow("RGB", image);
        }

        if (waitKey(30) >= 0)
            break;
    }
    return 0;
}
