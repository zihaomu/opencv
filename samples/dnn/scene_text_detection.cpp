/*
 * Real-time Scene Text Detection with Differentiable Binarization
 * Copyright (C) 2020, The patent is owned by <Huazhong University of Science and Technology>
 * Author List:
 *      Minghui Liao    <Huazhong University of Science and Technology>
 *      Zhaoyi Wan      <Megvii>
 *      Cong Yao        <Megvii>
 *      Kai Chen        <Shanghai Jiao Tong University>
 *      Xiang Bai       <Huazhong University of Science and Technology>, Corresponding author
 *
 * This script is written by Wenqing Zhang <Huazhong University of Science and Technology>.
 *
 * The code has been contributed to OpenCV under the terms of Apache 2 license:
 * https://www.apche.org/licenses/LICENSE-2.0
*/

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

std::string keys =
        "{ help  h                          | | Print help message. }"
        "{ inputImage i                     | | Path to an input image. Skip this argument to capture frames from a camera. }"
        "{ device d                         | 0 | camera device number. }"
        "{ modelPath mp                     | | Path to a binary .onnx file contains trained DB detector model.}"
        "{ binaryThreshold bt               | 0.3 | Confidence threshold of the binary map. }"
        "{ polygonThreshold pt              | 0.5 | Confidence threshold of polygons. }"
        "{ maxCandidate mc                  | 100 | Max candidates of polygons. }";


void getTextPolygons(const Mat & binary, float binThresh, float polyThresh, size_t maxCandidates,
                     std::vector<std::vector<Point>> & ploygons);

double boxFastScore(const Mat & binary, std::vector<Point> & contour);

void unclip(std::vector<Point> &inPoly, std::vector<Point> &outPoly, double ratio);

int main(int argc, char** argv)
{
    // Parse arguments
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run the official PyTorch implementation (https://github.com/MhLiao/DB) of "
                 "Real-time Scene Text Detection with Differentiable Binarization (https://arxiv.org/abs/1911.08947)\n"
                 "The current version of this script is a variant of the original network w/o deformable convolution");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float binThresh = parser.get<float>("binaryThreshold");
    float polyThresh = parser.get<float>("polygonThreshold");
    uint maxCandidates = parser.get<int>("maxCandidate");
    String modelPath = parser.get<String>("modelPath");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load the network
    CV_Assert(!modelPath.empty());
    Net detector = readNet(modelPath);

    // Create a window
    static const std::string winName = "Real-time Scene Text Detection with Differentiable Binarization";
    namedWindow(winName, WINDOW_NORMAL);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("inputImage"))
        cap.open(parser.get<String>("inputImage"));
    else
        cap.open(parser.get<int>("device"));

    // Detect
    Mat frame, blob;
    double scale = 1.0 / 255.0;
    Size inputSize = Size(736, 736);
    Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        // Create a 4D blob from a frame
        blob = blobFromImage(frame, scale, inputSize, mean, false, false);

        // Set input blob
        detector.setInput(blob);

        // Network Inference
        Mat binary = detector.forward("out");

        // Post-Processing
        std::vector<std::vector<Point>> polygons;
        getTextPolygons(binary, binThresh, polyThresh, maxCandidates, polygons);

        Mat dst;
        resize(frame, dst, Size(736, 736));
        polylines(dst, polygons, true, Scalar(0, 255, 0), 2);

        imshow(winName, dst);
    }

    return 0;
}


void getTextPolygons(const Mat & binary, float binThresh, float polyThresh, size_t maxCandidate,
                     std::vector<std::vector<Point>> & ploygons)
{
    // Threshold
    Mat bitmap;
    threshold(binary, bitmap, binThresh, 255, THRESH_BINARY);

    // Find Contours
    std::vector<std::vector<Point>> contours;
    bitmap.convertTo(bitmap, CV_8UC1);
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // filter for candidates
    size_t numCandidate = 0;
    if (contours.size() < maxCandidate) {
        numCandidate = contours.size();
    } else {
        numCandidate = maxCandidate;
    }

    for (size_t i = 0; i < numCandidate; i++) {
        std::vector<Point> contour = contours[i];
        double epsilon = arcLength(contour, true) * 0.01;
        std::vector<Point> approx;
        approxPolyDP(contour, approx, epsilon, true);
        if (approx.size() < 4) continue;

        // calculate averaged box score
        if (boxFastScore(binary, approx) < polyThresh) continue;

        // unclip
        std::vector<Point> polygon;
        unclip(approx, polygon, 2);
        ploygons.push_back(polygon);
    }
}

double boxFastScore(const Mat & binary, std::vector<Point> & contour)
{
    int rows = binary.rows;
    int cols = binary.cols;

    int xmin = cols - 1;
    int xmax = 0;
    int ymin = rows - 1;
    int ymax = 0;
    for (uint i = 0; i < contour.size(); i++) {
        Point pt = contour[i];
        if (pt.x < xmin) xmin = pt.x;
        if (pt.x > xmax) xmax = pt.x;
        if (pt.y < ymin) ymin = pt.y;
        if (pt.y > ymax) ymax = pt.y;
    }

    if (xmin < 0) xmin = 0;
    if (xmax > cols) xmax = cols - 1;
    if (ymin < 0) ymin = 0;
    if (ymax > rows) ymax = rows - 1;

    Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

    Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    std::vector<std::vector<Point>> roiContours;
    std::vector<Point> roiContour;
    for (uint i = 0; i < contour.size(); i++) {
        Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
        roiContour.push_back(pt);
    }
    roiContours.push_back(roiContour);
    fillPoly(mask, roiContours, Scalar(1));
    double score = mean(binROI, mask).val[0];

    return score;
}


void unclip(std::vector<Point> &inPoly, std::vector<Point> &outPoly, double ratio)
{
    double area = contourArea(inPoly);
    double length = arcLength(inPoly, true);
    double distance = area * ratio / length;

    size_t numPoints = inPoly.size();
    std::vector<std::vector<Point2f>> newLines;
    for (size_t i = 0; i < numPoints; i++) {
        std::vector<Point2f> newLine;
        Point pt1 = inPoly[i];
        Point pt2 = inPoly[(i + 1) % numPoints];
        Point vec = pt2 - pt1;
        float unclipDis = (float)(distance / norm(vec));
        Point2f rotateVec = Point2f(-vec.y * unclipDis, vec.x * unclipDis);
        newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
        newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
        newLines.push_back(newLine);
    }

    size_t numLines = newLines.size();
    for (size_t i = 0; i < numLines; i++) {
        Point2f a = newLines[i][0];
        Point2f b = newLines[i][1];
        Point2f c = newLines[(i + 1) % numLines][0];
        Point2f d = newLines[(i + 1) % numLines][1];
        Point pt;
        Point2f v1 = b - a;
        Point2f v2 = d - c;
        double cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

        if( fabs(cosAngle) > 0.7 ) {
            pt.x = (int)((b.x + c.x) / 2);
            pt.y = (int)((b.y + c.y) / 2);
        } else {
            double denom = a.x * (double)(d.y - c.y) + b.x * (double)(c.y - d.y) +
                           d.x * (double)(b.y - a.y) + c.x * (double)(a.y - b.y);
            double num = a.x * (double)(d.y - c.y) + c.x * (double)(a.y - d.y) + d.x * (double)(c.y - a.y);
            double s = num / denom;

            pt.x = (int)(a.x + s*(b.x - a.x));
            pt.y = (int)(a.y + s*(b.y - a.y));
        }

        outPoly.push_back(pt);
    }
}
