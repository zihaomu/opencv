/*
 * Convolutional Recurrent Neural Network (CRNN):
 * An End-to-End Trainable Neural Network for Image-based SequenceRecognition and Its Application to Scene Text Recognition
 * Copyright (C) 2020,
 * Author List:
 *      Baoguang Shi    <Huazhong University of Science and Technology>
 *      Cong Yao        <Huazhong University of Science and Technology>
 *      Xiang Bai       <Huazhong University of Science and Technology>
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
        "{ modelPath mp                     | | Path to a binary .onnx file contains trained CRNN text recognition model.}";


const std::string vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz";

std::string decode(Mat prediction);

int main(int argc, char** argv)
{
    // Parse arguments
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run the PyTorch implementation (https://github.com/meijieru/crnn.pytorch) of "
                 "An End-to-End Trainable Neural Network for Image-based SequenceRecognition and Its Application to Scene Text Recognition "
                 "(https://arxiv.org/abs/1507.05717)");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelPath = parser.get<String>("modelPath");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load the network
    CV_Assert(!modelPath.empty());
    Net recognizer = readNet(modelPath);

    // Create a window
    static const std::string winName = "Input Cropped Image";
    namedWindow(winName, WINDOW_NORMAL);

    // Open an image file
    CV_Assert(parser.has("inputImage"));
    Mat frame = imread(parser.get<String>("inputImage"), 0);

    // Preprocess
    // Create a 4D blob from a frame
    Mat blob;
    float scale = 1.0 / 255.0;
    Size inputSize = Size(100, 32);
    blob = blobFromImage(frame, scale, inputSize);
    blob -= 0.5;
    blob /= 0.5;

    // Set input blob
    recognizer.setInput(blob);

    // Network Inference
    Mat prediction = recognizer.forward("out");

    // Decode
    std::string result = decode(prediction);

    imshow(winName, frame);
    std::cout << "Result: " << result << std::endl;
    waitKey();

    return 0;
}

std::string decode(Mat prediction)
{
    std::string decodeSeq = "";
    bool ctcFlag = true;
    for (int i = 0; i < prediction.size[0]; i++) {
        int maxLoc = 0;
        float maxScore = prediction.at<float>(i, 0);
        for (uint j = 0; j < vocabulary.length() + 1; j++) {
            float score = prediction.at<float>(i, j);
            if (maxScore < score) {
                maxScore = score;
                maxLoc = j;
            }
        }
        if (maxLoc > 0) {
            char currentChar = vocabulary[maxLoc - 1];
            if (currentChar != decodeSeq.back() || ctcFlag) {
                decodeSeq += currentChar;
                ctcFlag = false;
            }
        } else {
            ctcFlag = true;
        }
    }

    return decodeSeq;
}
