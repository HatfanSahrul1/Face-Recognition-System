#ifndef FACEDETECTOR_HPP
#define FACEDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>

class FaceDetector {
public:
    FaceDetector();
    explicit FaceDetector(const std::string& cascadePath);
    ~FaceDetector() = default;

    bool loadCascade(const std::string& cascadePath);
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    cv::Rect getLargestFace(const cv::Mat& image);
    cv::Mat cropLargestFace(const cv::Mat& image);

private:
    cv::CascadeClassifier faceCascade;
    bool isLoaded;
};

#endif // FACEDETECTOR_HPP