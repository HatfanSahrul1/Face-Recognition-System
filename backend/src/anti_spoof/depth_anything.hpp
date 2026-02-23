#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <vector>
#include <string>

class DepthAntiSpoofing
{
public:
    explicit DepthAntiSpoofing(const std::string& modelPath, float flatThreshold = 0.02f);

    bool isSpoof(const cv::Mat& frame, const cv::Rect& faceRect, float& stddevOut);
    bool isSpoof(const cv::Mat& frame, const cv::Rect& faceRect);

    cv::Mat getDepthMap(const cv::Mat& frame, const cv::Rect& faceRect, cv::Size targetSize = cv::Size());

private:
    float flatThreshold_;
    int inputH_, inputW_;  // auto-detect dari model
    Ort::Env env_;
    Ort::Session session_;

    const float mean_[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3]  = {0.229f, 0.224f, 0.225f};

    std::vector<float> preprocess(const cv::Mat& frame);
    cv::Mat runInference(const cv::Mat& frame);
    cv::Mat postprocess(const cv::Mat& depthMap, cv::Size targetSize);
    float getDepthStddev(const cv::Mat& depthMap);
};