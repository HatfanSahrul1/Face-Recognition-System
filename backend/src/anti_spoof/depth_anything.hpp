#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <vector>
#include <string>

class DepthAntiSpoofing
{
public:
    explicit DepthAntiSpoofing(const std::string& modelPath, float flatThreshold = 0.02f)
        : flatThreshold_(flatThreshold),
          env_(ORT_LOGGING_LEVEL_WARNING, "DepthAntiSpoofing"),
          session_(nullptr)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, modelPath.c_str(), opts);
    }

    bool isSpoof(const cv::Mat& faceRoi)
    {
        float stddev;
        return isSpoof(faceRoi, stddev);
    }

    bool isSpoof(const cv::Mat& faceRoi, float& stddevOut)
    {
        cv::Mat depthMap = runInference(faceRoi);
        stddevOut = getDepthStddev(depthMap);

        cv::imwrite("spoof_detect.jpg", postprocess(depthMap, faceRoi.size()));
        return stddevOut < flatThreshold_;
    }

    cv::Mat getDepthMap(const cv::Mat& faceRoi, cv::Size targetSize = cv::Size())
    {
        cv::Mat depthMap = runInference(faceRoi);
        return postprocess(depthMap, targetSize.empty() ? faceRoi.size() : targetSize);
    }

private:
    float flatThreshold_;
    Ort::Env env_;
    Ort::Session session_;

    const float mean_[3] = {0.485f, 0.456f, 0.406f};  // R, G, B
    const float std_[3]  = {0.229f, 0.224f, 0.225f};

    std::vector<float> preprocess(const cv::Mat& faceRoi)
    {
        cv::Mat resized, rgb;
        cv::resize(faceRoi, resized, cv::Size(322, 322), 0, 0, cv::INTER_CUBIC);
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // Isi buffer NCHW [1,3,322,322] secara manual
        std::vector<float> blob(1 * 3 * 322 * 322);
        int planeSize = 322 * 322;

        for (int y = 0; y < 322; ++y) {
            for (int x = 0; x < 322; ++x) {
                cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
                int idx = y * 322 + x;
                blob[0 * planeSize + idx] = (pixel[0] / 255.0f - mean_[0]) / std_[0];  // R
                blob[1 * planeSize + idx] = (pixel[1] / 255.0f - mean_[1]) / std_[1];  // G
                blob[2 * planeSize + idx] = (pixel[2] / 255.0f - mean_[2]) / std_[2];  // B
            }
        }
        return blob;
    }

    cv::Mat runInference(const cv::Mat& faceRoi)
    {
        std::vector<float> inputData = preprocess(faceRoi);
        std::vector<int64_t> inputShape = {1, 3, 322, 322};

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size()
        );

        const char* inputNames[]  = {"input"};
        const char* outputNames[] = {"depth"};

        auto outputTensors = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1
        );

        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape  = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Output shape: [1, 322, 322] — bukan [1, 1, H, W]
        int outH = static_cast<int>(outputShape[1]);
        int outW = static_cast<int>(outputShape[2]);

        return cv::Mat(outH, outW, CV_32F, outputData).clone();
    }

    float getDepthStddev(const cv::Mat& depthMap)
    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(depthMap, mean, stddev);
        return static_cast<float>(stddev[0]);
    }

    cv::Mat postprocess(const cv::Mat& depthMap, cv::Size targetSize)
    {
        double minVal, maxVal;
        cv::minMaxLoc(depthMap, &minVal, &maxVal);

        cv::Mat depthVis;
        depthMap.convertTo(depthVis, CV_8U,
            255.0 / (maxVal - minVal),
            -minVal * 255.0 / (maxVal - minVal));

        cv::Mat depthColored;
        cv::applyColorMap(depthVis, depthColored, cv::COLORMAP_MAGMA);

        if (targetSize != depthColored.size())
            cv::resize(depthColored, depthColored, targetSize, 0, 0, cv::INTER_LINEAR);

        return depthColored;
    }
};