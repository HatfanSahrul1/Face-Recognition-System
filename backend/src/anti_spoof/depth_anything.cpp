#include "depth_anything.hpp"

DepthAntiSpoofing::DepthAntiSpoofing(float flatThreshold) 
:   flatThreshold_(flatThreshold),
    env_(ORT_LOGGING_LEVEL_WARNING, "DepthAntiSpoofing"),
    session_(nullptr),
    inputH_(0),
    inputW_(0)
{}

DepthAntiSpoofing::DepthAntiSpoofing(const std::string& modelPath, float flatThreshold)
    : flatThreshold_(flatThreshold),
        env_(ORT_LOGGING_LEVEL_WARNING, "DepthAntiSpoofing"),
        session_(nullptr)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, modelPath.c_str(), opts);

    // Auto-detect input size dari model — tidak perlu hardcode
    auto inputShape = session_.GetInputTypeInfo(0)
                        .GetTensorTypeAndShapeInfo().GetShape();
    // inputShape: [1, 3, H, W]
    inputH_ = static_cast<int>(inputShape[2]);
    inputW_ = static_cast<int>(inputShape[3]);
    printf("[DepthAntiSpoofing] Input size: %dx%d\n", inputW_, inputH_);
}

bool DepthAntiSpoofing::LoadModel(const std::string& modelPath, float flatThreshold)
{
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, modelPath.c_str(), opts);

        // Auto-detect input size from model
        auto inputShape = session_.GetInputTypeInfo(0)
                            .GetTensorTypeAndShapeInfo().GetShape();
        // inputShape: [1, 3, H, W]
        inputH_ = static_cast<int>(inputShape[2]);
        inputW_ = static_cast<int>(inputShape[3]);
        printf("[DepthAntiSpoofing] Model loaded. Input size: %dx%d\n", inputW_, inputH_);
        
        return true;
    }
    catch (const std::exception& e) {
        printf("[DepthAntiSpoofing] Error loading model: %s\n", e.what());
        return false;
    }
}

bool DepthAntiSpoofing::isSpoof(const cv::Mat& frame, const cv::Rect& faceRect, float& stddevOut)
{
    cv::Mat depthMap = runInference(frame);

    // Scale rect dari koordinat frame -> koordinat depth map
    float scaleX = static_cast<float>(inputW_) / frame.cols;
    float scaleY = static_cast<float>(inputH_) / frame.rows;

    cv::Rect scaledRect(
        static_cast<int>(faceRect.x * scaleX),
        static_cast<int>(faceRect.y * scaleY),
        static_cast<int>(faceRect.width  * scaleX),
        static_cast<int>(faceRect.height * scaleY)
    );
    scaledRect &= cv::Rect(0, 0, inputW_, inputH_);

    if (scaledRect.empty()) {
        stddevOut = 0.0f;
        return true;
    }

    cv::Mat faceDepth = depthMap(scaledRect);
    stddevOut = getDepthStddev(faceDepth);

    // Debug imwrite
    cv::Mat debugVis = postprocess(depthMap, cv::Size(frame.cols, frame.rows));
    cv::rectangle(debugVis, faceRect, cv::Scalar(0, 255, 0), 2);
    std::string label = (stddevOut < flatThreshold_ ? "SPOOF" : "REAL");
    label += " std=" + std::to_string(stddevOut);
    cv::putText(debugVis, label, cv::Point(faceRect.x, faceRect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::imwrite("spoof_detect.jpg", debugVis);

    return stddevOut < flatThreshold_;
}

bool DepthAntiSpoofing::isSpoof(const cv::Mat& frame, const cv::Rect& faceRect)
{
    float stddev;
    return isSpoof(frame, faceRect, stddev);
}

cv::Mat DepthAntiSpoofing::getDepthMap(const cv::Mat& frame, const cv::Rect& faceRect, cv::Size targetSize)
{
    cv::Mat depthMap = runInference(frame);
    cv::Size outSize = targetSize.empty() ? cv::Size(frame.cols, frame.rows) : targetSize;
    cv::Mat depthVis = postprocess(depthMap, outSize);

    float scaleX = static_cast<float>(outSize.width)  / frame.cols;
    float scaleY = static_cast<float>(outSize.height) / frame.rows;
    cv::Rect scaledRect(
        static_cast<int>(faceRect.x * scaleX),
        static_cast<int>(faceRect.y * scaleY),
        static_cast<int>(faceRect.width  * scaleX),
        static_cast<int>(faceRect.height * scaleY)
    );
    scaledRect &= cv::Rect(0, 0, outSize.width, outSize.height);
    if (!scaledRect.empty())
        cv::rectangle(depthVis, scaledRect, cv::Scalar(0, 255, 0), 2);

    return depthVis;
}

std::vector<float> DepthAntiSpoofing::preprocess(const cv::Mat& frame)
{
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(inputW_, inputH_), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> blob(1 * 3 * inputH_ * inputW_);
    int planeSize = inputH_ * inputW_;

    for (int y = 0; y < inputH_; ++y) {
        for (int x = 0; x < inputW_; ++x) {
            cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
            int idx = y * inputW_ + x;
            blob[0 * planeSize + idx] = (pixel[0] / 255.0f - mean_[0]) / std_[0];
            blob[1 * planeSize + idx] = (pixel[1] / 255.0f - mean_[1]) / std_[1];
            blob[2 * planeSize + idx] = (pixel[2] / 255.0f - mean_[2]) / std_[2];
        }
    }
    return blob;
}

cv::Mat DepthAntiSpoofing::runInference(const cv::Mat& frame)
{
    std::vector<float> inputData = preprocess(frame);
    std::vector<int64_t> inputShape = {1, 3, inputH_, inputW_};

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
    int outH = static_cast<int>(outputShape[1]);
    int outW = static_cast<int>(outputShape[2]);

    return cv::Mat(outH, outW, CV_32F, outputData).clone();
}

float DepthAntiSpoofing::getDepthStddev(const cv::Mat& depthMap)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(depthMap, mean, stddev);
    return static_cast<float>(stddev[0]);
}

cv::Mat DepthAntiSpoofing::postprocess(const cv::Mat& depthMap, cv::Size targetSize)
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