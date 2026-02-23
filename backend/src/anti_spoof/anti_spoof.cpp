#include "anti_spoof.hpp"
    
AntiSpoofing::AntiSpoofing(const std::string& modelPath, float threshold)
    : threshold_(threshold)
{
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty())
        throw std::runtime_error("AntiSpoofing: gagal load model -> " + modelPath);
}


bool AntiSpoofing::isSpoof(const cv::Mat& faceRoi)
{
    float score;
    return isSpoof(faceRoi, score);
}


bool AntiSpoofing::isSpoof(const cv::Mat& faceRoi, float& scoreOut)
{
    cv::Mat blob = preprocess(faceRoi);
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    scoreOut = output.at<float>(0, 0);
    return scoreOut > threshold_;
}


cv::Mat AntiSpoofing::preprocess(const cv::Mat& faceRoi)
{
    cv::Mat resized, rgb, floatImg;

    // Resize ke 224x224
    cv::resize(faceRoi, resized, cv::Size(224, 224));

    // BGR -> RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Normalize [0, 1]
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // blobFromImage: HWC -> NCHW [1, 3, 224, 224]
    // swapRB=false karena sudah convert manual
    return cv::dnn::blobFromImage(
        floatImg,
        1.0,
        cv::Size(224, 224),
        cv::Scalar(0, 0, 0),
        false,   // swapRB
        false    // crop
    );
}
