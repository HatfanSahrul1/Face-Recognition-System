#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <stdexcept>

class AntiSpoofing
{
public:
    /**
     * @param modelPath  Path ke file mobilenetv2_spoof.onnx
     * @param threshold  Sigmoid score di atas nilai ini = SPOOF (default 0.5)
     */
    explicit AntiSpoofing(const std::string& modelPath, float threshold = 0.5f);

    /**
     * @param faceRoi  Crop wajah dalam format BGR (hasil dari face detector)
     * @return true  = SPOOF
     *         false = REAL
     */
    bool isSpoof(const cv::Mat& faceRoi);

    /**
     * Sama seperti isSpoof() tapi juga mengembalikan raw sigmoid score.
     * score > 0.5 = SPOOF, score <= 0.5 = REAL
     */
    bool isSpoof(const cv::Mat& faceRoi, float& scoreOut);

private:
    cv::dnn::Net net_;
    float threshold_;

    // Preprocessing sesuai training:
    //   - resize ke 224x224
    //   - BGR -> RGB
    //   - normalize [0, 1]
    cv::Mat preprocess(const cv::Mat& faceRoi);
};