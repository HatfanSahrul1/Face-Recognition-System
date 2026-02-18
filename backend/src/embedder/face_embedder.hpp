#ifndef FACE_EMBEDDER_HPP
#define FACE_EMBEDDER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>

class FaceEmbedder {
public:
    FaceEmbedder();
    explicit FaceEmbedder(const std::string& modelPath);
    ~FaceEmbedder() = default;

    bool loadModel(const std::string& modelPath);
    std::vector<float> getEmbedding(const cv::Mat& faceImage);
    std::vector<float> getNormalizedEmbedding(const cv::Mat& faceImage);
    
    int getEmbeddingSize() const { return 512; }

private:
    cv::dnn::Net net;
    bool isLoaded;
    cv::Size inputSize;
    cv::Scalar mean;
    cv::Scalar std;
    
    cv::Mat preprocess(const cv::Mat& faceImage);
    void l2Normalize(std::vector<float>& embedding);
};

#endif