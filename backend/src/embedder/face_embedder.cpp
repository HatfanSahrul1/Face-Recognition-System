#include "face_embedder.hpp"
#include <iostream>
#include <cmath>

FaceEmbedder::FaceEmbedder() 
    : isLoaded(false), 
      inputSize(112, 112), 
      mean(127.5, 127.5, 127.5), 
      std(127.5, 127.5, 127.5) {}

FaceEmbedder::FaceEmbedder(const std::string& modelPath) 
    : isLoaded(false), 
      inputSize(112, 112), 
      mean(127.5, 127.5, 127.5), 
      std(127.5, 127.5, 127.5) {
    loadModel(modelPath);
}

bool FaceEmbedder::loadModel(const std::string& modelPath) {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        
        // Gunakan CPU (bisa ganti ke CUDA jika ada)
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        isLoaded = true;
        std::cout << "Model loaded: " << modelPath << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        isLoaded = false;
    }
    return isLoaded;
}

cv::Mat FaceEmbedder::preprocess(const cv::Mat& faceImage) {
    cv::Mat resized, blob;
    cv::resize(faceImage, resized, inputSize);
    
    // Konversi ke blob dengan normalisasi (mean=127.5, std=127.5) [citation:2]
    blob = cv::dnn::blobFromImage(resized, 1.0, inputSize, mean, true, false);
    
    // Manual per-channel normalization jika diperlukan
    // Karena blobFromImage hanya melakukan scale=1.0, kita perlu normalisasi manual
    blob /= 127.5;  // scale ke [0,2]
    blob -= 1.0;    // shift ke [-1,1]
    
    return blob;
}

void FaceEmbedder::l2Normalize(std::vector<float>& embedding) {
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-6) {
        for (float& val : embedding) {
            val /= norm;
        }
    }
}

std::vector<float> FaceEmbedder::getEmbedding(const cv::Mat& faceImage) {
    std::vector<float> embedding;
    if (!isLoaded || faceImage.empty()) {
        return embedding;
    }
    
    try {
        cv::Mat blob = preprocess(faceImage);
        net.setInput(blob);
        cv::Mat output = net.forward();
        
        // Convert ke vector float (dimensi 512) [citation:2]
        embedding.assign((float*)output.data, (float*)output.data + output.total());
    } catch (const cv::Exception& e) {
        std::cerr << "Embedding error: " << e.what() << std::endl;
    }
    
    return embedding;
}

std::vector<float> FaceEmbedder::getNormalizedEmbedding(const cv::Mat& faceImage) {
    std::vector<float> embedding = getEmbedding(faceImage);
    if (!embedding.empty()) {
        l2Normalize(embedding);
    }
    return embedding;
}