#include "face_detector.hpp"
#include <iostream>

FaceDetector::FaceDetector() : isLoaded(false) {}

FaceDetector::FaceDetector(const std::string& cascadePath) {
    loadCascade(cascadePath);
}

bool FaceDetector::loadCascade(const std::string& cascadePath) {
    isLoaded = faceCascade.load(cascadePath);
    if (!isLoaded) {
        std::cerr << "Error loading cascade classifier from: " << cascadePath << std::endl;
    }
    return isLoaded;
}

std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& image){
    std::vector<cv::Rect> faces;
    if (!isLoaded || image.empty()) {
        return faces;
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Deteksi wajah dengan parameter default
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    return faces;
}

cv::Rect FaceDetector::getLargestFace(const cv::Mat& image){
    auto faces = detectFaces(image);
    if (faces.empty()) {
        return cv::Rect();
    }

    // Cari wajah terbesar (berdasarkan area)
    auto largest = std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });
    return *largest;
}

cv::Mat FaceDetector::cropLargestFace(const cv::Mat& image){
    cv::Rect faceRect = getLargestFace(image);
    if (faceRect.empty()) {
        return cv::Mat();
    }
    return image(faceRect).clone();
}

void FaceDetector::cropFace(const cv::Mat& image, cv::Mat& outCropped, cv::Mat& outSpoofness) {
    cv::Rect faceRect = getLargestFace(image);
    if (faceRect.empty()) {
        return;
    }

    outCropped = image(faceRect).clone();

    int dw = faceRect.width * 0.25;
    int dh = faceRect.height * 0.25;

    cv::Rect spoofRect;
    spoofRect.x = std::max(0, faceRect.x - dw / 2);
    spoofRect.y = std::max(0, faceRect.y - dh / 2);
    spoofRect.width = faceRect.width + dw;
    spoofRect.height = faceRect.height + dh;

    spoofRect = spoofRect & cv::Rect(0, 0, image.cols, image.rows);

    outSpoofness = image(spoofRect).clone();
}