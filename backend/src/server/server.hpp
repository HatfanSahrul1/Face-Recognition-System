#ifndef SERVER_HPP
#define SERVER_HPP

#include <cpprest/http_listener.h>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "detector/face_detector.hpp"
#include "embedder/face_embedder.hpp"
#include "db/face_db.hpp"

class FaceRecognitionServer {
public:
    FaceRecognitionServer(const std::string& address);
    ~FaceRecognitionServer();

    void start();
    void stop();

private:
    web::http::experimental::listener::http_listener listener;
    void handleGet(web::http::http_request request);
    void handlePost(web::http::http_request request);
    void handleOptions(web::http::http_request request);
    void processImage(const std::string& base64Image);

    void handleRegister(web::http::http_request request);
    void handleVerify(web::http::http_request request);
    // method dummy
    void registerFace(const std::string& name, const std::string& base64Image);
    void verifyFace(const std::string& base64Image, std::string& outName, float& outConfidence);

    void ResetImages();

    // cache
    std::unique_ptr<FaceDetector> detector_;
    std::unique_ptr<FaceEmbedder> embedder_;
    std::unique_ptr<FaceDB> db_;

    cv::Mat full_image_, cropped_face_image_, spoof_detection_image_;
};

#endif