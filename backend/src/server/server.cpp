#include "server/server.hpp"
#include "base64/base64.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

FaceRecognitionServer::FaceRecognitionServer(const std::string& address) 
    : listener(address) {
    listener.support(methods::GET, std::bind(&FaceRecognitionServer::handleGet, this, std::placeholders::_1));
    listener.support(methods::POST, std::bind(&FaceRecognitionServer::handlePost, this, std::placeholders::_1));
    listener.support(methods::OPTIONS, std::bind(&FaceRecognitionServer::handleOptions, this, std::placeholders::_1));
}

void FaceRecognitionServer::handleGet(http_request request) {
    auto path = request.request_uri().path();
    if (path == U("/health")) {
        http_response response(status_codes::OK);
        response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
        response.set_body(U("Backend is running"));
        request.reply(response);
    } else {
        request.reply(status_codes::NotFound);
    }
}

void FaceRecognitionServer::handlePost(http_request request) {
    auto path = request.request_uri().path();
    if (path == U("/test")) {
        request.extract_json().then([this, request](json::value body) {
            auto imageBase64 = body.at(U("image")).as_string();
            this->processImage(imageBase64);
            
            http_response response(status_codes::OK);
            response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
            json::value resp;
            resp[U("status")] = json::value::string(U("received"));
            resp[U("image_size")] = json::value::number(imageBase64.size());
            response.set_body(resp);
            request.reply(response);
        }).wait();
    } else if (path == U("/register")) {
        handleRegister(request);
    } else if (path == U("/verify")) {
        handleVerify(request);
    } else {
        request.reply(status_codes::NotFound);
    }
}

void FaceRecognitionServer::handleOptions(http_request request) {
    http_response response(status_codes::OK);
    response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
    response.headers().add(U("Access-Control-Allow-Methods"), U("GET, POST, OPTIONS"));
    response.headers().add(U("Access-Control-Allow-Headers"), U("Content-Type"));
    request.reply(response);
}

void FaceRecognitionServer::handleRegister(http_request request) {
    request.extract_json().then([this, request](json::value body) {
        auto name = body.at(U("name")).as_string();
        auto imageBase64 = body.at(U("image")).as_string();
        
        this->registerFace(name, imageBase64);
        
        http_response response(status_codes::OK);
        response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
        json::value resp;
        resp[U("status")] = json::value::string(U("registered"));
        resp[U("name")] = json::value::string(name);
        response.set_body(resp);
        request.reply(response);
    }).wait();
}

void FaceRecognitionServer::handleVerify(http_request request) {
    request.extract_json().then([this, request](json::value body) {
        auto imageBase64 = body.at(U("image")).as_string();
        
        std::string name;
        float confidence;
        this->verifyFace(imageBase64, name, confidence);
        
        http_response response(status_codes::OK);
        response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
        json::value resp;
        resp[U("status")] = json::value::string(U("verified"));
        resp[U("name")] = json::value::string(name);
        resp[U("confidence")] = json::value::number(confidence);
        response.set_body(resp);
        request.reply(response);
    }).wait();
}

void FaceRecognitionServer::processImage(const std::string& base64Image) {
    std::ofstream txtfile("/app/data/received_image.txt");
    txtfile << base64Image;
    txtfile.close();

    try {
        std::vector<unsigned char> decoded = Base64::decode(base64Image);
        cv::Mat img = cv::imdecode(decoded, cv::IMREAD_COLOR);
        if (!img.empty()) {
            cv::imwrite("/app/data/received_image.jpg", img);
            std::cout << "Gambar tersimpan: /app/data/received_image.jpg" << std::endl;
        } else {
            std::cout << "Gambar kosong setelah decode" << std::endl;
        }
    } catch (std::exception const& e) {
        std::cerr << "Decode error: " << e.what() << std::endl;
    }
}

void FaceRecognitionServer::registerFace(const std::string& name, const std::string& base64Image) {
    std::cout << "Register face for: " << name << std::endl;
    // Simpan gambar untuk dummy
    try {
        std::vector<unsigned char> decoded = Base64::decode(base64Image);
        cv::Mat img = cv::imdecode(decoded, cv::IMREAD_COLOR);
        if (!img.empty()) {
            cv::imwrite("/app/data/register_" + name + ".jpg", img);
            std::cout << "Saved: /app/data/register_" << name << ".jpg" << std::endl;
        }
    } catch(...) {}
}

void FaceRecognitionServer::verifyFace(const std::string& base64Image, std::string& outName, float& outConfidence) {
    std::cout << "Verify face" << std::endl;
    // Dummy: selalu kenali sebagai "John Doe" dengan confidence 0.95
    outName = "John Doe";
    outConfidence = 0.95f;
}

void FaceRecognitionServer::start() {
    listener.open().wait();
    std::cout << "Server running on http://0.0.0.0:8080" << std::endl;
}

void FaceRecognitionServer::stop() {
    listener.close().wait();
}