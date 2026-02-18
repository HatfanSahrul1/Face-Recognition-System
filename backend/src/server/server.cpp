#include "server.hpp"
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

void FaceRecognitionServer::processImage(const std::string& base64Image) {
    // Simpan base64 ke file txt (debug)
    std::ofstream txtfile("/app/data/received_image.txt");
    txtfile << base64Image;
    txtfile.close();

    // Decode base64 ke gambar
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

void FaceRecognitionServer::start() {
    listener.open().wait();
    std::cout << "Server running on http://0.0.0.0:8080" << std::endl;
}

void FaceRecognitionServer::stop() {
    listener.close().wait();
}