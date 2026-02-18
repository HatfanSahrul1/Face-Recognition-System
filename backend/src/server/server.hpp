#ifndef SERVER_HPP
#define SERVER_HPP

#include <cpprest/http_listener.h>
#include <string>

class FaceRecognitionServer {
public:
    FaceRecognitionServer(const std::string& address);
    void start();
    void stop();

private:
    web::http::experimental::listener::http_listener listener;
    void handleGet(web::http::http_request request);
    void handlePost(web::http::http_request request);
    void handleOptions(web::http::http_request request);
    void processImage(const std::string& base64Image);
};

#endif