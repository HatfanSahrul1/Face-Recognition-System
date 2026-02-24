#include "server/server.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    FaceRecognitionServer server("http://0.0.0.0:8080");
    try {
        server.start();
        std::cout << "Server running. Press Ctrl+C to stop." << std::endl;
        // Loop forever
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        // server.stop();
    } catch (std::exception const & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}