#include "server/server.hpp"
#include <iostream>

int main() {
    FaceRecognitionServer server("http://0.0.0.0:8080");
    try {
        server.start();
        std::cout << "Press Enter to stop..." << std::endl;
        std::cin.get();
        server.stop();
    } catch (std::exception const & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}