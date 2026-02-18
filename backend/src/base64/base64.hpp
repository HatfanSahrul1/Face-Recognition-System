#ifndef BASE64_HPP
#define BASE64_HPP

#include <string>
#include <vector>

class Base64 {
public:
    static std::vector<unsigned char> decode(const std::string& encoded_string);
};

#endif