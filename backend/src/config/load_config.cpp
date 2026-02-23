#include "load_config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

Config::Config(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filepath);
    }

    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        // Remove comments (start with #)
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos)
        {
            line = line.substr(0, commentPos);
        }
        
        // Trim whitespace (optional)
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        parseLine(line);
    }
}

void Config::parseLine(const std::string& line) {
    size_t eqPos = line.find('=');
    if (eqPos == std::string::npos) {
        std::cerr << "Warning: ignoring malformed line (no '='): " << line << std::endl;
        return;
    }

    std::string key = line.substr(0, eqPos);
    std::string value = line.substr(eqPos + 1);

    // Trim key and value
    key.erase(0, key.find_first_not_of(" \t\r\n"));
    key.erase(key.find_last_not_of(" \t\r\n") + 1);
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);

    if (key.empty()) {
        std::cerr << "Warning: empty key, ignoring" << std::endl;
        return;
    }

    data_[key] = value;
}

std::string Config::getString(const std::string& key, const std::string& def) const {
    auto it = data_.find(key);
    if (it != data_.end())
        return it->second;
    return def;
}

float Config::getFloat(const std::string& key, float def) const {
    auto it = data_.find(key);
    if (it != data_.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            std::cerr << "Warning: cannot parse float for key '" << key << "', using default" << std::endl;
        }
    }
    return def;
}

int Config::getInt(const std::string& key, int def) const {
    auto it = data_.find(key);
    if (it != data_.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            std::cerr << "Warning: cannot parse int for key '" << key << "', using default" << std::endl;
        }
    }
    return def;
}

bool Config::has(const std::string& key) const {
    return data_.find(key) != data_.end();
}