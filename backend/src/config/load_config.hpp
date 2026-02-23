#ifndef LOAD_CONFIG_HPP
#define LOAD_CONFIG_HPP

#include <string>
#include <map>
#include <stdexcept>

class Config {
public:
    explicit Config(const std::string& filepath);

    std::string getString(const std::string& key, const std::string& def = "") const;
    float getFloat(const std::string& key, float def = 0.0f) const;
    int getInt(const std::string& key, int def = 0) const;

    bool has(const std::string& key) const;

private:
    std::map<std::string, std::string> data_;
    void parseLine(const std::string& line);
};

#endif