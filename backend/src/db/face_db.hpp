#ifndef FACE_DB_HPP
#define FACE_DB_HPP

#include <vector>
#include <string>
#include <random>
#include <fstream>

struct FaceRecord {
    std::string id;
    std::string name;
    std::vector<float> embedding;
};

class FaceDB {
public:
    FaceDB(const std::string& dbPath = "");
    ~FaceDB();

    void add(const std::string& name, const std::vector<float>& emb);
    std::pair<std::string, float> find(const std::vector<float>& queryEmb, float threshold = 0.6) const;
    bool save(const std::string& path = "") const;
    bool load(const std::string& path = "");
    void clear();

private:
    std::vector<FaceRecord> records;
    std::string filePath;
    std::mt19937 rng;
    
    std::string generateId(const std::string& name);
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const;
};

#endif