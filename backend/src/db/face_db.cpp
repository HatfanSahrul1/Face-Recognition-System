#include "face_db.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>

FaceDB::FaceDB(const std::string& dbPath) : filePath(dbPath), rng(std::random_device{}()) {
    if (!filePath.empty()) {
        load(filePath);
    }
}

FaceDB::~FaceDB() {
    if (!filePath.empty()) {
        save(filePath);
    }
}

std::string FaceDB::generateId(const std::string& name) {
    std::uniform_int_distribution<int> dist(0, 15);
    const char* hex = "0123456789abcdef";
    std::stringstream ss;
    ss << name << "_";
    for (int i = 0; i < 8; ++i) ss << hex[dist(rng)];
    return ss.str();
}

void FaceDB::add(const std::string& name, const std::vector<float>& emb) {
    FaceRecord rec;
    rec.id = generateId(name);
    rec.name = name;
    rec.embedding = emb;
    records.push_back(rec);
}

float FaceDB::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const {
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

std::pair<std::string, float> FaceDB::find(const std::vector<float>& queryEmb, float threshold) const {
    float bestSim = -1.0f;
    std::string bestId;
    for (const auto& rec : records) {
        float sim = cosineSimilarity(queryEmb, rec.embedding);
        if (sim > bestSim) {
            bestSim = sim;
            bestId = rec.id;
        }
    }
    if (bestSim >= threshold) return {bestId, bestSim};
    return {"", 0.0f};
}

bool FaceDB::save(const std::string& path) const {
    std::string savePath = path.empty() ? filePath : path;
    if (savePath.empty()) return false;

    std::ofstream file(savePath, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t count = records.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& rec : records) {
        // id
        uint32_t idLen = rec.id.size();
        file.write(reinterpret_cast<const char*>(&idLen), sizeof(idLen));
        file.write(rec.id.c_str(), idLen);

        // name
        uint32_t nameLen = rec.name.size();
        file.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        file.write(rec.name.c_str(), nameLen);

        // embedding
        uint32_t embSize = rec.embedding.size();
        file.write(reinterpret_cast<const char*>(&embSize), sizeof(embSize));
        file.write(reinterpret_cast<const char*>(rec.embedding.data()), embSize * sizeof(float));
    }
    return true;
}

bool FaceDB::load(const std::string& path) {
    std::string loadPath = path.empty() ? filePath : path;
    if (loadPath.empty()) return false;

    std::ifstream file(loadPath, std::ios::binary);
    if (!file.is_open()) return false;

    records.clear();

    uint32_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (uint32_t i = 0; i < count; ++i) {
        FaceRecord rec;

        // id
        uint32_t idLen;
        file.read(reinterpret_cast<char*>(&idLen), sizeof(idLen));
        rec.id.resize(idLen);
        file.read(&rec.id[0], idLen);

        // name
        uint32_t nameLen;
        file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        rec.name.resize(nameLen);
        file.read(&rec.name[0], nameLen);

        // embedding
        uint32_t embSize;
        file.read(reinterpret_cast<char*>(&embSize), sizeof(embSize));
        rec.embedding.resize(embSize);
        file.read(reinterpret_cast<char*>(rec.embedding.data()), embSize * sizeof(float));

        records.push_back(rec);
    }
    return true;
}

void FaceDB::clear() {
    records.clear();
}