#pragma once
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <fstream>
#include <thread>
#include <iomanip>
#include <iostream>
#include <set>

namespace profiler {

// ============================================================
// Aggregated entry
// ============================================================
struct AggEntry {
    double total_time_us = 0.0;
    uint64_t calls = 0;
    std::set<unsigned long long> thread_ids;
};

// key = "section|layer"
inline std::unordered_map<std::string, AggEntry> g_agg;
inline std::mutex g_mutex;

// ============================================================
// Scoped timer (same macro interface)
// ============================================================
class ScopeTimer {
public:
    ScopeTimer(const std::string& tag, int layer)
        : tag_(tag), layer_(layer),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopeTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start_).count();

        unsigned long long tid =
            std::hash<std::thread::id>{}(std::this_thread::get_id());

        // Compose map key
        std::string key = tag_ + "|" + std::to_string(layer_);

        std::lock_guard<std::mutex> lock(g_mutex);

        auto& agg = g_agg[key];
        agg.total_time_us += us;
        agg.calls += 1;
        agg.thread_ids.insert(tid);
    }

private:
    std::string tag_;
    int layer_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================
// Save final aggregated CSV
// ============================================================
inline void saveCSV(const std::string& filename = "profile_agg.csv") {
    std::lock_guard<std::mutex> lock(g_mutex);

    std::ofstream ofs(filename, std::ios::out);
    ofs << "section,layer,total_time_us,calls,avg_time_us,threads\n";

    for (const auto& it : g_agg) {
        const std::string& key = it.first;
        const AggEntry& e = it.second;

        // Split "section|layer"
        size_t pos = key.find('|');
        std::string section = key.substr(0, pos);
        int layer = std::stoi(key.substr(pos + 1));

        double avg = (e.calls ? e.total_time_us / e.calls : 0.0);

        // Encode thread IDs as a single string
        std::string tid_str;
        for (auto t : e.thread_ids) {
            tid_str += std::to_string(t) + ";";
        }

        ofs << section << ","
            << layer << ","
            << std::fixed << std::setprecision(3) << e.total_time_us << ","
            << e.calls << ","
            << avg << ","
            << tid_str << "\n";
    }

    ofs.close();
    std::cout << "[profiler] saved aggregated results to " << filename << std::endl;
}

} // namespace profiler

// macro stays exactly the same
#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)
//#define PROFILE_SCOPE(tag, layer) profiler::ScopeTimer CONCAT(_profiler_timer_, __COUNTER__)(tag, layer)
#define PROFILE_SCOPE(tag, layer) //does nothing
