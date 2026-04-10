#if defined(_WIN32)
  #include <windows.h>
  #include <psapi.h>
#elif defined(__APPLE__)
  #include <mach/mach.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include "../EncLib/CABACEncoder.h"  // your header
#include "../EncLib/BinEncoder_simple.h"
#include "../CommonLib/ContextModel.h"
#include "../../StaticCoder.h" 
#include "../CommonLib/TypeDef.h"
#include "../DecLib/CABACDecoder.h"

#include <string>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <map>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>



//#define TENSOR_BIN_DIR "models/bert_tensors_binaries/"
//#define META_FILE "models/bert_tensors.meta"
//#define TENSOR_BIN_DIR "models/gpt_tensors_binaries/"
//#define META_FILE "models/gpt_tensors.meta"
#define TENSOR_BIN_DIR "models/efficientnet_b7/binaries/"
#define META_FILE "models/efficientnet_b7/tensors.meta"

#define MODEL_NAME "efficientnet_b7"

// ============================================================
// Peak Memory Sampler — mirrors Python psutil RSS sampling
// ============================================================

// Cross-platform RSS query (bytes)
static size_t getCurrentRSS()
{
#if defined(_WIN32)
    // Windows: use GetProcessMemoryInfo
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<size_t>(pmc.WorkingSetSize);
    return 0;

#elif defined(__linux__)
    // Linux: read /proc/self/status VmRSS
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
        {
            std::istringstream iss(line);
            std::string key;
            size_t kb;
            iss >> key >> kb;
            return kb * 1024;
        }
    }
    return 0;

#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        return static_cast<size_t>(info.resident_size);
    return 0;
#else
    return 0;
#endif
}

struct MemoryStats
{
    size_t baselineBytes = 0;   // RSS before the call
    size_t peakBytes     = 0;   // peak RSS during the call
    size_t deltaBytes    = 0;   // peak - baseline
};

class PeakMemorySampler
{
public:
    // interval_ms: sampling interval (default 10 ms, matches Python)
    explicit PeakMemorySampler(int interval_ms = 10)
        : m_interval(interval_ms), m_done(false), m_peak(0) {}

    // Call just before the workload
    void start()
    {
        m_done = false;
        m_peak = getCurrentRSS();
        m_thread = std::thread([this]()
        {
            while (!m_done.load(std::memory_order_relaxed))
            {
                size_t rss = getCurrentRSS();
                size_t cur = m_peak.load(std::memory_order_relaxed);
                while (rss > cur &&
                       !m_peak.compare_exchange_weak(cur, rss,
                           std::memory_order_relaxed))
                {}
                std::this_thread::sleep_for(m_interval);
            }
        });
    }

    // Call just after the workload; returns filled MemoryStats
    MemoryStats stop(size_t baseline)
    {
        m_done.store(true, std::memory_order_relaxed);
        if (m_thread.joinable())
            m_thread.join();

        MemoryStats s;
        s.baselineBytes = baseline;
        s.peakBytes     = m_peak.load();
        s.deltaBytes    = (s.peakBytes > s.baselineBytes)
                              ? s.peakBytes - s.baselineBytes
                              : 0;
        return s;
    }

private:
    std::chrono::milliseconds  m_interval;
    std::atomic<bool>          m_done;
    std::atomic<size_t>        m_peak;
    std::thread                m_thread;
};

// Convenience: print a MemoryStats block with a label
static void printMemStats(const std::string& label, const MemoryStats& ms)
{
    auto toMB = [](size_t b){ return b / (1024.0 * 1024.0); };
    std::cout << label << "\n"
              << "  baseline : " << toMB(ms.baselineBytes) << " MB\n"
              << "  peak     : " << toMB(ms.peakBytes)     << " MB\n"
              << "  delta    : " << toMB(ms.deltaBytes)     << " MB\n";
}

// ============================================================




struct CodingStats
{
    uint64_t weights = 0;
    uint64_t rawBits = 0;
};

std::vector<int32_t> read_tensor_bin(const std::string &path)
{
    std::ifstream infile(path, std::ios::binary | std::ios::ate);
    if(!infile.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return {};
    }

    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if(size % sizeof(int32_t) != 0)
    {
        std::cerr << "File size is not a multiple of int32_t: " << size << std::endl;
        return {};
    }

    std::vector<int32_t> buffer(size / sizeof(int32_t));
    if(!infile.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        std::cerr << "Error reading file: " << path << std::endl;
        return {};
    }

    return buffer;
}

// ------------------------------------------------------------
// Utility: load metadata + tensors
// ------------------------------------------------------------
bool loadModelTensors(std::vector<TensorMeta>& tensors)
{
    std::ifstream meta(META_FILE);

    if(!meta)
    {
        std::cout << "Failed to open metadata file\n";
        return false;
    }

    std::string tag;
    uint32_t numTensors;
    float qstep;

    meta >> tag >> numTensors;

    std::cout << "Loading model tensors: " << numTensors << "\n";

    tensors.resize(numTensors);

    for(uint32_t i = 0; i < numTensors; i++)
    {
        TensorMeta& t = tensors[i];

        meta >> t.tensorId;
        meta >> t.name;
        t.name.erase(std::remove(t.name.begin(), t.name.end(), '\r'), t.name.end());

        std::string typeStr;
        meta >> typeStr;

        if(typeStr == "Weight") t.tensorType = TensorType::Weight;
        else if(typeStr == "Bias") t.tensorType = TensorType::Bias;
        else t.tensorType = TensorType::Bias; // in case it is neither, better consider a sensitive tensor 'bias' type

        uint32_t bw;
        meta >> bw;

        t.tensorBitwidth = bitwidthFromLiteral(bw);

        meta >> t.numDims;

        t.shape.resize(t.numDims);

        for(uint32_t d = 0; d < t.numDims; d++)
            meta >> t.shape[d];

        meta >> qstep;

        // load bin tensor
        std::string binPath = std::string(TENSOR_BIN_DIR) + t.name + ".bin";

        //std::cout << "Trying to open: " << binPath << std::endl;
        if(!std::filesystem::exists(binPath))
        {
            std::cout << "File does not exist!\n";
        }

        t.data = read_tensor_bin(binPath);

        if(t.data.empty())
        {
            std::cout << "Failed loading tensor data\n";
            return false;
        }
    }

    return true;
}

// ------------------------------------------------------------
// Utility: compute tensor element count
// ------------------------------------------------------------
uint64_t numel(const TensorMeta& t)
{
    uint64_t n = 1;

    for(auto s : t.shape)
        n *= s;

    return n;
}



// ------------------------------------------------------------
// Utility: tensor type parser
// ------------------------------------------------------------
TensorType parseTensorType(const std::string& name)
{
    if(name.find("bias") != std::string::npos)
        return TensorType::Bias;

    if(name.find("norm") != std::string::npos)
        return TensorType::Bias;

    return TensorType::Weight;
}


// Save decoded TensorMeta in the same format it was encoded (.bin tensors + metadata)

void saveDecodedModel(const std::vector<TensorMeta>& model,
                      const std::string& dir)
{
    std::filesystem::create_directory(dir);

    std::ofstream meta(dir + "/decoded_tensors.meta");

    meta << "numTensors " << model.size() << "\n\n";

    for(const auto& t : model)
    {
        // generate filename from tensorId
        std::string filename = "tensor_" + std::to_string(t.tensorId) + ".bin";
        std::string path = dir + "/" + filename;

        std::ofstream out(path, std::ios::binary);

        if (!out)
        {
            std::cerr << "Failed to write: " << path << "\n";
            continue;
        }

        out.write(reinterpret_cast<const char*>(t.data.data()),
                  t.data.size() * sizeof(int32_t));

        out.close();

        meta << t.tensorId << " "
             << filename << " "
             << static_cast<int>(t.tensorType) << " "
             << static_cast<int>(t.tensorBitwidth) << " "
             << t.numDims << " ";

        for(auto s : t.shape)
            meta << s << " ";

        meta << "\n";
    }
    meta.close();

    std::cout << "Decoded tensors saved to: " << dir << "\n";
}

// ------------------------------------------------------------
// Validation
// ------------------------------------------------------------
void validateModel(
    const std::vector<TensorMeta>& original,
    const std::vector<TensorMeta>& decoded)
{
    uint64_t totalMismatch = 0;
    uint64_t totalWeights  = 0;

    for(size_t t = 0; t < original.size(); t++)
    {
        const auto& A = original[t];
        const auto& B = decoded[t];

        uint64_t mism = 0;

        for(size_t i = 0; i < A.data.size(); i++)
        {
            if(A.data[i] != B.data[i]){
                mism++;
                printf("mismatch: original: %d | decoded: %d \n", A.data[i], B.data[i]);
            }
                
        }

        totalMismatch += mism;
        totalWeights  += A.data.size();

       if(mism > 0)
        {
            std::cout << "Tensor mismatch: " << A.name
                      << " mismatches=" << mism << "\n";
        }
    }

    std::cout << "\n===== Validation =====\n";

    std::cout << "Total weights: " << totalWeights << "\n";
    std::cout << "Total mismatches: " << totalMismatch << "\n";

    if(totalMismatch == 0)
        std::cout << "Perfect reconstruction\n";
    else
        std::cout << "Reconstruction errors detected\n";
}





// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{

    std::map<TensorBitwidth, CodingStats> stats;

    std::vector<TensorMeta> modelTensors;

    if(!loadModelTensors(modelTensors))
        return -1;

    std::cout << "Loaded tensors successfully\n";

    for(const auto& t : modelTensors)
    {
        uint32_t bw = getBitwidthFromEnum(t.tensorBitwidth);

        CodingStats& s = stats[t.tensorBitwidth];

        s.weights += t.data.size();
        s.rawBits += (uint64_t)t.data.size() * bw;
    }


    // --------------------------------------------------
    // ENCODING
    // --------------------------------------------------

    Encoder encoder;

    uint32_t numGtxFlags = 4;

    std::cout << "\n=== Encoding Model ===\n";

    PeakMemorySampler encSampler;
    size_t baselineEncMem = getCurrentRSS();          // baseline before encode

    encSampler.start();

    auto encStart = std::chrono::high_resolution_clock::now();

    encoder.initCtxModels(numGtxFlags);
    const std::vector<uint8_t>& bytestream =
        encoder.encodeModel(modelTensors);

    auto encEnd = std::chrono::high_resolution_clock::now();

    MemoryStats encMemStats = encSampler.stop(baselineEncMem);

    double encTime = std::chrono::duration<double>(encEnd - encStart).count();

    std::cout << "Compressed size: "
              << bytestream.size()
              << " bytes\n";


    uint64_t totalRawBits = 0;

    for(auto& [bw, s] : stats)
        totalRawBits += s.rawBits;

    uint64_t compressedBits = bytestream.size() * 8;

    std::ofstream f("efficientnet_b7_bitstream.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytestream.data()),
        bytestream.size());

    std::cout << "\n===== Bitwidth Statistics =====\n";

    for(auto& [bw, s] : stats)
    {
        uint64_t compBits =
            (double)s.rawBits / totalRawBits * compressedBits;

        double bitsPerWeight =
            (double)compBits / s.weights;

        std::cout
            << "Bitwidth "
            << getBitwidthFromEnum(bw)
            << "-bit\n";

        std::cout
            << "  weights: "
            << s.weights << "\n";

        std::cout
            << "  bits/weight: "
            << bitsPerWeight << "\n";
    }

    int num_tensors = modelTensors.size();
    uint64_t originalBytes = 0;

    for(const auto& t : modelTensors)
    {
        uint32_t bw = getBitwidthFromEnum(t.tensorBitwidth);

        originalBytes +=
            (uint64_t)t.data.size() * bw / 8;
    }


        // --------------------------------------------------
    // Free model tensors before decoding — decoder only needs the bytestream
    // --------------------------------------------------
/*     std::cout << "\n=== Freeing model tensors before decode ===\n";
    size_t beforeFree = getCurrentRSS();

    for (auto& t : modelTensors)
    {
        std::vector<int32_t>().swap(t.data);  // force-free each tensor's heap data
    }
    modelTensors.clear();
    modelTensors.shrink_to_fit();

    size_t afterFree = getCurrentRSS();
    auto toMB = [](size_t b){ return b / (1024.0 * 1024.0); };
    std::cout << "Memory before free : " << toMB(beforeFree) << " MB\n";
    std::cout << "Memory after free  : " << toMB(afterFree)  << " MB\n";
    std::cout << "Freed              : " << toMB(beforeFree - afterFree) << " MB\n";
 */

    // --------------------------------------------------
    // DECODING
    // --------------------------------------------------

    std::cout << "\n=== Decoding Model ===\n";

    Decoder decoder;
    std::vector<TensorMeta> decodedModel;

    PeakMemorySampler decSampler;
    size_t baselineDecMem = getCurrentRSS();          // baseline before decode

    decSampler.start();
    auto decStart = std::chrono::high_resolution_clock::now();
    
    decoder.setStream(const_cast<std::vector<uint8_t>&>(bytestream));

    decoder.initCtxModels(numGtxFlags);
    printf("start decoding...\n");
    decoder.decodeModel(decodedModel);
   // decoder.finishDecoding();

    auto decEnd = std::chrono::high_resolution_clock::now();
    MemoryStats decMemStats = decSampler.stop(baselineDecMem);

    double decTime = std::chrono::duration<double>(decEnd-decStart).count();

    std::cout << "Decoded tensors: "
              << decodedModel.size()
              << "\n";

   

    // --------------------------------------------------
    // VALIDATION
    // --------------------------------------------------

    validateModel(modelTensors, decodedModel);


    /// save decoded tensormeta
    saveDecodedModel(decodedModel, ("efficientnet_b7_decoded"));


    // --------------------------------------------------
    // SUMMARY
    // --------------------------------------------------

    

    uint64_t compressedBytes = bytestream.size();

    double ratio =
        (double)originalBytes / compressedBytes;

    double encodeMB =
    (double)originalBytes / (1024.0*1024.0);

    auto toMB = [](size_t b){ return b / (1024.0 * 1024.0); };
    std::cout << "\n========== MODEL CODING SUMMARY ==========\n"
              << "Tensors processed  : " << num_tensors        << "\n"
              << "Original size      : " << encodeMB                    << " MB\n"
              << "Compressed size    : " << compressedBytes/(1024.0*1024.0) << " MB\n"
              << "Compression ratio  : " << ratio                       << "\n"
              << "\nEncoding time      : " << encTime                   << " sec\n"
              << "Decoding time      : " << decTime                     << " sec\n"
              << "Encode speed       : " << encodeMB / encTime          << " MB/s\n"
              << "Decode speed       : " << encodeMB / decTime          << " MB/s\n";

    std::cout << "\n========== MEMORY USAGE ==========\n";
    printMemStats("Encode memory:", encMemStats);
    printMemStats("Decode memory:", decMemStats);

    // Structured summary matching the Python dict fields
    //auto toMB = [](size_t b){ return b / (1024.0 * 1024.0); };
    std::cout << "\n  peak_enc_mem   : " << toMB(encMemStats.peakBytes)     << " MB\n"
              << "  peak_dec_mem   : " << toMB(decMemStats.peakBytes)     << " MB\n"
              << "  peak_delta_enc : " << toMB(encMemStats.deltaBytes)    << " MB\n"
              << "  peak_delta_dec : " << toMB(decMemStats.deltaBytes)    << " MB\n"
              << "==========================================\n";

    return 0;
}
