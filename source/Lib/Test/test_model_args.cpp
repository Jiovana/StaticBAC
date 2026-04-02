/*
===============================================================================
StaticBAC Model Encoder / Decoder
===============================================================================

Description:
------------
This executable encodes and decodes neural network tensors using the
StaticBAC entropy coding framework. It reads pre-quantized tensor data
(.bin files) along with metadata, compresses them into a single bitstream,
and optionally reconstructs (decodes) them back for validation.

The tool also reports:
- Compression ratio
- Bit-per-weight statistics
- Encoding / decoding speed
- Peak memory usage during encoding and decoding

-------------------------------------------------------------------------------
Input Requirements for encoding:
-------------------------------------------------------------------------------
1. Tensor binaries directory (--binaries)
   - Contains one .bin file per tensor
   - Each file must store int32_t values
   - Filename must match tensor name from metadata:
        <tensor_name>.bin

2. Metadata file (--meta)
   - Text file describing all tensors
   - Format:

        numTensors <N>

        <tensorId> <name> <type> <bitwidth> <numDims> <shape...> <qstep>

   - Example:
        0 conv1.weight Weight 8 4 64 3 7 7 0.02

3. Model name (--name)
   - Used to generate output filenames

-------------------------------------------------------------------------------
Usage:
-------------------------------------------------------------------------------

    ./staticBac
        --binaries <tensor_bin_dir>
        --meta <meta_file>
        --name <model_name>
        [--scaling]

-------------------------------------------------------------------------------
Arguments:
-------------------------------------------------------------------------------

--binaries <path>
    Path to directory containing tensor .bin files

--meta <file>
    Path to metadata file describing tensors

--name <string>
    Model name used for output files

--scaling (optional flag)
    Enables scaling-aware encoding
    Default: disabled (false)

--encode (optional flag)
    Enables encoding-only run
    Default: false

--decode (optional flag)
    Enables decoding-only run
    Default: false
    
If no explicity --encode and/or --decode, performs full encoding + decoding

--bitstream <file> (for decoder only run)
    Path to compressed bitstream file


-------------------------------------------------------------------------------
Outputs:
-------------------------------------------------------------------------------

1. Compressed bitstream:
        <model_name>[_scaled|_noscale].bin

2. Decoded tensors:
        <model_name>[_scaled|_noscale]_decoded/
            ├── tensor_0.bin
            ├── tensor_1.bin
            └── decoded_tensors.meta

3. Console statistics:
    - Compression ratio
    - Bits per weight per bitwidth group
    - Encoding / decoding time
    - Throughput (MB/s)
    - Peak memory usage

-------------------------------------------------------------------------------
Notes:
-------------------------------------------------------------------------------
- All tensors are expected to be already quantized and stored as int32.
- Bitwidth information is used only for entropy coding efficiency.
- Scaling mode (--scaling) changes encoder behavior (e.g., normalization).
- Buffers (if included in metadata) should NOT be quantized and must already
  be stored as int32.

-------------------------------------------------------------------------------
Example:
-------------------------------------------------------------------------------

    ./staticBac \
        --binaries ./tensors/ \
        --meta model.meta \
        --name resnet50 \
        --scaling

-------------------------------------------------------------------------------
Author:
-------------------------------------------------------------------------------
Jiovana Sousa Gomes <gomesjiovana@gmail.com> 
Based on DeepCABAC and NNCodec https://github.com/d-becking/nncodec2
===============================================================================
*/



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
#include <unordered_map>


std::string g_tensorBinDir;
std::string g_metaFile;
std::string g_modelName;
bool g_doEncode = false;
bool g_doDecode = false;
std::string g_bitstreamFile;

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
    std::ifstream meta(g_metaFile);

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
        std::string binPath = g_tensorBinDir + t.name + ".bin";

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


bool parseArgs(int argc, char* argv[])
{
    std::unordered_map<std::string, std::string> args;

    for(int i = 1; i < argc; i++)
    {
        std::string key = argv[i];

       
        if(key == "--encode")
        {
            g_doEncode = true;
        }
        else if(key == "--decode")
        {
            g_doDecode = true;
        }
        
        else if(key.rfind("--", 0) == 0) // starts with --
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Missing value for " << key << "\n";
                return false;
            }

            args[key] = argv[i + 1];
            i++;
        }
        else
        {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }

    // Default: if neither specified → do both
    if(!g_doEncode && !g_doDecode)
    {
        g_doEncode = true;
        g_doDecode = true;
    }

    // Required arguments
    if(g_doEncode)
    {
         if(args.count("--binaries")) g_tensorBinDir = args["--binaries"];
        else { std::cerr << "--binaries location required\n"; return false; }

        if(args.count("--meta")) g_metaFile = args["--meta"];
        else { std::cerr << "--meta file required\n"; return false; }
    }

    if(g_doDecode)
    {
        if(!g_doEncode && !args.count("--bitstream"))
        {
            std::cerr << "Decode-only mode requires --bitstream\n";
            return false;
        }else { g_bitstreamFile = args["--bitstream"]; }
    }
   

    if(args.count("--name")) g_modelName = args["--name"];
    else { std::cerr << "--model name required\n"; return false; }


    // normalize path
    if(!g_tensorBinDir.empty())
    {
        char last = g_tensorBinDir.back();
        if(last != '/' && last != '\\')
            g_tensorBinDir += "/";
    }

    return true;
}


// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(!parseArgs(argc, argv))
    {
        std::cout << "\nUsage:\n"
                  << argv[0]
                  << " --binaries <tensor_bin_dir> (for enc)"
                  << " --meta <meta_file> (for enc)"
                  << " --name <model_name>"
                  << " --encode --decode --bitstream (for dec) \n\n";

        return -1;
    }

    std::cout << "=== Configuration ===\n";
    if (g_doEncode){
        std::cout << "Tensor bin dir : " << g_tensorBinDir << "\n";
        std::cout << "Meta file      : " << g_metaFile << "\n";
        std::cout << "Model name     : " << g_modelName << "\n";
    }
    if (g_doDecode){
        std::cout << "Bitstream file : " << g_bitstreamFile << "\n";
        std::cout << "Model name     : " << g_modelName << "\n";
    }
    

    std::map<TensorBitwidth, CodingStats> stats;

    std::vector<TensorMeta> modelTensors;

    if(g_doEncode)
    {
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
    }

    

    std::vector<uint8_t> bytestream;
    uint32_t numGtxFlags = 4;
    // --------------------------------------------------
    // ENCODING
    // --------------------------------------------------
    if (g_doEncode){
        Encoder encoder;

        std::cout << "\n=== Encoding Model ===\n";

        PeakMemorySampler encSampler;
        size_t baselineEncMem = getCurrentRSS();          // baseline before encode

        encSampler.start();

        auto encStart = std::chrono::high_resolution_clock::now();

        encoder.initCtxModels(numGtxFlags);
        bytestream = encoder.encodeModel(modelTensors);

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


        std::string bitstreamFile = g_modelName + ".bin";
        std::string decodedDir    = g_modelName + "_decoded";

        std::ofstream f(bitstreamFile, std::ios::binary);
        f.write(reinterpret_cast<const char*>(bytestream.data()),
            bytestream.size());

        std::cout << "Bitstream saved to: " << bitstreamFile << "\n";

        

        int num_tensors = modelTensors.size();
        uint64_t originalBytes = 0;

        for(const auto& t : modelTensors)
        {
            uint32_t bw = getBitwidthFromEnum(t.tensorBitwidth);

            originalBytes +=
                (uint64_t)t.data.size() * bw / 8;
        }

         uint64_t compressedBytes = bytestream.size();

        double ratio =
            (double)originalBytes / compressedBytes;

        double encodeMB =
        (double)originalBytes / (1024.0*1024.0);


    std::cout << "\n========== ENCODING SUMMARY ==========\n"
              << "Tensors processed  : " << num_tensors        << "\n"
              << "Original size      : " << encodeMB                    << " MB\n"
              << "Compressed size    : " << compressedBytes/(1024.0*1024.0) << " MB\n"
              << "Compression ratio  : " << ratio                       << "\n"
              << "\nEncoding time      : " << encTime                   << " sec\n"
              << "Encode speed       : " << encodeMB / encTime          << " MB/s\n";

         printMemStats("Encode memory:", encMemStats);

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
    }

    if (g_doEncode && g_doDecode){    
        // --------------------------------------------------
        // Free model tensors before decoding — decoder only needs the bytestream
        // --------------------------------------------------
        std::cout << "\n=== Freeing model tensors before decode ===\n";
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

    }
    // --------------------------------------------------
    // DECODING
    // --------------------------------------------------
    // necessary to load bitstream from file if encoding not performed
    if (g_doDecode && !g_doEncode){
        std::ifstream f(g_bitstreamFile, std::ios::binary | std::ios::ate);
        if(!f)
        {
            std::cerr << "Failed to open bitstream\n";
            return -1;
        }
        std::streamsize size = f.tellg();
        f.seekg(0, std::ios::beg);
        bytestream.resize(size);
        f.read(reinterpret_cast<char*>(bytestream.data()), size);
        std::cout << "Loaded bitstream: " << size << " bytes\n";
    }

    if(g_doDecode){
        std::cout << "\n=== Decoding Model ===\n";

        Decoder decoder;
        std::vector<TensorMeta> decodedModel;

        PeakMemorySampler decSampler;
        size_t baselineDecMem = getCurrentRSS();          // baseline before decode

        decSampler.start();
        auto decStart = std::chrono::high_resolution_clock::now();
        
        //decoder.setStream(const_cast<std::vector<uint8_t>&>(bytestream));

        decoder.initCtxModels(numGtxFlags);
        printf("Start decoding...\n");
        decoder.decodeModel(decodedModel, bytestream);

        auto decEnd = std::chrono::high_resolution_clock::now();
        MemoryStats decMemStats = decSampler.stop(baselineDecMem);

        double decTime = std::chrono::duration<double>(decEnd-decStart).count();

        std::cout << "Decoded tensors: "
                << decodedModel.size()
                << "\n";


        /// save decoded tensormeta
        std::string decodedDir = g_modelName + "_decoded";
        saveDecodedModel(decodedModel, decodedDir);

        uint64_t decodedBytes = 0;

        for(const auto& t : decodedModel)
        {
            decodedBytes += t.data.size() * sizeof(int32_t);
        }

        double decodedMB = decodedBytes / (1024.0 * 1024.0);

        std::cout << "\n========== DECODING SUMMARY ==========\n"
            << "Decoded size       : " << decodedMB << " MB\n"
            << "Decoding time      : " << decTime                     << " sec\n"
            << "Decode speed       : " << decodedMB / decTime          << " MB/s\n";
        printMemStats("Decode memory:", decMemStats);
    }
   
    

    return 0;
}
