#pragma once
#include <fstream>
#include <string>
#include <mutex>
#include <filesystem>

class SimpleLogger
{
public:
    SimpleLogger(const std::string &baseDir = "ctx_logs")
        : baseDirectory(baseDir)
    {
        // create logs directory if needed (C++17)
        // note: you may need to include <filesystem> and use std::filesystem::create_directories
         std::filesystem::create_directories(baseDirectory);
    }

    ~SimpleLogger()
    {
        closeFile();
    }

    // switch the output file to a new tensor
    void setTensorName(const std::string &tensorName)
    {
        std::lock_guard<std::mutex> guard(mtx);
        closeFile();
        std::string sanitized = sanitize(tensorName);
        std::string path = baseDirectory + "/" + sanitized + ".txt";
        out.open(path, std::ios::out | std::ios::trunc);
    }

    // write a line
    void log(const std::string &line)
    {
        std::lock_guard<std::mutex> guard(mtx);
        if (out.is_open())
        {
            out << line << "\n";
        }
    }

    // write a number
    template<typename T>
    void logVal(T value)
    {
        log(std::to_string(value));
    }

private:
    std::ofstream out;
    std::mutex mtx;
    std::string baseDirectory;

    void closeFile()
    {
        if (out.is_open())
        {
            out.close();
        }
    }

    std::string sanitize(const std::string &name)
    {
        std::string s = name;
        for (auto &c : s)
            if (!isalnum(c) && c != '_') c = '_';
        return s;
    }
};

// Macro for easy logging (like PROFILE_SCOPE)
#define LOG_LINE(logger, content) \
    { if(logger) (logger)->log(content); }

#define LOG_VAL(logger, value) \
    { if(logger) (logger)->logVal(value); }
