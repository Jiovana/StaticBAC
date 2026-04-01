#include <bitset>
#include <limits>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstdint>
#include <string>


class BitstreamWriter {
    std::vector<uint8_t>* m_buf;
    uint8_t m_currByte;
    uint8_t m_bitsFilled; // how many bits used (0-7)

public:
    BitstreamWriter(std::vector<uint8_t>* buf) 
        : m_buf(buf), m_currByte(0), m_bitsFilled(0) {}

    void writeBit(uint8_t bit) {
        m_currByte = (m_currByte << 1) | (bit & 1);
        m_bitsFilled++;
        if (m_bitsFilled == 8) {
            m_buf->push_back(m_currByte);
            m_currByte = 0;
            m_bitsFilled = 0;
        }
    }

    void writeBits(uint32_t value, uint8_t numBits) {
        for (int i = numBits-1; i >= 0; i--) {
            writeBit((value >> i) & 1);
        }
    }

    // pad to next byte boundary
    void flushToByte() {
        if (m_bitsFilled > 0) {
            m_currByte <<= (8 - m_bitsFilled);
            m_buf->push_back(m_currByte);
            m_currByte = 0;
            m_bitsFilled = 0;
        }
    }

    void writeIAE(uint8_t v, int32_t value)
    {
        if (v == 0) return;

        uint32_t mask = (v == 32) ? 0xFFFFFFFFu : ((1u << v) - 1);
        uint32_t pattern = uint32_t(value) & mask;

        writeBits(pattern, v);
    }
};


class BitstreamReader {
    const std::vector<uint8_t>* m_buf = nullptr;  // vector mode
    const uint8_t* m_ptr  = nullptr;             // pointer mode
    bool m_usePtr = false;

    size_t m_pos = 0;    // byte index for vector
    uint8_t m_currByte = 0;
    uint8_t m_bitsLeft = 0;

public:

    const uint8_t* m_initialPtr = nullptr; // only for pointer mode

    BitstreamReader(const uint8_t* ptr)
        : m_buf(nullptr), m_ptr(ptr), m_initialPtr(ptr), m_usePtr(true),
        m_pos(0), m_currByte(0), m_bitsLeft(0) {}

    // Vector constructor (existing)
    BitstreamReader(const std::vector<uint8_t>* buf)
        : m_buf(buf), m_ptr(nullptr), m_usePtr(false),
          m_pos(0), m_currByte(0), m_bitsLeft(0) {}


    uint8_t readBit() {
        if (m_bitsLeft == 0) {
            if (m_usePtr) {
                m_currByte = *m_ptr++;
            } else {
                m_currByte = (*m_buf)[m_pos++];
            }
            m_bitsLeft = 8;
        }
        uint8_t bit = (m_currByte >> (m_bitsLeft-1)) & 1;
        m_bitsLeft--;
        return bit;
    }

    uint32_t readBits(uint8_t numBits) {
        uint32_t val = 0;
        for (int i = 0; i < numBits; i++)
            val = (val << 1) | readBit();
        return val;
    }

    void alignToByte() {
        m_bitsLeft = 0;
    }

    size_t getBytesRead() const {
        if (m_usePtr) {
            return m_ptr - m_initialPtr; // you need to store initial pointer
        } else {
            return m_pos;
        }
    }
    
    int32_t readIAE(uint8_t v)
    {
        if (v == 0) return 0;

        uint32_t pattern = readBits(v);

        // sign extension
        return int32_t(pattern << (32 - v)) >> (32 - v);
    }
};