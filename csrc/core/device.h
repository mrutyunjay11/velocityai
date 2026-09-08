#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include "csrc/core/error.h"

namespace velocityai {

enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    METAL = 2
};

struct Device {
    DeviceType type;
    int index;

    Device() : type(DeviceType::CPU), index(0) {}
    Device(DeviceType t, int idx = 0) : type(t), index(idx) {}

    static Device CPU() { return Device(DeviceType::CPU, 0); }
    static Device CUDA(int idx = 0) { return Device(DeviceType::CUDA, idx); }
    static Device METAL(int idx = 0) { return Device(DeviceType::METAL, idx); }

    bool is_cpu() const { return type == DeviceType::CPU; }
    bool is_cuda() const { return type == DeviceType::CUDA; }
    bool is_metal() const { return type == DeviceType::METAL; }

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    std::string str() const {
        std::ostringstream oss;
        switch (type) {
            case DeviceType::CPU: return "cpu";
            case DeviceType::CUDA: oss << "cuda:" << index; return oss.str();
            case DeviceType::METAL: oss << "metal:" << index; return oss.str();
            default: return "unknown";
        }
    }

    static Device from_string(const std::string& dev_str) {
        if (dev_str == "cpu") return Device(DeviceType::CPU, 0);
        if (dev_str == "metal") return Device(DeviceType::METAL, 0);
        if (dev_str.rfind("metal:", 0) == 0) {
            int idx = std::stoi(dev_str.substr(6));
            return Device(DeviceType::METAL, idx);
        }
        if (dev_str == "cuda") return Device(DeviceType::CUDA, 0);
        if (dev_str.rfind("cuda:", 0) == 0) {
            int idx = std::stoi(dev_str.substr(5));
            return Device(DeviceType::CUDA, idx);
        }
        throw DeviceError("Unknown device string: " + dev_str);
    }
};

} // namespace velocityai
