#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace velocityai {

class VelocityError : public std::runtime_error {
public:
    explicit VelocityError(const std::string& msg) : std::runtime_error(msg) {}
};

class ShapeError : public VelocityError {
public:
    explicit ShapeError(const std::string& msg) : VelocityError("ShapeError: " + msg) {}
};

class TypeError : public VelocityError {
public:
    explicit TypeError(const std::string& msg) : VelocityError("TypeError: " + msg) {}
};

class DeviceError : public VelocityError {
public:
    explicit DeviceError(const std::string& msg) : VelocityError("DeviceError: " + msg) {}
};

#define VAI_CHECK(condition, msg) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << "[" << __FILE__ << ":" << __LINE__ << "] Check failed: " #condition ". " << msg; \
            throw ::velocityai::VelocityError(oss.str()); \
        } \
    } while (0)

#define VAI_CHECK_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::ostringstream oss; \
            oss << "[" << __FILE__ << ":" << __LINE__ << "] Check failed: (" #a " == " #b ") [" \
                << (a) << " vs " << (b) << "]. " << msg; \
            throw ::velocityai::VelocityError(oss.str()); \
        } \
    } while (0)

} // namespace velocityai
