#pragma once
#include <cstdio>
#include <string>


template <typename... Ts>
inline std::string
format(const char* format, Ts... args) {
    constexpr size_t len = 512;
    char buff[len];
    snprintf(buff, len, format, args...);
    return buff;
}


inline void
_errMsg(const char* msg, const char* file, int line, const char* func) {
    printf("\n  File \"%s\", line %d, in %s\n", file, line, func);
    printf(">>> Error: %s\n", msg);
    exit(EXIT_FAILURE);
}
inline void
_errMsg(const std::string& msg, const char* file, int line, const char* func) {
    printf("\n  File \"%s\", line %d, in %s\n", file, line, func);
    printf(">>> Error: %s\n", msg.c_str());
    exit(EXIT_FAILURE);
}
#define errMsg(msg) (_errMsg(msg, __FILE__, __LINE__, __func__))


inline void
_assert(const char* msg, const char* file, int line, const char* func) {
    printf("\n  File \"%s\", line %d, in %s\n", file, line, func);
    printf(">>> Assertion Failed: %s\n", msg);
    exit(EXIT_FAILURE);
}
#define ASSERT(assertion) { if (!(assertion)) _assert(#assertion, __FILE__, __LINE__, __func__); }

