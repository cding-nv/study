#include <string.h>
#include <iostream>

using namespace std;


class Blob {
public:
    Blob()
    : data_(nullptr), size_(0) {
        log("Blob's default constructor");
    }

    explicit Blob(size_t size)
    : data_(new char[size]), size_(size) {
        log("Blob's parameter constructor");
    }

    ~Blob() {
        log("Blob's destructor");
        delete[] data_;
    }

    Blob(const Blob& other) {
        log("Blob's copy constructor");
        data_ = new char[other.size_];
        memcpy(data_, other.data_, other.size_);
        size_ = other.size_;
    }

    Blob& operator=(const Blob& other) {
        log("Blob's copy assignment operator");
        if (this == &other) {
            return *this;
        }
        delete[] data_;
        data_ = new char[other.size_];
        memcpy(data_, other.data_, other.size_);
        size_ = other.size_;
        return *this;
    }

    void set(size_t offset, size_t len, const void* src) {
        len = min(len, size_ - offset);
        memcpy(data_ + offset, src, len);
    }

private:
    char* data_;
    size_t size_;

    void log(const char* msg) {
        cout << "[" << this << "] " << msg << endl;
    }
};

Blob createBlob(const char* str) {
    size_t len = strlen(str);
    Blob blob(len);
    blob.set(0, len, str);
    return blob;
}

int main() {

    Blob blob;

    cout << "Start assigning value..." << endl;
    blob = createBlob("A very very very long string representing serialized data");
    cout << "End assigning value" << endl;

    return 0;
}
