#ifndef STACK_H
#define STACK_H

#include <cassert>
#include <cstdint>
#include <string>

#include "cuda.h"

template <class T>
class Stack {
public:
    CUDA_CALL Stack(size_t capacity) : size(0), capacity(capacity), wrapper(false) {
        assert(capacity != 0 && "Capacity must be positive.");
        buffer = new T[capacity];
    }

    CUDA_CALL ~Stack() {
        if(!wrapper)
            delete[] buffer;
    }

    CUDA_CALL static Stack wrap(T *buffer, size_t capacity) {
        return Stack(buffer, capacity);
    }

    CUDA_CALL void push(const T &value) {
        assert(!is_full() && "The stack is full.");
        buffer[size++] = value;
    }

    CUDA_CALL T pop() {
        assert(!is_empty() && "The stack is empty.");
        return buffer[--size];
    }

    CUDA_CALL int is_empty() const { 
        return size == 0; 
    }

    CUDA_CALL int is_full() const { 
        return size == capacity; 
    } 

    CUDA_CALL size_t get_size() const {
        return size;
    }

    CUDA_CALL size_t get_capacity() const {
        return capacity;
    }

    CUDA_CALL T *get_buffer() const {
        return buffer;
    }
private:
    size_t size, capacity;
    T *buffer;
    bool wrapper;

    CUDA_CALL Stack(T *buffer, size_t capacity) :
            size(0), capacity(capacity), buffer(buffer), wrapper(true) {}
};

#endif // STACK_H