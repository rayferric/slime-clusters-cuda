#pragma once

#include <cassert>
#include <cstdint>
#include <string>

#include "cuda.h"

template <class T>
class Stack {
public:
	HYBRID_CALL Stack(size_t capacity) : size(0), capacity(capacity), wrapper(false) {
		assert(capacity != 0 && "Capacity must be positive.");
		buffer = new T[capacity];
	}

	HYBRID_CALL ~Stack() {
		if (!wrapper)
			delete[] buffer;
	}

	HYBRID_CALL static Stack wrap(T *buffer, size_t capacity) {
		return Stack(buffer, capacity);
	}

	HYBRID_CALL bool push(const T &value) {
		if (is_full())
			return false;
		buffer[size++] = value;
		return true;
	}

	HYBRID_CALL T pop() {
		assert(!is_empty() && "The stack is empty.");
		return buffer[--size];
	}

	HYBRID_CALL void clear() { 
		size = 0;
	}

	HYBRID_CALL int is_empty() const { 
		return size == 0; 
	}

	HYBRID_CALL int is_full() const { 
		return size == capacity; 
	} 

	HYBRID_CALL size_t get_size() const {
		return size;
	}

	HYBRID_CALL size_t get_capacity() const {
		return capacity;
	}

	HYBRID_CALL T *get_buffer() const {
		return buffer;
	}
private:
	size_t size, capacity;
	T *buffer;
	bool wrapper;

	HYBRID_CALL Stack(T *buffer, size_t capacity) :
			size(0), capacity(capacity), buffer(buffer), wrapper(true) {}
};
