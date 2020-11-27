#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "cuda.h"

#define UINT64_BITS (sizeof(uint64_t) * 8)

class BitField {
public:
	// Constructs a zero bit field. Deleting this object also frees the internal buffer.
	HYBRID_CALL BitField(uint64_t size) : size(size), wrapper(false) {
		// Equivalent to: ceil(size / 64)
		size_t buffer_size = ((size + (UINT64_BITS - 1)) / UINT64_BITS);

		buffer = new uint64_t[buffer_size];
		memset(buffer, 0, buffer_size * sizeof(uint64_t));
	}

	HYBRID_CALL ~BitField() {
		if (!wrapper)
			delete[] buffer;
	}

	std::string to_string() const {
		std::string builder;
		builder.reserve(size);

		for(uint64_t i = 0; i < size; i++)
			builder.append(get(i) ? "1" : "0");

		return builder;
	}

	// Constructs a wrapper bit field. Deleting this object does not free the wrapped buffer.
	HYBRID_CALL static BitField wrap(uint64_t *buffer, uint64_t size) {
		return BitField(buffer, size);
	}

	HYBRID_CALL bool get(uint64_t index) const {
		assert(index < size && "Index is out of bounds.");

		size_t buffer_index = index / UINT64_BITS;
		int32_t bit_offset = static_cast<int32_t>(index % UINT64_BITS);

		return ((buffer[buffer_index] >> bit_offset) & 1) == 1;
	}

	HYBRID_CALL void set(uint64_t index, bool state) {
		assert(index < size && "Index is out of bounds.");

		size_t buffer_index = index / UINT64_BITS;
		int32_t bit_offset = static_cast<int32_t>(index % UINT64_BITS);

		if (state)
			buffer[buffer_index] |= 1ULL << bit_offset;
		else
			buffer[buffer_index] &= ~(1ULL << bit_offset);
	}

	HYBRID_CALL void clear() {
		memset(buffer, 0, (size + 7) / 8);
	}

	HYBRID_CALL uint64_t *get_buffer() const {
		return buffer;
	}

private:
	uint64_t size;
	uint64_t *buffer;
	bool wrapper;

	HYBRID_CALL BitField(uint64_t *buffer, uint64_t size) :
			size(size), buffer(buffer), wrapper(true) {}
};
