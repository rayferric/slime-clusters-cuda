#ifndef BIT_FIELD_H
#define BIT_FIELD_H

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "cuda.h"

#define UINT64_BITS (sizeof(uint64_t) * 8)

class BitField {
public:
    // Constructs a zero bit field. Deleting this object also frees the internal array.
    CUDA_CALL BitField(uint64_t size) : size(size), wrapper(false) {
        // Equivalent to: ceil(size / 64)
        size_t array_size = ((size + (UINT64_BITS - 1)) / UINT64_BITS);

        array = new uint64_t[array_size];
        memset(array, 0, array_size * sizeof(uint64_t));
    }

    CUDA_CALL ~BitField() {
        if(!wrapper)
            delete[] array;
    }

    std::string to_string() const {
        std::string builder;
        builder.reserve(size);

        for(uint64_t i = 0; i < size; i++)
            builder.append(get(i) ? "1" : "0");

        return builder;
    }

    // Constructs a wrapper bit field. Deleting this object does not free the wrapped array.
    CUDA_CALL static BitField *wrap(uint64_t *array, uint64_t size) {
        return new BitField(array, size);
    }

    CUDA_CALL bool get(uint64_t index) const {
        assert(index < size && "Index is out of bounds.");

        size_t array_index = index / UINT64_BITS;
        int32_t bit_offset = (int32_t)(index % UINT64_BITS);

        return ((array[array_index] >> bit_offset) & 1) == 1;
    }

    CUDA_CALL void set(uint64_t index, bool state) {
        assert(index < size && "Index is out of bounds.");

        size_t array_index = index / UINT64_BITS;
        int32_t bit_offset = (int32_t)(index % UINT64_BITS);

        if(state)
            array[array_index] |= 1UI64 << bit_offset;
        else
            array[array_index] &= ~(1UI64 << bit_offset);
    }

    CUDA_CALL uint64_t *get_array() const {
        return array;
    }

private:
    uint64_t size;
    uint64_t *array;
    bool wrapper;

    CUDA_CALL BitField(uint64_t *array, uint64_t size) :
            size(size), array(array), wrapper(true) {}
};

#endif // BIT_FIELD_H