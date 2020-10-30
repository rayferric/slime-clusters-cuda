#ifndef LCG_H
#define LCG_H

#include <cstdint>

#include "cuda.h"

class LCG {
public:
    CUDA LCG(uint64_t multiplier, uint64_t addend, uint64_t modulus) :
            multiplier(multiplier), addend(addend), modulus(modulus) {
        // True for when the modulus is a power of two:
        can_mask = (modulus & -modulus) == modulus;
    }

    CUDA LCG of_step(int32_t step) const {
        if(step == 0) return LCG(1, 0, modulus);
        if(step == 1) return *this;

        uint64_t base_multiplier = multiplier;
        uint64_t base_addend = addend;

        uint64_t multiplier = 1;
        uint64_t addend = 0;

        for(uint64_t i = step; i != 0; i >>= 1) {
            if((i & 1) != 0) {
                multiplier *= base_multiplier;
                addend *= base_multiplier;
                addend += base_addend;
            }

            base_addend *= (base_multiplier + 1);
            base_multiplier *= base_multiplier;
        }

        return LCG(multiplier, addend, modulus);
    }

    CUDA uint64_t mod(uint64_t seed) const {
        if(can_mask)
            return seed & (modulus - 1);
        else
            return seed % modulus;
    }

    CUDA uint64_t scramble(uint64_t seed) const {
        return mod(seed ^ multiplier);
    }

    CUDA uint64_t next(uint64_t seed) const {
        return mod(seed * multiplier + addend);
    }

    CUDA uint64_t get_multiplier() const {
        return multiplier;
    }

    CUDA uint64_t get_addend() const {
        return addend;
    }

    CUDA uint64_t get_modulus() const {
        return modulus;
    }

private:
    uint64_t multiplier, addend, modulus;
    bool can_mask;
};

#endif // LCG_H