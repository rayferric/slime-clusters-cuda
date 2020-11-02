#ifndef RANDOM_H
#define RANDOM_H

#include <cstdint>

#include "cuda.h"
#include "lcg.h"

class Random {
public:
    HYBRID_CALL Random(const LCG &lcg, uint64_t seed) : lcg(lcg) {
        set_seed(seed);
    }

    HYBRID_CALL void scramble(const LCG &lcg) {
        seed = lcg.scramble(seed);
    }

    HYBRID_CALL void scramble() {
        scramble(lcg);
    }

    HYBRID_CALL void skip() {
        seed = lcg.next(seed);
    }

    HYBRID_CALL void skip(int32_t step) {
        seed = lcg.of_step(step).next(seed);
    }

    HYBRID_CALL LCG get_lcg() const {
        return lcg;
    }

    HYBRID_CALL void set_lcg(const LCG &lcg) {
        this->lcg = lcg;
    }

    HYBRID_CALL uint64_t get_seed() const {
        return seed;
    }

    HYBRID_CALL void set_seed(uint64_t seed) {
        this->seed = lcg.mod(seed);
    }

    HYBRID_CALL uint64_t next_seed() {
        skip();
        return seed;
    }
protected:
    LCG lcg;
    uint64_t seed;
};

#endif // RANDOM_H