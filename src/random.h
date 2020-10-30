#ifndef RANDOM_H
#define RANDOM_H

#include <cstdint>

#include "cuda.h"
#include "lcg.h"

class Random {
public:
	CUDA Random(const LCG &lcg, uint64_t seed) : lcg(lcg) {
		set_seed(seed);
	}

    CUDA void scramble(const LCG &lcg) {
		seed = lcg.scramble(seed);
	}

    CUDA void scramble() {
		scramble(lcg);
	}

    CUDA void skip() {
		seed = lcg.next(seed);
	}

    CUDA void skip(int32_t step) {
		seed = lcg.of_step(step).next(seed);
	}

    CUDA LCG get_lcg() const {
		return lcg;
	}

    CUDA void set_lcg(const LCG &lcg) {
		this->lcg = lcg;
	}

    CUDA uint64_t get_seed() const {
		return seed;
	}

    CUDA void set_seed(uint64_t seed) {
		this->seed = lcg.mod(seed);
	}

    CUDA uint64_t next_seed() {
		skip();
		return seed;
	}
protected:
    LCG lcg;
    uint64_t seed;
};

#endif // RANDOM_H