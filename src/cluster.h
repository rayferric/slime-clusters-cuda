#ifndef CLUSTER_H
#define CLUSTER_H

#include <cstdint>
#include <string>

#include "cuda.h"

class Cluster {
public:
    CUDA Cluster() : Cluster(0, 0, 0, 0) {}

    CUDA Cluster(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z, int32_t size) :
            world_seed(world_seed), chunk_x(chunk_x), chunk_z(chunk_z), size(size) {}

    std::string to_string() const {
        return "Cluster of " + std::to_string(size) + " chunks at " + std::to_string(chunk_x * 16) +
                ", " + std::to_string(chunk_z * 16) + " (seed: " + std::to_string(world_seed) + ").";
    }
    
    CUDA uint64_t get_world_seed() const {
        return world_seed;
    }

    CUDA int32_t get_chunk_x() const {
        return chunk_x;
    }

    CUDA int32_t get_chunk_z() const {
        return chunk_z;
    }

    CUDA int32_t get_size() const {
        return size;
    }

private:
    uint64_t world_seed;
    int32_t chunk_x, chunk_z, size;
};

#endif // CLUSTER_H