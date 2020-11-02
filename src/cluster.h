#ifndef CLUSTER_H
#define CLUSTER_H

#include <cstdint>
#include <string>

#include "cuda.h"

class Cluster {
public:
    HYBRID_CALL Cluster() : Cluster(0, 0, 0, 0, 0, 0) {}

    HYBRID_CALL Cluster(uint64_t world_seed, int32_t size, int32_t min_x, int32_t min_z, int32_t max_x, int32_t max_z) :
            world_seed(world_seed), size(size), min_x(min_x), min_z(min_z), max_x(max_x), max_z(max_z) {}

    HYBRID_CALL bool operator==(const Cluster& other) const {
        return world_seed == other.world_seed && size == other.size &&
                min_x == other.min_x && min_z == other.min_z &&
                max_x == other.max_x && max_z == other.max_z;
    }

    std::string to_string() const {
        return "Cluster of " + std::to_string(size) + " chunks at " + std::to_string(get_center_x() * 16) +
                ", " + std::to_string(get_center_z() * 16) + " (seed: " + std::to_string(world_seed) + ").";
    }
    
    HYBRID_CALL uint64_t get_world_seed() const {
        return world_seed;
    }

    HYBRID_CALL int32_t get_center_x() const {
        return (max_x + min_x) / 2;
    }

    HYBRID_CALL int32_t get_center_z() const {
        return (max_z + min_z) / 2;
    }

    HYBRID_CALL int32_t get_size() const {
        return size;
    }

private:
    uint64_t world_seed;
    int32_t size, min_x, min_z, max_x, max_z;
};

#endif // CLUSTER_H