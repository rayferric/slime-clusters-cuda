#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "boinc_api.h"

#include "bit_field.h"
#include "cluster.h"
#include "cuda.h"
#include "java_random.h"
#include "lcg.h"
#include "random.h"
#include "stack.h"

// This can be tuned, should be a multiple of warp size. (Doesn't influence device occupancy.)
#define BLOCK_SIZE 512

// Size of the collector array, or how many items are expected to be found in a single work unit:
#define MAX_COLLECTOR_SIZE (1 << 16)

#define LOG_FILE "clusters.txt"
#define STATE_FILE ".state"

#define STACK_SIZE 128

// 32x32 cache around the examined chunk.
// Algorithm ignores everything outside this region.
// Duplicate entries may appear when the clipping occurs.
// (Happens only to clusters whose AABB is larger than 16x16.)
// Set this to the smallest value possible:
#define CACHE_EXTENTS 16

#define UINT64_BITS (sizeof(uint64_t) * 8)

const size_t CACHE_SIZE_BITS = (CACHE_EXTENTS * 2ULL) * (CACHE_EXTENTS * 2ULL);
const size_t CACHE_SIZE_UINT64 = ((CACHE_SIZE_BITS + (UINT64_BITS - 1)) / UINT64_BITS);

// Device code:

__constant__ int32_t c_min_cluster_size, c_search_region_extents;
__constant__ uint64_t c_chunks_per_seed;

class Offset {
public:
    int8_t x, z;

    HYBRID_CALL Offset() : Offset(0, 0) {}

    HYBRID_CALL Offset(int8_t x, int8_t z) : x(x), z(z) {}
};

HYBRID_CALL bool check_slime_chunk(JavaRandom &rand, uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
    world_seed += CUDA::wrapping_mul(CUDA::wrapping_mul(chunk_x, chunk_x), 0x4C1906);
    world_seed += CUDA::wrapping_mul(chunk_x, 0x5AC0DB);
    world_seed += CUDA::wrapping_mul(chunk_z, chunk_z) * 0x4307A7ULL;
    world_seed += CUDA::wrapping_mul(chunk_z, 0x5F24F);
    world_seed ^= 0x3AD8025FULL;

    rand.set_seed(world_seed);
    rand.scramble();
    return rand.next_int(10) == 0;
}

DEVICE_CALL Cluster explore_cluster(BitField &cache, Stack<Offset> &stack, JavaRandom &rand, uint64_t world_seed, int32_t origin_x, int32_t origin_z) {
    stack.clear();
    stack.push(Offset(0, 0));

    int32_t cluster_size = 0;
    int32_t min_x = INT_MAX, min_z = INT_MAX, max_x = INT_MIN, max_z = INT_MIN;
    while(!stack.is_empty()) {
        Offset offset = stack.pop();

        int32_t chunk_x = origin_x + offset.x;
        int32_t chunk_z = origin_z + offset.z;

        uint64_t cache_idx = (offset.x + CACHE_EXTENTS) * (CACHE_EXTENTS * 2ULL) + (offset.z + CACHE_EXTENTS);
        if(cache.get(cache_idx))
            continue;
        cache.set(cache_idx, true);

        if(!check_slime_chunk(rand, world_seed, chunk_x, chunk_z))
            continue;

        if(chunk_x < min_x)
            min_x = chunk_x;
        if(chunk_z < min_z)
            min_z = chunk_z;
        if(chunk_x > max_x)
            max_x = chunk_x;
        if(chunk_z > max_z)
            max_z = chunk_z;

        cluster_size++;

        if(chunk_x + 1 < c_search_region_extents && offset.x + 1 < CACHE_EXTENTS)
            stack.push(Offset(offset.x + 1, offset.z));
        if(chunk_x - 1 >= -c_search_region_extents && offset.x - 1 >= -CACHE_EXTENTS)
            stack.push(Offset(offset.x - 1, offset.z));
        if(chunk_z + 1 < c_search_region_extents && offset.z + 1 < CACHE_EXTENTS)
            stack.push(Offset(offset.x, offset.z + 1));
        if(chunk_z - 1 >= -c_search_region_extents && offset.z - 1 >= -CACHE_EXTENTS)
            stack.push(Offset(offset.x, offset.z - 1));
    }

    return Cluster(world_seed, cluster_size, min_x, min_z, max_x, max_z);
}

__global__ void kernel(uint64_t block_count, uint64_t offset, uint64_t *collector_size, Cluster *collector) {
    uint64_t world_seed = blockIdx.x + offset;

    // Preferred number of chunks to be processed by a single thread:
    uint64_t min_chunks_per_thread = c_chunks_per_seed / blockDim.x;
    uint64_t starting_chunk = threadIdx.x * min_chunks_per_thread;
    uint64_t chunks_here; // Number of chunks processed by this thread.

    // The last thread will process any extra chunks (shouldn't happen with a good BLOCK_SIZE).
    if(threadIdx.x == blockDim.x - 1) 
        chunks_here = c_chunks_per_seed - (blockDim.x - 1) * min_chunks_per_thread;
    else
        chunks_here = min_chunks_per_thread;

    // Static allocation is the key to performance:
    uint64_t cache_buffer[CACHE_SIZE_UINT64];
    Offset stack_buffer[STACK_SIZE];
    BitField cache = BitField::wrap(cache_buffer, CACHE_SIZE_BITS);
    Stack<Offset> stack = Stack<Offset>::wrap(stack_buffer, STACK_SIZE);
    JavaRandom rand = JavaRandom();

    for(uint64_t i = 0; i < chunks_here; i++) {
        uint64_t chunk_idx = i + starting_chunk;
        int32_t chunk_x = (chunk_idx / (c_search_region_extents * 2)) - c_search_region_extents;
        int32_t chunk_z = (chunk_idx % (c_search_region_extents * 2)) - c_search_region_extents;

        cache.clear();
        // Stack is left empty after every call to explore_cluster(...), so there's no need to clear it here.
        Cluster cluster = explore_cluster(cache, stack, rand, world_seed, chunk_x, chunk_z);
        if(cluster.get_size() >= c_min_cluster_size) {
            uint64_t collector_idx = atomicAdd(collector_size, 1);
            collector[collector_idx] = cluster;
        }
    }
}

// End of device code.

uint64_t get_millis() {
	auto epoch = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
}

//
// Accepted options:
// -s <int> | --start   | what seed to start the search at (inclusive) (default: 0)
// -e <int> | --end     | what seed to end the search at (inclusive) (default: 0)
// -c <int> | --cluster | set the minimum searched size of a cluster (default: 25)
// -r <int> | --region  | resize the search region, this value is half the side of the bounding rectangle (default: 512)
// 
int32_t main(int32_t argc, char **argv) {
    uint64_t start_offset = 0, end_offset = 0, offset = 0;
    int32_t min_cluster_size = 25, search_region_extents = 512;
    
    for(int32_t i = 1; i < argc; i += 2) {
        const char *option = argv[i], *value = argv[i + 1];

        if(strcmp(option, "-s") == 0 || strcmp(option, "--start") == 0)
            sscanf(value, "%llu", &start_offset);
        else if(strcmp(option, "-e") == 0 || strcmp(option, "--end") == 0)
            sscanf(value, "%llu", &end_offset);
        else if(strcmp(option, "-c") == 0 || strcmp(option, "--cluster") == 0)
            sscanf(value, "%d", &min_cluster_size);
        else if(strcmp(option, "-r") == 0 || strcmp(option, "--region") == 0)
            sscanf(value, "%d", &search_region_extents);
    }
    if(end_offset < start_offset)
        end_offset = start_offset;

    BOINC_OPTIONS options;

    boinc_options_defaults(options);
    options.normal_thread_priority = true; // Since we're using CUDA.
    boinc_init_options(&options);

    APP_INIT_DATA aid;
    boinc_get_init_data(aid);
    
    int32_t device_index = 0;
    if(aid.gpu_device_num > 0)
        fprintf(stderr, "Setting up BOINC specified device #%d\n", device_index = aid.gpu_device_num);
    else
        fprintf(stderr, "Using  default device #0.\n");

    std::string boinc_log_path, boinc_state_path;
    if(boinc_resolve_filename_s(LOG_FILE, boinc_log_path))
        boinc_temporary_exit(5, "Failed to access log file.", true);
    if(boinc_resolve_filename_s(STATE_FILE, boinc_state_path))
        boinc_temporary_exit(5, "Failed to access checkpoint file.", true);

    FILE *log_file = boinc_fopen(boinc_log_path.c_str(), "a");

    FILE *state_file = boinc_fopen(boinc_state_path.c_str(), "r");
    if(state_file == nullptr) {
        fprintf(stderr, "No checkpoint yet available.\n");
        offset = start_offset;
    } else {
        fscanf(state_file, "%llu", &offset);
        fclose(state_file);
        if(offset < start_offset || offset > end_offset) {
            fprintf(stderr, "Checkpoint was outdated, restarting from the beginning.\n");
            offset = start_offset;
        } else
            fprintf(stderr, "Loaded checkpoint: %llu\n", offset);
    }

    fprintf(stderr, "Start offset: %llu\n", start_offset);
    fprintf(stderr, "End offset: %llu\n", end_offset);
    fprintf(stderr, "Current offset: %llu\n", offset);
    fprintf(stderr, "Required cluster size: %d\n", min_cluster_size);

    // Init CUDA device:

    cudaSetDevice(device_index);

    // Approximate the number of threads that has to be launched per wave for maximum performance:
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    int32_t pref_wave_size = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    int32_t max_block_count = pref_wave_size / BLOCK_SIZE;

    fprintf(stderr, "Blocks per wave: %d\n", max_block_count);

    // Allocate constants:
    uint64_t chunks_per_seed = (search_region_extents * 2ULL) * (search_region_extents * 2ULL);
    cudaMemcpyToSymbol(c_min_cluster_size, &min_cluster_size, sizeof(int32_t));
    cudaMemcpyToSymbol(c_search_region_extents, &search_region_extents, sizeof(int32_t));
    cudaMemcpyToSymbol(c_chunks_per_seed, &chunks_per_seed, sizeof(uint64_t));

    // Found items are collected in video memory.
    // Any collected items will be transferred to the host in-between work unit executions.
    uint64_t *d_collector_size;
    Cluster *d_collector;
    cudaMalloc(&d_collector_size, sizeof(uint64_t));
    cudaMalloc(&d_collector, MAX_COLLECTOR_SIZE * sizeof(Cluster));

    uint64_t total_found = 0;
    int32_t secs_since_checkpoint = 0;
    uint64_t timer = get_millis();
    while(offset <= end_offset) {
        // Reset caches:
        cudaMemset(d_collector_size, 0, sizeof(uint64_t));

        // Launch one wave (one block does one seed):
        uint64_t block_count = end_offset - offset + 1; // +1, cause {end} is inclusive
        if(block_count > max_block_count)
            block_count = max_block_count;
        kernel<<<block_count, BLOCK_SIZE>>>(block_count, offset, d_collector_size, d_collector);
        offset += block_count;

        // Synchronize:
        cudaDeviceSynchronize();

        // Catch errors:
        cudaError_t code;
        if((code = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(code));
            break;
        }

        // Dump the collector:
        uint64_t collector_size;
        cudaMemcpy(&collector_size, d_collector_size, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        Cluster *clusters = new Cluster[collector_size];
        cudaMemcpy(clusters, d_collector, collector_size * sizeof(Cluster), cudaMemcpyDeviceToHost);

        Cluster *end = clusters + collector_size;
        for(Cluster *cluster = clusters; cluster != end; cluster++) {
            // Remove all other duplicates of this cluster:
            end = std::remove(cluster + 1, end, *cluster);

            std::string cluster_info = cluster->to_string();
            fprintf(log_file, "%s\n", cluster_info.c_str());
            fflush(log_file);

            total_found++;
        }
        delete[] clusters;

        // Monitoring and checkpoints (executes every second):

        if(get_millis() - timer < 1000)
            continue;
        timer = get_millis();
        
        boinc_fraction_done((double)(offset - start_offset) / (end_offset - start_offset + 1));

        // 10 secs have passed or BOINC has been suspended:
        if(boinc_time_to_checkpoint() || ++secs_since_checkpoint >= 10) {
            boinc_begin_critical_section();

            // Recreate checkpoint file ("w" implicitly clears contents):
            state_file = boinc_fopen(boinc_state_path.c_str(), "w");

            // Actually, this should always execute:
            if(state_file != nullptr) {
                fprintf(state_file, "%llu", offset);    
                fclose(state_file);
            }
            
            boinc_end_critical_section();
            boinc_checkpoint_completed();

            fprintf(stderr, "Checkpoint saved: %llu\n", offset);
            secs_since_checkpoint = 0;
        }
    }

    cudaFree(d_collector);
    cudaFree(d_collector_size);

    fclose(log_file);

    if(offset <= end_offset)
        return boinc_finish_message(1, "Program has finished but the task is not yet complete.\n"
                "This indicates that your CUDA device has shut down during the process.\n"
                "See error logs for more information.", true);
    
    // Done! ^w^
    return boinc_finish_message(0, ("Finished, " + std::to_string(total_found) + " cluster" +
            (total_found == 1 ? " was" : "s were") + " found.").c_str(), false);
}