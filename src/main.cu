#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "boinc/boinc_api.h"

#include "bit_field.h"
#include "cluster.h"
#include "cuda.h"
#include "java_random.h"
#include "lcg.h"
#include "random.h"
#include "stack.h"

// How many compute devices to use:
#define DEVICE_COUNT 1

// This can be tuned, should be a multiple of warp size. (Doesn't influence device occupancy.)
#define BLOCK_SIZE 512

// Where to start and how far to explore:
#define WORK_OFFSET 0
#define WORK_SIZE (1UI64 << 48)

// Size of the collector array, or how many items are expected to be found in a single work unit:
#define MAX_COLLECTOR_SIZE (1 << 16)

#define REPORT_DELAY 10000
#define LOG_FILE "clusters.txt"

#define EXTENTS 512
#define MIN_CLUSTER_SIZE 20

#define STACK_SIZE 128

// 32x32 cache around the examined chunk.
// Algorithm ignores everything outside this region.
// Duplicate entries may appear when the clipping occurs.
// (Happens only to clusters whose AABB is larger than 16x16.)
// Set this to the smallest value possible:
#define CACHE_EXTENTS 16

#define UINT64_BITS (sizeof(uint64_t) * 8)

const size_t CHUNKS_PER_SEED = (EXTENTS * 2UI64) * (EXTENTS * 2UI64);
const size_t CACHE_SIZE_BITS = (CACHE_EXTENTS * 2UI64) * (CACHE_EXTENTS * 2UI64);
const size_t CACHE_SIZE_UINT64 = ((CACHE_SIZE_BITS + (UINT64_BITS - 1)) / UINT64_BITS);

// Device code:

class Offset {
public:
    int8_t x, z;

    HYBRID_CALL Offset() : Offset(0, 0) {}

    HYBRID_CALL Offset(int8_t x, int8_t z) : x(x), z(z) {}
};

HYBRID_CALL bool check_slime_chunk(JavaRandom &rand, uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
    world_seed += CUDA::wrapping_mul(CUDA::wrapping_mul(chunk_x, chunk_x), 0x4C1906);
    world_seed += CUDA::wrapping_mul(chunk_x, 0x5AC0DB);
    world_seed += CUDA::wrapping_mul(chunk_z, chunk_z) * 0x4307A7UI64;
    world_seed += CUDA::wrapping_mul(chunk_z, 0x5F24F);
    world_seed ^= 0x3AD8025FUI64;

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

        uint64_t cache_idx = (offset.x + CACHE_EXTENTS) * (CACHE_EXTENTS * 2UI64) + (offset.z + CACHE_EXTENTS);
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

        if(chunk_x + 1 < EXTENTS && offset.x + 1 < CACHE_EXTENTS)
            stack.push(Offset(offset.x + 1, offset.z));
        if(chunk_x - 1 >= -EXTENTS && offset.x - 1 >= -CACHE_EXTENTS)
            stack.push(Offset(offset.x - 1, offset.z));
        if(chunk_z + 1 < EXTENTS && offset.z + 1 < CACHE_EXTENTS)
            stack.push(Offset(offset.x, offset.z + 1));
        if(chunk_z - 1 >= -EXTENTS && offset.z - 1 >= -CACHE_EXTENTS)
            stack.push(Offset(offset.x, offset.z - 1));
    }

    return Cluster(world_seed, cluster_size, min_x, min_z, max_x, max_z);
}

__global__ void kernel(uint64_t block_count, uint64_t offset, uint64_t *collector_size, Cluster *collector) {
    uint64_t world_seed = blockIdx.x + offset + WORK_OFFSET;

    uint64_t min_chunks_per_thread = CHUNKS_PER_SEED / blockDim.x; // Preferred number of chunks to be processed by a thread.
    uint64_t starting_chunk = threadIdx.x * min_chunks_per_thread;
    uint64_t chunks_here; // Number of chunks processed by this thread.
    if(threadIdx.x == blockDim.x - 1) // The last thread will process any extra chunks (shouldn't happen with a good BLOCK_SIZE).
        chunks_here = CHUNKS_PER_SEED - (blockDim.x - 1) * min_chunks_per_thread;
    else
        chunks_here = min_chunks_per_thread;

    // Static allocation is the key to performance:
    uint64_t cache_buffer[CACHE_SIZE_UINT64];
    Offset stack_buffer[STACK_SIZE];
    BitField cache = BitField::wrap(cache_buffer, CACHE_SIZE_BITS);
    Stack<Offset> stack = Stack<Offset>::wrap(stack_buffer, STACK_SIZE);
    JavaRandom rand = JavaRandom();

    // We iterate in a specific manner to slightly improve caching performance:
    for(uint64_t i = 0; i < chunks_here; i++) {
        uint64_t chunk_idx = i + starting_chunk;
        int32_t chunk_x = chunk_idx / (EXTENTS * 2) - EXTENTS;
        int32_t chunk_z = chunk_idx % (EXTENTS * 2) - EXTENTS;

        cache.clear();
        // Stack is left empty after very call to explore_cluster(...), so there's no need to clear it here.
        Cluster cluster = explore_cluster(cache, stack, rand, world_seed, chunk_x, chunk_z);
        if(cluster.get_size() >= MIN_CLUSTER_SIZE) {
            uint64_t collector_idx = atomicAdd(collector_size, 1);
            collector[collector_idx] = cluster;
        }
    }
}

// End of device code.

FILE *log_file;
uint64_t offset = 0;
std::atomic<uint64_t> found_total(0);
std::atomic<int32_t> devices_running(DEVICE_COUNT);
std::mutex stdout_mutex, stderr_mutex, log_file_mutex, offset_mutex;

void log(const std::string &text) {
    std::lock_guard<std::mutex> log_file_guard(log_file_mutex);
    fprintf(log_file, "%s\n", text.c_str());
    fflush(log_file);
}

void info(int32_t device_index, const std::string &text) {
    std::lock_guard<std::mutex> stdout_guard(stdout_mutex);
    fprintf(stdout, "[DEVICE #%d/INFO]: %s\n", device_index,  text.c_str());
    fflush(stdout);
}

void error(int32_t device_index, const std::string &text) {
    std::lock_guard<std::mutex> stderr_guard(stderr_mutex);
    fprintf(stderr, "[DEVICE #%d/ERROR]: %s\n", device_index,  text.c_str());
    fflush(stderr);
}

void manage_device(int32_t device_index) {
    cudaSetDevice(device_index);

    // Approximate the number of threads that has to be launched per wave for maximum performance:
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_index);
    int32_t pref_wave_size = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    int32_t max_block_count = pref_wave_size / BLOCK_SIZE;

    info(device_index, "Blocks per wave: " + std::to_string(max_block_count));

    // Found items are collected in video memory.
    // Any collected items will be transferred to the host in-between work unit executions.
    uint64_t *d_collector_size;
    Cluster *d_collector;
    cudaMalloc(&d_collector_size, sizeof(uint64_t));
    cudaMalloc(&d_collector, MAX_COLLECTOR_SIZE * sizeof(Cluster));
    
    // Offset is also accessed from within the while condition.
    offset_mutex.lock();
    while(offset < WORK_SIZE) {
        // Reset caches:
        cudaMemset(d_collector_size, 0, sizeof(uint64_t));

        // Launch one wave (one block does one seed):
        uint64_t block_count = WORK_SIZE - offset;
        if(block_count > max_block_count)
            block_count = max_block_count;
        kernel<<<block_count, BLOCK_SIZE>>>(block_count, offset, d_collector_size, d_collector);
        offset += block_count;
        offset_mutex.unlock();

        // Synchronize:
        cudaDeviceSynchronize();

        // Catch errors:
        cudaError_t code;
        if((code = cudaGetLastError()) != cudaSuccess) {
            error(device_index, cudaGetErrorString(code));
            offset_mutex.lock();
            break;
        }

        // Dump the collector:
        uint64_t collector_size;
        cudaMemcpy(&collector_size, d_collector_size, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        std::vector<Cluster> clusters(collector_size);
        cudaMemcpy(clusters.data(), d_collector, collector_size * sizeof(Cluster), cudaMemcpyDeviceToHost);

        auto end = clusters.end();
        for(auto cluster = clusters.begin(); cluster != end; cluster++) {
            // Remove all other duplicates of this cluster:
            end = std::remove(cluster + 1, end, *cluster);

            std::string cluster_info = cluster->to_string();
            log(cluster_info);
            info(device_index, cluster_info);

            found_total++;
        }

        offset_mutex.lock();
    }
    offset_mutex.unlock();

    cudaFree(d_collector);
    cudaFree(d_collector_size);

    devices_running--;
}

uint64_t get_millis() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

int32_t main(int32_t argc, char **argv) {
    BOINC_OPTIONS options;

    boinc_options_defaults(options);
    options.multi_thread = true;
    options.multi_process = true;
    options.normal_thread_priority = true;
    boinc_init_options(&options);

    log_file = fopen(LOG_FILE, "w");

    // Launch host threads:
    std::vector<std::thread> threads(DEVICE_COUNT);
    for(int32_t i = 0; i < DEVICE_COUNT; i++)
        threads[i] = std::thread(manage_device, i);

    // Monitor progress:
    uint64_t start_time = get_millis();
    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(REPORT_DELAY));

        if(devices_running.load() == 0)
            break;

        uint64_t elapsed = get_millis() - start_time;
        double progress, speed, eta;
        {
            std::lock_guard<std::mutex> offset_guard(offset_mutex);
            
            progress = ((double)offset / WORK_SIZE) * 100;
            speed = offset / (elapsed * 0.001);
            eta = (WORK_SIZE - offset) / speed;
        }

        {
            std::lock_guard<std::mutex> stdout_guard(stdout_mutex);
            printf("%f %% - %llu found - %llu seeds/s - %llu s elapsed - ETA %llu s\n", progress, found_total.load(), (uint64_t)speed, elapsed / 1000, (uint64_t)eta);
            fflush(stdout);
        }
    }

    // Join host threads:
    for(std::thread &thread : threads)
        thread.join();

    fclose(log_file);

    uint64_t found_snapshot = found_total.load();
    printf("Finished, %llu cluster%s found. (%llu s)\n", found_snapshot, found_snapshot == 1 ? " was" : "s were", (uint64_t)((get_millis() - start_time) * 0.001));
    
    return boinc_finish(0);
}