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

#include "bit_field.h"
#include "cluster.h"
#include "cuda.h"
#include "java_random.h"
#include "lcg.h"
#include "random.h"
#include "stack.h"

// How many compute devices to use:
#define DEVICE_COUNT 1

// This can be tuned. (Doesn't influence device occupancy.)
// Or should we use cudaDeviceProp::maxBlocksPerMultiProcessor instead?
#define PREF_BLOCKS_PER_SM 8 

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
#define CACHE_EXTENTS 64

#define UINT64_BITS (sizeof(uint64_t) * 8)

const size_t CACHE_SIZE_BITS = (CACHE_EXTENTS * 2UI64) * (CACHE_EXTENTS * 2UI64);
const size_t CACHE_SIZE_UINT64 = ((CACHE_SIZE_BITS + (UINT64_BITS - 1)) / UINT64_BITS);

// Device code:

class Offset {
public:
    int8_t x, z;

    CUDA_CALL Offset() : Offset(0, 0) {}

    CUDA_CALL Offset(int8_t x, int8_t z) : x(x), z(z) {}
};

CUDA_CALL bool check_slime_chunk(JavaRandom &rand, uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
    world_seed += CUDA::wrapping_mul(CUDA::wrapping_mul(chunk_x, chunk_x), 0x4C1906);
    world_seed += CUDA::wrapping_mul(chunk_x, 0x5AC0DB);
    world_seed += CUDA::wrapping_mul(chunk_z, chunk_z) * 0x4307A7UI64;
    world_seed += CUDA::wrapping_mul(chunk_z, 0x5F24F);
    world_seed ^= 0x3AD8025FUI64;

    rand.set_seed(world_seed);
    rand.scramble();
    return rand.next_int(10) == 0;
}

CUDA_CALL Cluster explore_cluster(JavaRandom &rand, uint64_t world_seed, int32_t origin_x, int32_t origin_z, Offset *stack_buffer, uint64_t *cache_buffer) {
    memset(cache_buffer, 0, CACHE_SIZE_UINT64 * sizeof(uint64_t));
    BitField cache = BitField::wrap(cache_buffer, CACHE_SIZE_BITS);

    Stack<Offset> stack = Stack<Offset>::wrap(stack_buffer, STACK_SIZE);
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

        // We assume the new position is within cache bounds:
        if(chunk_x + 1 < EXTENTS)
            stack.push(Offset(offset.x + 1, offset.z));
        if(chunk_x - 1 >= -EXTENTS)
            stack.push(Offset(offset.x - 1, offset.z));
        if(chunk_z + 1 < EXTENTS)
            stack.push(Offset(offset.x, offset.z + 1));
        if(chunk_z - 1 >= -EXTENTS)
            stack.push(Offset(offset.x, offset.z - 1));
    }

    return Cluster(world_seed, cluster_size, min_x, min_z, max_x, max_z);
}

__global__ void kernel(uint64_t wave_size, uint64_t offset, uint64_t *collector_size, Cluster *collector) {
    uint64_t local_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(local_index >= wave_size)
        return;

    uint64_t world_seed = local_index + offset + WORK_OFFSET;

    JavaRandom rand = JavaRandom();

    // This should reside in registers?
    Offset stack_buffer[STACK_SIZE];
    uint64_t cache_buffer[CACHE_SIZE_UINT64];

    for(int32_t chunk_x = -EXTENTS; chunk_x < EXTENTS; chunk_x++) {
        for(int32_t chunk_z = -EXTENTS; chunk_z < EXTENTS; chunk_z++) {
            Cluster cluster = explore_cluster(rand, world_seed, chunk_x, chunk_z, stack_buffer, cache_buffer);
            if(cluster.get_size() >= MIN_CLUSTER_SIZE) {
                uint64_t collector_index = atomicAdd(collector_size, 1);
                collector[collector_index] = cluster;
            }
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

    info(device_index, "Preferred number of threads per wave: " + std::to_string(pref_wave_size));

    // Calculate optimal block size:
    int32_t block_size = prop.maxThreadsPerMultiProcessor / PREF_BLOCKS_PER_SM;
    if(block_size > prop.maxThreadsPerBlock)
        block_size = prop.maxThreadsPerBlock;

    // Round to the smaller multiple of warp size:
    block_size /= prop.warpSize;
    block_size *= prop.warpSize;

    info(device_index, "Threads per block: " + std::to_string(block_size));

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

        // Launch one wave:
        uint64_t wave_size = WORK_SIZE - offset;
        if(wave_size > pref_wave_size)
            wave_size = pref_wave_size;
        kernel<<<(wave_size + (block_size - 1)) / block_size, block_size>>>(wave_size, offset, d_collector_size, d_collector);
        offset += wave_size;
        offset_mutex.unlock();

        // Synchronize:
        cudaDeviceSynchronize();
        info(device_index, "Synchronized.");

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
            // Remove all duplicates of this cluster:
            end = std::remove(cluster + 1, end, *cluster);

            std::string cluster_info = cluster->to_string();
            log(cluster_info);
            info(device_index, cluster_info);
        }

        found_total += collector_size;

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
    
    return 0;
}