#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <bitset>

#include "cuda.h"
#include "cluster.h"
#include "lcg.h"
#include "random.h"
#include "java_random.h"
#include "bit_field.h"

// How many compute devices to use:
#define DEVICE_COUNT 1

// Keep at 50 % of the capable value for best results:
#define BLOCK_SIZE 512

// Where to start and how far to explore:
#define WORK_OFFSET 0
#define WORK_SIZE 1 // (1UI64 << 16)

// Size of the collector array, or how many items are expected to be found in a single work unit:
#define MAX_COLLECTOR_SIZE (1 << 16)

#define REPORT_DELAY 1000
#define LOG_FILE "clusters.txt"

#define EXTENTS 128
#define MIN_CLUSTER_SIZE 6

#define UINT64_BITS (sizeof(uint64_t) * 8)

const size_t CACHE_SIZE_BITS = (EXTENTS * 2UI64) * (EXTENTS * 2UI64);
const size_t CACHE_SIZE_UINT64 = ((CACHE_SIZE_BITS + (UINT64_BITS - 1)) / UINT64_BITS);

CUDA bool check_slime_chunk(JavaRandom *rand, uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
	world_seed += chunk_x * chunk_x * 0x4C1906;
	world_seed += chunk_x * 0x5AC0DB;
	world_seed += chunk_z * chunk_z * 0x4307A7UI64;
	world_seed += chunk_z * 0x5F24F;
	world_seed ^= 0x3AD8025FUI64;

	// rand->set_seed(world_seed);
	// rand->scramble();
	// return rand->next_int(10) == 0;
	return ((uint64_t)((((world_seed ^ 0x5DEECE66DUI64) & ((1UI64 << 48) - 1)) * 0x5DEECE66DUI64 + 0xB) & ((1UI64 << 48) - 1)) >> 17) % 10 == 0;
}

CUDA int32_t find_clusters(JavaRandom *rand, BitField *cache, uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
	// Map two-dimensional position to a one-dimensional index:
	uint64_t cache_idx = (chunk_x + EXTENTS) * (EXTENTS * 2UI64) + (chunk_z + EXTENTS);

	// We don't want to process chunks multiple times:
	if(cache->get(cache_idx))
		return 0;

	cache->set(cache_idx, true);

	if(!check_slime_chunk(rand, world_seed, chunk_x, chunk_z))
		return 0;

	int32_t count = 1;

	// Process neighboring chunks:
	if(chunk_x + 1 < EXTENTS)
		count += find_clusters(rand, cache, world_seed, chunk_x + 1, chunk_z);
	if(chunk_x - 1 >= -EXTENTS)
		count += find_clusters(rand, cache, world_seed, chunk_x - 1, chunk_z);
	if(chunk_z + 1 < EXTENTS)
		count += find_clusters(rand, cache, world_seed, chunk_x, chunk_z + 1);
	if(chunk_z - 1 >= -EXTENTS)
		count += find_clusters(rand, cache, world_seed, chunk_x, chunk_z - 1);

	return count;
}

// Fails for higher distances from (0, 0). (only on GPU) What might be the case?
__global__ void kernel(uint64_t work_item_count, uint64_t offset, uint64_t *caches, uint64_t *collector_size, Cluster *collector) {
	uint64_t local_index = threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
	if(local_index >= work_item_count)
		return;

	BitField *cache = BitField::wrap(caches + (local_index * CACHE_SIZE_UINT64), CACHE_SIZE_BITS);
	JavaRandom *rand = new JavaRandom();

	uint64_t world_seed = local_index + offset + WORK_OFFSET;

	for(int32_t chunk_x = -EXTENTS; chunk_x < EXTENTS; chunk_x++) {
		for(int32_t chunk_z = -EXTENTS; chunk_z < EXTENTS; chunk_z++) {
			int32_t size = find_clusters(rand, cache, world_seed, chunk_x, chunk_z);
			if(size >= MIN_CLUSTER_SIZE) {
				uint64_t collector_index = atomicAdd(collector_size, 1);
    			collector[collector_index] = Cluster(world_seed, chunk_x, chunk_z, size);
			}
		}
	}

	delete rand;
	delete cache;
}

FILE *log_file;
uint64_t offset = 0;
std::atomic<uint64_t> found_total(0);
std::atomic<int32_t> devices_running(DEVICE_COUNT);
std::mutex stdout_mutex, stderr_mutex, log_file_mutex, offset_mutex;

void manage_device(int32_t device_index) {
	cudaSetDevice(device_index);

	// Calculate the maximum number of threads per kernel execution:
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_index);
	int32_t mp_count = prop.multiProcessorCount;
	int32_t mp_size = prop.maxThreadsPerMultiProcessor;
	int32_t thread_limit = mp_count * mp_size;
	printf("COUNT: %d + SIZE: %d = LIMIT: %d\n", mp_count, mp_size, thread_limit);

	// Found items are collected in the video memory.
	// Any collected items will be transferred to the host in-between work unit executions.
	uint64_t *d_caches;
	uint64_t *d_collector_size;
	Cluster *d_collector;
	cudaMalloc(&d_caches, thread_limit * CACHE_SIZE_UINT64 * sizeof(uint64_t));
	cudaMalloc(&d_collector_size, sizeof(uint64_t));
	cudaMalloc(&d_collector, MAX_COLLECTOR_SIZE * sizeof(Cluster));
	
	offset_mutex.lock();
	while(offset < WORK_SIZE) {
		cudaMemset(d_caches, 0, thread_limit * CACHE_SIZE_UINT64 * sizeof(uint64_t));
		cudaMemset(d_collector_size, 0, sizeof(uint64_t));

		// DEBUG: Print caches:

		// uint64_t *caches = new uint64_t[thread_limit * CACHE_SIZE_UINT64];
		// cudaMemcpy(caches, d_caches, 1 * CACHE_SIZE_UINT64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		// for(int i = 0; i < CACHE_SIZE_UINT64; i++) {
		// 	std::cout << std::bitset<64>(caches[i]);
		// }
		// std::cout << std::endl;
		// delete[] caches;

		//

		uint64_t work_item_count = WORK_SIZE - offset;
		if(work_item_count > thread_limit)
			work_item_count = thread_limit;
		kernel<<<(work_item_count + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(work_item_count, offset, d_caches, d_collector_size, d_collector);
		offset += work_item_count;
		offset_mutex.unlock();

		cudaDeviceSynchronize();
		cudaError_t code;
		if((code = cudaGetLastError()) != cudaSuccess) {
			std::lock_guard<std::mutex> stderr_guard(stderr_mutex);
			fprintf(stderr, "[DEVICE #%d/ERROR]: %s\n", device_index, cudaGetErrorString(code));
			fflush(stderr);
			devices_running--;
			return;
		}

		uint64_t collector_size;
		cudaMemcpy(&collector_size, d_collector_size, sizeof(uint64_t), cudaMemcpyDeviceToHost);
		
		Cluster *cluster = new Cluster();
		for(uint64_t i = 0; i < collector_size; i++) {
			cudaMemcpy(cluster, d_collector + i, sizeof(Cluster), cudaMemcpyDeviceToHost);
			std::string cluster_info = cluster->to_string();
			{
				std::lock_guard<std::mutex> log_file_guard(log_file_mutex);
				fprintf(log_file, "%s\n", cluster_info.c_str());
				fflush(log_file);
			}
			{
				std::lock_guard<std::mutex> stdout_guard(stdout_mutex);
				printf("[DEVICE #%d/INFO]: %s\n", device_index, cluster_info.c_str());
				fflush(stdout);
			}
		}
		delete cluster;

		// DEBUG: Print caches:

		// caches = new uint64_t[thread_limit * CACHE_SIZE_UINT64];
		// cudaMemcpy(caches, d_caches, 1 * CACHE_SIZE_UINT64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		// for(int i = 0; i < CACHE_SIZE_UINT64; i++) {
		// 	std::cout << std::bitset<64>(caches[i]);
		// }
		// std::cout << std::endl;
		// delete[] caches;

		//

		found_total += collector_size;

		offset_mutex.lock();
	}
	offset_mutex.unlock();

	cudaFree(d_collector);
	cudaFree(d_collector_size);
	cudaFree(d_caches);

	devices_running--;
}

uint64_t get_millis() {
	auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

int32_t main(int32_t argc, char **argv) {
	log_file = fopen(LOG_FILE, "w");

	std::vector<std::thread> threads(DEVICE_COUNT);
	for(int32_t i = 0; i < DEVICE_COUNT; i++)
		threads[i] = std::thread(manage_device, i);

	uint64_t start_time = get_millis();
	uint64_t offset_snapshot;
	while(true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(REPORT_DELAY));

		if(devices_running.load() == 0)
			break;

		{
			std::lock_guard<std::mutex> offset_guard(offset_mutex);
			offset_snapshot = offset;
		}

		uint64_t elapsed = get_millis() - start_time;

		double progress = ((double)offset_snapshot / WORK_SIZE) * 100.0;
		double speed = offset_snapshot / (elapsed * 0.001);
		double eta = (WORK_SIZE - offset_snapshot) / speed;

		{
			std::lock_guard<std::mutex> stdout_guard(stdout_mutex);
			printf("%f %% - %llu found - %llu clusters/s - %llu s elapsed - ETA %llu s\n", progress, found_total.load(), (uint64_t)speed, (uint64_t)(elapsed * 0.001), (uint64_t)eta);
			fflush(stdout);
		}
	}

	for(std::thread &thread : threads)
		thread.join();

	fclose(log_file);

	uint64_t found_snapshot = found_total.load();
	printf("Finished, %llu cluster%s found. (%llu s)\n", found_snapshot, found_snapshot == 1 ? " was" : "s were", (uint64_t)((get_millis() - start_time) * 0.001));
	
	return 0;
}