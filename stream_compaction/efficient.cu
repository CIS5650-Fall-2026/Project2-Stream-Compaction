#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        namespace {

            const int BLOCK_SIZE = 64;

            int nextPow2(int n) {
                int m = 1;
                while (m < n) {
                    m <<= 1;
                }
                return m;
            }

        }  // namespace

        /**
         * Up-sweep phase of the Blelloch scan: builds the inclusive-scan tree.
         * offset is the distance between the pair of tree nodes merged at this
         * level (1, 2, 4, ...).
         */
        __global__ void kernEfficientUpSweep(int m, int offset, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int active = m / (2 * offset);
            if (index < active) {
                int idx = (index + 1) * (2 * offset) - 1;
                if (idx < m) {
                    data[idx] += data[idx - offset];
                }
            }
        }

        /**
         * Down-sweep phase of the Blelloch scan: turns the inclusive tree into
         * an exclusive prefix sum. offset halves each level (m/2, m/4, ...).
         */
        __global__ void kernEfficientDownSweep(int m, int offset, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int active = m / (2 * offset);
            if (index < active) {
                int node0 = (2 * index + 1) * offset - 1;
                int node1 = node0 + offset;
                if (node1 < m) {
                    data[node1] += data[node0];
                    data[node0] = data[node1] - data[node0];
                }
            }
        }

        __global__ void kernEfficientSetLastZero(int m, int *data) {
            if (m > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
                data[m - 1] = 0;
            }
        }

        /**
         * Runs the work-efficient exclusive scan in place on a device array of
         * length m, where m is a power of two. Values in the array are the
         * "input"; padded slots beyond the logical length must already be 0.
         *
         * This helper intentionally does not touch the timer: compact() reuses
         * it while its own GPU timer is already running.
         */
        static void scanDevice(int m, int *data) {
            if (m <= 0) {
                return;
            }
            if (m == 1) {
                data[0] = 0;
                return;
            }

            for (int offset = 1; offset < m; offset <<= 1) {
                int active = m / (2 * offset);
                int blocks = (active + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernEfficientUpSweep<<<blocks, BLOCK_SIZE>>>(m, offset, data);
            }

            kernEfficientSetLastZero<<<1, 1>>>(m, data);

            for (int offset = m / 2; offset > 0; offset >>= 1) {
                int active = m / (2 * offset);
                int blocks = (active + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernEfficientDownSweep<<<blocks, BLOCK_SIZE>>>(m, offset, data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int m = nextPow2(n);
            int *devData = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&devData), m * sizeof(int));
            cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (m > n) {
                cudaMemset(devData + n, 0, (m - n) * sizeof(int));
            }

            timer().startGpuTimer();
            scanDevice(m, devData);
            timer().endGpuTimer();

            cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devData);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int m = nextPow2(n);
            const int blockSize = 128;
            const dim3 fullBlocks((n + blockSize - 1) / blockSize);

            int *devIData = nullptr;
            int *devBools = nullptr;
            int *devIndices = nullptr;
            int *devOdata = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&devIData), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devBools), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devIndices), m * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devOdata), n * sizeof(int));

            cudaMemcpy(devIData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (m > n) {
                cudaMemset(devIndices + n, 0, (m - n) * sizeof(int));
            }

            timer().startGpuTimer();

            // Map 1/0 keep/remove values into two buffers: one stays as the
            // scatter predicate and the other is overwritten by the scan.
            Common::kernMapToBoolean<<<fullBlocks, blockSize>>>(n, devBools, devIData);
            Common::kernMapToBoolean<<<fullBlocks, blockSize>>>(n, devIndices, devIData);

            scanDevice(m, devIndices);

            Common::kernScatter<<<fullBlocks, blockSize>>>(
                n, devOdata, devIData, devBools, devIndices);

            timer().endGpuTimer();

            int lastIndex = 0;
            cudaMemcpy(&lastIndex, devIndices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + ((idata[n - 1] != 0) ? 1 : 0);
            cudaMemcpy(odata, devOdata, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(devIData);
            cudaFree(devBools);
            cudaFree(devIndices);
            cudaFree(devOdata);
            return count;
        }
    }
}
