#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * One doubling step of the naive (Hillis-Steele) inclusive scan.
         * Reads from `in` and writes the next iteration into `out`.
         */
        __global__ void kernNaiveScanStep(int n, int offset, int *out, const int *in) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < n) {
                out[index] = (index >= offset) ? in[index] + in[index - offset] : in[index];
            }
        }

        /**
         * Converts an inclusive prefix sum into an exclusive one:
         * out[0] = 0, out[i] = in[i - 1].
         */
        __global__ void kernExclusiveShift(int n, int *out, const int *in) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index == 0) {
                out[0] = 0;
            } else if (index < n) {
                out[index] = in[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            const int blockSize = 512;
            const dim3 fullBlocks((n + blockSize - 1) / blockSize);

            int *devA = nullptr;
            int *devB = nullptr;
            int *devOut = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&devA), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devB), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devOut), n * sizeof(int));
            cudaMemcpy(devA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Double-buffered naive inclusive scan.  Each iteration doubles
            // the span of elements folded into each output position.
            int *src = devA;
            int *dst = devB;
            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScanStep<<<fullBlocks, blockSize>>>(n, offset, dst, src);
                int *tmp = src;
                src = dst;
                dst = tmp;
            }

            // Shift the inclusive result by one to produce the exclusive scan.
            kernExclusiveShift<<<fullBlocks, blockSize>>>(n, devOut, src);

            timer().endGpuTimer();

            cudaMemcpy(odata, devOut, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devOut);
        }
    }
}
