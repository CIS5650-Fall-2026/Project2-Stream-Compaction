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
        // TODO: __global__


        __global__ void kernScanStep(int n, int offset, int* odata, const int* idata) {

            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) {
                return;
            }



            // Each pass adds the value offset positions earlier
            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }



        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {

            int index = threadIdx.x + (blockIdx.x * blockDim.x);


            if (index >= n) {
                return;
            }



            // Shift the inclusive result right by one position
            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // timer().startGpuTimer();
            // TODO

            int* dev_a;
            int* dev_b;



            cudaMalloc((void**)&dev_a, n * sizeof(int));

            cudaMalloc((void**)&dev_b, n * sizeof(int));

            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);



            int blockSize = 64;
            int blocksPerGrid = (n + blockSize - 1) / blockSize;


            int* in = dev_a;
            int* out = dev_b;

            timer().startGpuTimer();


            for (int d = 0; d < ilog2ceil(n); d++) {

                int offset = 1 << d;

                kernScanStep << <blocksPerGrid, blockSize >> > (
                    n, offset, out, in
                );


                int* temp = in;
                in = out;
                out = temp;



            }

            // Convert inclusive result to exclusive scan
            kernInclusiveToExclusive << <blocksPerGrid, blockSize >> > (
                n, out, in
            );


            timer().endGpuTimer();

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}
