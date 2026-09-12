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

        // Going to add helpers though there is no TODO here..
        // Seems the starter only leaves TODOs inside scan() and compact()...



        // Build partial sums up the reduction tree
        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);


            int stride = 1 << (d + 1);

            int right = (index + 1) * stride - 1;


            if (right < n) {
                int left = right - (stride >> 1);
                data[right] += data[left];
            }
        }



        // Propagate prefix sums back down the tree
        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);


            int stride = 1 << (d + 1);


            int right = (index + 1) * stride - 1;


            if (right < n) {
                int left = right - (stride >> 1);


                int temp = data[left];
                data[left] = data[right];
                data[right] += temp;
            }
        }


        // Set the root to 0 before starting downsweep
        __global__ void kernSetZero(int n, int* data) {

            if (threadIdx.x == 0 && blockIdx.x == 0) {

                data[n - 1] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // timer().startGpuTimer();
            // TODO



            int levels = ilog2ceil(n);

            // Pad the working array to the next power of two
            int paddedN = 1 << levels;

            int* dev_data;

            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));


            // Initialize padded region to 0
            cudaMemset(dev_data, 0, paddedN * sizeof(int));

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int blockSize = 64;
            // int blocksPerGrid = (paddedN + blockSize - 1) / blockSize;


            //// Upsweep: build partial sums toward the root
            //for (int d = 0; d < levels; d++) {

            //    kernUpSweep << <blocksPerGrid, blockSize >> > (
            //        paddedN, d, dev_data
            //    );
            //}

            // Attempted extra credit Upsweep: launch only threads needed at this tree level
            for (int d = 0; d < levels; d++) {

                int activeThreads = paddedN >> (d + 1);

                int activeBlocks = (activeThreads + blockSize - 1) / blockSize;


                kernUpSweep << <activeBlocks, blockSize >> > (
                    paddedN, d, dev_data
                );
            }

            // Set the root to 0 before the downsweep
            kernSetZero << <1, 1 >> > (
                paddedN, dev_data
            );


            //// Downsweep: propagate prefix sums back down the tree
            //for (int d = levels - 1; d >= 0; d--) {

            //    kernDownSweep << <blocksPerGrid, blockSize >> > (
            //        paddedN, d, dev_data
            //    );
            //}

            // Attempted extra credit Downsweep: launch only threads needed at this tree level
            for (int d = levels - 1; d >= 0; d--) {

                int activeThreads = paddedN >> (d + 1);

                int activeBlocks = (activeThreads + blockSize - 1) / blockSize;


                kernDownSweep << <activeBlocks, blockSize >> > (
                    paddedN, d, dev_data
                );
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
        }

        // Perform same work-efficient exclusive scan on data already on GPU
        void scanDevice(int n, int* dev_data) {
            
            int levels = ilog2ceil(n);

            int blockSize = 128;


            // Upsweep: build partial sums toward root
            for (int d = 0; d < levels; d++) {

                int activeThreads = n >> (d + 1);

                int activeBlocks = (activeThreads + blockSize - 1) / blockSize;


                kernUpSweep << <activeBlocks, blockSize >> > (
                    n, d, dev_data
                );
            }

            // Set root to 0 to make result an exclusive scan
            kernSetZero << <1, 1 >> > (
                n, dev_data
            );

            // Downsweep: propagate prefix sums back down the tree
            for (int d = levels - 1; d >= 0; d--) {

                int activeThreads = n >> (d + 1);

                int activeBlocks = (activeThreads + blockSize - 1) / blockSize;


                kernDownSweep << <activeBlocks, blockSize >> > (
                    n, d, dev_data
                );
            }


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
            // timer().startGpuTimer();
            // TODO


            int levels = ilog2ceil(n);

            int paddedN = 1 << levels;


            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;


            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));


            // Zero the padded regions used by the scan
            cudaMemset(dev_bools, 0, paddedN * sizeof(int));
            cudaMemset(dev_indices, 0, paddedN * sizeof(int));


            // Copy input array to GPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);



            int blockSize = 64;
            int blocksPerGrid = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // Step 1: Map input elements to keep flags
            Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (
                n, dev_bools, dev_idata
            );

            // Preserve the keep flags and scan a separate copy in place
            cudaMemcpy(dev_indices, dev_bools, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);

            
            // Step 2a: upsweep boolean flags into partial sums
            // int scanBlocksPerGrid = (paddedN + blockSize - 1) / blockSize;

            //for (int d = 0; d < levels; d++) {
            //    kernUpSweep << <scanBlocksPerGrid, blockSize >> > (
            //        paddedN, d, dev_indices
            //    );
            //}


            // Attempted extra credit Step 2a: upsweep boolean flags into partial sums
            for (int d = 0; d < levels; d++) {

                int activeThreads = paddedN >> (d + 1);

                int activeBlocks = (activeThreads + blockSize - 1) / blockSize;


                kernUpSweep << <activeBlocks, blockSize >> > (
                    paddedN, d, dev_indices
                );
            }
            

            // Set the root to 0 before starting downsweep
            kernSetZero << <1, 1 >> > (
                paddedN, dev_indices
            );

            //// Step 2b: downsweep to produce exclusive scan indices
            //for (int d = levels - 1; d >= 0; d--) {
            //    kernDownSweep << <scanBlocksPerGrid, blockSize >> > (
            //        paddedN, d, dev_indices
            //    );
            //}


            // Attempted extra credit Step 2b: Downsweep to produce exclusive scan indices
            for (int d = levels - 1; d >= 0; d--) {

                int activeThreads = paddedN >> (d + 1);

                int activeBlocks = (activeThreads + blockSize - 1) / blockSize;


                kernDownSweep << <activeBlocks, blockSize >> > (
                    paddedN, d, dev_indices
                );
            }

            // Step 3: scatter kept elements into compacted positions
            Common::kernScatter << <blocksPerGrid, blockSize >> > (
                n, dev_odata, dev_idata, dev_bools, dev_indices
            );



            timer().endGpuTimer();




            // Get number of elements remaining after compaction
            int lastIndex;
            int lastBool;

            cudaMemcpy(&lastIndex, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(&lastBool, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int count = lastIndex + lastBool;


            // Copy compacted output back to host
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);


            // return -1;
            return count;
        }
    }
}
