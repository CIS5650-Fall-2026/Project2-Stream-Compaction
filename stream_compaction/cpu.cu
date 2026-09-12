#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            // Serial exclusive scan used as the CPU baseline
            int sum = 0;

            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }



            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            // We keep track of the next open position in the compacted array
            int count = 0;

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            // return -1;
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            int* bools = new int[n];
            int* indices = new int[n];


            // Map each input element to 1 to keep it, or 0 otherwise
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }


            // Exclusive scan of the keep flags gives each element its output index
            int sum = 0;
            for (int i = 0; i < n; i++) {
                indices[i] = sum;
                sum += bools[i];
            }


            // Scatter each nonzero element to its position in the compacted array
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                }
            }



            delete[] bools;
            delete[] indices;
            timer().endCpuTimer();
            // return -1;
            return sum;
        }
    }
}
