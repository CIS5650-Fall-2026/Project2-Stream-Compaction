#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "shared.h"


namespace StreamCompaction {
	namespace Shared {

		using StreamCompaction::Common::PerformanceTimer;

		const int NUM_BANKS = 32;
		// const int LOG_NUM_BANKS = 5;


		__host__ __device__ int conflictFreeOffset(int index) {

			return index / NUM_BANKS;
		}


		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}


		__global__ void kernScanNaive(int n, int* odata, const int* idata) {

			extern __shared__ int temp[];


			int thid = threadIdx.x;
			int pout = 0;
			int pin = 1;


			// Shift input right by one to make scan exclusive
			temp[pout * n + thid] = (thid > 0) ? idata[thid - 1] : 0;

			__syncthreads();


			// Double-buffered naive scan in shared memory
			for (int offset = 1; offset < n; offset *= 2) {

				pout = 1 - pout;
				pin = 1 - pout;


				if (thid >= offset) {

					temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
				}
				else {
					temp[pout * n + thid] = temp[pin * n + thid];
				}

				__syncthreads();
			}

			odata[thid] = temp[pout * n + thid];
		}

		__global__ void kernScanWorkEfficient(int n, int* odata, const int* idata) {

			extern __shared__ int temp[];

			int thid = threadIdx.x;

			int offset = 1;


			// Each thread loads two elements into shared memory
			int ai = 2 * thid;
			int bi = 2 * thid + 1;


			temp[ai] = idata[ai];
			temp[bi] = idata[bi];


			__syncthreads();


			// Upsweep
			for (int d = n >> 1; d > 0; d >>= 1) {

				if (thid < d) {

					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;


					temp[bi] += temp[ai];
				}


				offset *= 2;

				__syncthreads();
			}


			// Clear root to make result exclusive
			if (thid == 0) {
				temp[n - 1] = 0;
			}

			__syncthreads();


			// Downsweep
			for (int d = 1; d < n; d *= 2) {

				offset >>= 1;


				if (thid < d) {

					int ai = offset * (2 * thid + 1) - 1;

					int bi = offset * (2 * thid + 2) - 1;


					int tempValue = temp[ai];


					temp[ai] = temp[bi];
					temp[bi] += tempValue;


				}

				__syncthreads();
			}

			odata[ai] = temp[ai];

			odata[bi] = temp[bi];
		}


		void scanNaive(int n, int* odata, const int* idata) {

			int* dev_idata;
			int* dev_odata;


			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));



			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);


			int sharedBytes = 2 * n * sizeof(int);


			timer().startGpuTimer();

			kernScanNaive << <1, n, sharedBytes >> > (
				n, dev_odata, dev_idata
		    );

			timer().endGpuTimer();


			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(dev_idata);
			cudaFree(dev_odata);
		}



		void scanWorkEfficient(int n, int* odata, const int* idata) {

			int* dev_idata;

			int* dev_odata;


			cudaMalloc((void**)&dev_idata, n * sizeof(int));

			cudaMalloc((void**)&dev_odata, n * sizeof(int));


			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);


			int sharedBytes = n * sizeof(int);


			timer().startGpuTimer();

			kernScanWorkEfficient << <1, n / 2, sharedBytes >> > (
				n, dev_odata, dev_idata
		    );

			timer().endGpuTimer();


			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(dev_idata);
			cudaFree(dev_odata);
		}


		__global__ void kernScanWorkEfficientBankConflictFree(int n, int* odata, const int* idata) {

			extern __shared__ int temp[];


			int thid = threadIdx.x;

			int offset = 1;


			// Load from separate halves and pad shared-memory indices
			int ai = thid;

			int bi = thid + (n / 2);


			int bankOffsetA = conflictFreeOffset(ai);
			int bankOffsetB = conflictFreeOffset(bi);


			temp[ai + bankOffsetA] = idata[ai];
			temp[bi + bankOffsetB] = idata[bi];


			__syncthreads();


			// Upsweep with conflict-free shared-memory indices
			for (int d = n >> 1; d > 0; d >>= 1) {

				if (thid < d) {

					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;


					ai += conflictFreeOffset(ai);
					bi += conflictFreeOffset(bi);


					temp[bi] += temp[ai];
				}


				offset *= 2;

				__syncthreads();


			}

			// Clear padded root to make result exclusive
			if (thid == 0) {
				int rootIndex = n - 1;


				int rootOffset = conflictFreeOffset(rootIndex);


				temp[rootIndex + rootOffset] = 0;
			}

			__syncthreads();


			// Downsweep with conflict-free shared-memory indices
			for (int d = 1; d < n; d *= 2) {
				offset >>= 1;


				if (thid < d) {
					int ai = offset * (2 * thid + 1) - 1;

					int bi = offset * (2 * thid + 2) - 1;


					ai += conflictFreeOffset(ai);
					bi += conflictFreeOffset(bi);


					int tempValue = temp[ai];


					temp[ai] = temp[bi];

					temp[bi] += tempValue;
				}

				__syncthreads();
			}



			// Write results back using saved padded offsets
			odata[ai] = temp[ai + bankOffsetA];

			odata[bi] = temp[bi + bankOffsetB];
		}


		void scanWorkEfficientBankConflictFree(int n, int* odata, const int* idata) {

			int* dev_idata;
			int* dev_odata;



			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));


			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);


			int paddedSize = n + conflictFreeOffset(n - 1);
			int sharedBytes = paddedSize * sizeof(int);


			timer().startGpuTimer();

			kernScanWorkEfficientBankConflictFree << <1, n / 2, sharedBytes >> > (
				n, dev_odata, dev_idata
			);

			timer().endGpuTimer();


			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
		}
	}
}