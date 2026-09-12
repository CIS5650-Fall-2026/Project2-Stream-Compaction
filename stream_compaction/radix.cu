#include <cuda.h>
#include <cuda_runtime.h>


#include "common.h"
#include "efficient.h"
#include "radix.h"


namespace StreamCompaction {
	namespace Radix {



		// Mark elements whose current bit is 0 for the scan-based split
		__global__ void kernMapBitToZero(int n, int bit, int* zeros, const int* idata) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index >= n) {
				return;
			}


			int bitValue = (idata[index] >> bit) & 1;

			zeros[index] = (bitValue == 0) ? 1 : 0;
		}



		// Scatter zeros first and ones after them while preserving their relative order
		__global__ void kernRadixScatter(int n, int bit, int totalFalses, int* odata, const int* idata, const int* indices) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);


			if (index >= n) {
				return;
			}


			int bitValue = (idata[index] >> bit) & 1;


			if (bitValue == 0) {
				odata[indices[index]] = idata[index];
			}
			else {

				int oneIndex = index - indices[index];

				odata[totalFalses + oneIndex] = idata[index];
			}
		}



		void sort(int n, int* odata, const int* idata) {

			if (n <= 0) {
				return;
			}

			int levels = ilog2ceil(n);


			// Pad scan arrays to the next power of two
			int paddedN = 1 << levels;



			int* dev_in;
			int* dev_out;
			int* dev_zeros;
			int* dev_indices;




			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));



			cudaMalloc((void**)&dev_zeros, paddedN * sizeof(int));
			cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));


			// Copy unsorted input to GPU
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);




			int blockSize = 128;
			int blocksPerGrid = (n + blockSize - 1) / blockSize;


			// Process one bit at a time from LSB to MSB

			for (int bit = 0; bit < 32; bit++) {

				// Zero padded scan storage before building this pass's flags
				cudaMemset(dev_zeros, 0, paddedN * sizeof(int));
				cudaMemset(dev_indices, 0, paddedN * sizeof(int));


				// e = !b....mark elements whose current bit is 0
				kernMapBitToZero << <blocksPerGrid, blockSize >> > (
					n, bit, dev_zeros, dev_in
			    );

				// Copy e into f, then exclusive scan f in place
				cudaMemcpy(dev_indices, dev_zeros, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);


				Efficient::scanDevice(paddedN, dev_indices);


				// totalFalses = e[n - 1] + f[n - 1]
				int lastZero;

				int lastIndex;

				cudaMemcpy(&lastZero, dev_zeros + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);


				cudaMemcpy(&lastIndex, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);


				int totalFalses = lastZero + lastIndex;


				// Scatter according to split addresses for this bit
				kernRadixScatter << <blocksPerGrid, blockSize >> > (
					n, bit, totalFalses, dev_out, dev_in, dev_indices
			    );


				// Use this pass's output as the input for the next bit
				int* temp = dev_in;
				dev_in = dev_out;
				dev_out = temp;


			}

			// Copy final sorted result back to the host
			cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(dev_zeros);
			cudaFree(dev_indices);
		}
	}
}