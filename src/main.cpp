/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <algorithm>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/shared.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
const int BENCHMARK_SIZE = 1000000;
const int BENCHMARK_TRIALS = 5;
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];
int* benchmarkInput = new int[BENCHMARK_SIZE];
int* benchmarkOutput = new int[BENCHMARK_SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("shared naive scan, power-of-two");
    StreamCompaction::Shared::scanNaive(SIZE, c, a);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("shared work-efficient scan, power-of-two");
    StreamCompaction::Shared::scanWorkEfficient(SIZE, c, a);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("shared work-efficient bank-conflict-free scan, power-of-two");
    StreamCompaction::Shared::scanWorkEfficientBankConflictFree(SIZE, c, a);
    printElapsedTime(
        StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(),
        "(CUDA Measured)"
    );
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);



    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    // Generate nonnegative values for the radix sort test
    genArray(SIZE, a, 50);

    // Use std::sort as the CPU reference
    for (int i = 0; i < SIZE; i++) {
        b[i] = a[i];
    }
    std::sort(b, b + SIZE);

    zeroArray(SIZE, c);
    printDesc("radix sort, power-of-two");
    StreamCompaction::Radix::sort(SIZE, c, a);

    printCmpResult(SIZE, b, c);



    // Test radix sort with a non-power-of-two input size
    for (int i = 0; i < NPOT; i++) {
        b[i] = a[i];
    }

    std::sort(b, b + NPOT);

    zeroArray(SIZE, c);
    printDesc("radix sort, non-power-of-two");
    StreamCompaction::Radix::sort(NPOT, c, a);

    printCmpResult(NPOT, b, c);


    // Fixed test with duplicate keys
    const int DUP_SIZE = 7;
    int dupInput[DUP_SIZE] = { 7, 2, 7, 0, 2, 5, 0 };
    int dupExpected[DUP_SIZE] = { 0, 0, 2, 2, 5, 7, 7 };
    int dupOutput[DUP_SIZE] = {};

    printDesc("radix sort, duplicates");
    StreamCompaction::Radix::sort(DUP_SIZE, dupOutput, dupInput);

    printCmpResult(DUP_SIZE, dupExpected, dupOutput);


    printf("\n");
    printf("************************\n");
    printf("** SCAN PERFORMANCE **\n");
    printf("************************\n");

    genArray(BENCHMARK_SIZE, benchmarkInput, 50);

    float cpuTotal = 0.0f;
    float naiveTotal = 0.0f;
    float efficientTotal = 0.0f;
    float thrustTotal = 0.0f;

    zeroArray(BENCHMARK_SIZE, benchmarkOutput);

    printDesc("benchmark cpu scan");

    for (int trial = 0; trial < BENCHMARK_TRIALS; trial++) {
        StreamCompaction::CPU::scan(
            BENCHMARK_SIZE,
            benchmarkOutput,
            benchmarkInput
        );

        cpuTotal += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }

    printf("   average elapsed time: %fms\n", cpuTotal / BENCHMARK_TRIALS);

    


    zeroArray(BENCHMARK_SIZE, benchmarkOutput);

    printDesc("benchmark naive scan");

    for (int trial = 0; trial < BENCHMARK_TRIALS; trial++) {
        StreamCompaction::Naive::scan(
            BENCHMARK_SIZE,
            benchmarkOutput,
            benchmarkInput
        );

        naiveTotal += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    }

    printf("   average elapsed time: %fms\n", naiveTotal / BENCHMARK_TRIALS);


    zeroArray(BENCHMARK_SIZE, benchmarkOutput);

    printDesc("benchmark work-efficient scan");

    for (int trial = 0; trial < BENCHMARK_TRIALS; trial++) {
        StreamCompaction::Efficient::scan(
            BENCHMARK_SIZE,
            benchmarkOutput,
            benchmarkInput
        );

        efficientTotal += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }

    printf("   average elapsed time: %fms\n", efficientTotal / BENCHMARK_TRIALS);


    zeroArray(BENCHMARK_SIZE, benchmarkOutput);

    printDesc("benchmark thrust scan");

    for (int trial = 0; trial < BENCHMARK_TRIALS; trial++) {
        StreamCompaction::Thrust::scan(
            BENCHMARK_SIZE,
            benchmarkOutput,
            benchmarkInput
        );

        thrustTotal += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    }

    printf("   average elapsed time: %fms\n", thrustTotal / BENCHMARK_TRIALS);


    delete[] benchmarkInput;
    delete[] benchmarkOutput;


    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
