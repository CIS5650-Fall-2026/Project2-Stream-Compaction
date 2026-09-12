#pragma once

#include "common.h"


namespace StreamCompaction {
	namespace Shared {

		StreamCompaction::Common::PerformanceTimer& timer();



		void scanNaive(int n, int *odata, const int *idata);

		void scanWorkEfficient(int n, int *odata, const int *idata);

		void scanWorkEfficientBankConflictFree(int n, int *odata, const int *idata);

	}
}