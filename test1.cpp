#include <iostream>
#include <vector>
#include <openacc.h>
#include "SLISC/dense/cut.h"

using namespace slisc;

// Function to initialize the vectors with values
void initialize(std::vector<double>& a, std::vector<double>& b, int n) {
	for(int i = 0; i < n; ++i) {
		a[i] = static_cast<double>(i);
		b[i] = static_cast<double>(2 * i);
	}
}

// detect if GPU is actually running
void detect_gpu()
{
		double a[100], b[100];
		#pragma acc parallel loop
		for (int i = 0; i < 100; ++i) {
				if (i == 10) {
						if (acc_on_device(acc_device_not_host))
								printf("Executing on GPU.\n");
						else
								printf("Not executing on GPU.\n");
				}
				a[i] += b[i];
		}
}

// Main function
int main() {
	const int n = 1000000; // Size of the vectors
	std::vector<double> a(n), b(n), c(n);
	double *pa = a.data(), *pb = b.data(), *pc = c.data();

	// Initialize vectors a and b
	initialize(a, b, n);

	detect_gpu();

	Scmat3Comp Psi(1000, 1000, 100);
	for (Long i = 0; i < 1000; ++i)
		for (Long j = 0; j < 1000; ++j)
			for (Long k = 0; k < 100; ++k)
				Psi(i, j, k) = i + 1000*j + 0.01 * k;

	// Using OpenACC to offload the following computation to an accelerator
	// and explicitly handle data movement
	#pragma acc data copyin(pa[0:n], pb[0:n]) copyout(pc[0:n])
	{
	#pragma acc parallel loop
		for(int i = 0; i < n; ++i)
			pc[i] = pa[i] + pb[i];
	}

	// Display the first 10 results
	for(int i = 0; i < 10; ++i) {
		std::cout << "c[" << i << "] = " << c[i] << std::endl;
	}

	return 0;
}

