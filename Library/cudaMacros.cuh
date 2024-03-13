#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuGlobalErrorCheck() { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }
#define gpuDebugF(var) { printf("%f", var); printf("\n");}
#define gpuDebugArrF(var, z) { for (int i = 0; i < z; ++i) {printf("%f ", var[i]);} printf("\n");}


void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);
