#include "helper.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

// aprox 1 mil
#define MSIZE (1 << 16)

__global__ void pop_in_range(float *lat, float *lon, int *pop, int *accpop,
                             int kmrange, int n) {
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  for (int j = 0; j < n; j++) {
    float phi1 = (90.f - lat[i]) * DEGREE_TO_RADIANS;
    float phi2 = (90.f - lat[j]) * DEGREE_TO_RADIANS;

    float theta1 = lon[i] * DEGREE_TO_RADIANS;
    float theta2 = lon[j] * DEGREE_TO_RADIANS;

    float cs =
        sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2);
    if (cs > 1) {
      cs = 1;
    } else if (cs < -1) {
      cs = -1;
    }
    if ((6371.f * acos(cs)) <= kmrange) {
      accpop[i] += pop[j];
    }
  }
}

int main(int argc, char **argv) {
  DIE(argc == 1, "./accpop <kmrange1> <file1in> <file1out> ...");
  DIE((argc - 1) % 3 != 0, "./accpop <kmrange1> <file1in> <file1out> ...");

  for (int argcID = 1; argcID < argc; argcID += 3) {
    float kmRange = atof(argv[argcID]);
    string geon;

    float *lat = (float *)malloc(MSIZE * sizeof(float));
    float *lon = (float *)malloc(MSIZE * sizeof(float));
    int *pop = (int *)malloc(MSIZE * sizeof(int));

    ifstream ifs(argv[argcID + 1]);
    ofstream ofs(argv[argcID + 2]);

    int i = 0;
    while (ifs >> geon >> lat[i] >> lon[i] >> pop[i]) {
      i++;
    }
    int nrOfElements = i;
    int *accpop = (int *)malloc(nrOfElements * sizeof(int));

    float *device_lat = 0;
    float *device_lon = 0;
    int *device_pop = 0, *device_accpop = 0;

    // Allocating the device's array; notice that we use a special
    // function named cudaMalloc that takes the reference of the
    // pointer declared above and the number of bytes.
    cudaMalloc((void **)&device_lat, nrOfElements * sizeof(float));
    cudaMalloc((void **)&device_lon, nrOfElements * sizeof(float));
    cudaMalloc((void **)&device_pop, nrOfElements * sizeof(int));
    cudaMalloc((void **)&device_accpop, nrOfElements * sizeof(int));

    cudaMemcpy(device_lat, lat, nrOfElements * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_lon, lon, nrOfElements * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_pop, pop, nrOfElements * sizeof(float),
               cudaMemcpyHostToDevice);

    const size_t block_size = 256;
    size_t blocks_no = nrOfElements / block_size;

    if (nrOfElements % block_size)
      ++blocks_no;

    // Launch the kernel
    pop_in_range<<<blocks_no, block_size>>>(device_lat, device_lon, device_pop,
                                            device_accpop, kmRange,
                                            nrOfElements);

    // Copy the result back to the host memory space
    cudaMemcpy(accpop, device_accpop, nrOfElements * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (i = 0; i < nrOfElements; i++) {
      ofs << accpop[i] << endl;
    }

    // Deallocate memory
    free(lat);
    free(lon);
    free(pop);
    free(accpop);
    cudaFree(device_lat);
    cudaFree(device_lon);
    cudaFree(device_pop);
    cudaFree(device_accpop);

    ifs.close();
    ofs.close();
  }
}
