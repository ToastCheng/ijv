//#include <cutil.h> //cuda toolkit below 5.0 support for CUDA_SAFE_CALL()
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>
#include <memory>

using namespace std;

// DEFINES 
#define NUM_BLOCKS 5*16//20*16 //5*16 //dimGrid //Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)

//The register usage varies with platform. 64-bit Linux and 32.bit Windows XP have been tested.
#ifdef __linux__ //uses 25 registers per thread (64-bit)
	#define NUM_THREADS_PER_BLOCK 320 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS NUM_BLOCKS*NUM_THREADS_PER_BLOCK
#elif _WIN64
	#define NUM_THREADS_PER_BLOCK 256 //dimBlock
	#define NUM_THREADS NUM_BLOCKS*NUM_THREADS_PER_BLOCK
#else //uses 26 registers per thread
	#define NUM_THREADS_PER_BLOCK 288 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS NUM_BLOCKS*NUM_THREADS_PER_BLOCK   
#endif




#define NUMSTEPS_GPU       6000
#define PI                 3.141592654f
#define RPI                0.318309886f
#define MAX_LAYERS         100
#define STR_LEN            200
#define NUM_OF_DETECTOR    5
#define NORMAL             0
#define ANGLE              0

#define NAOfSource         0.12
#define NAOfDetector       0.12
#define n_detector         1.457
#define n_source           1.457
#define illumination_r     0.022					//radius //YU-modified
#define collect_r          0.022			//radius //YU-modified
#define NUMBER_PHOTONS     1000000000
#define NUMBER_SIMULATION  27

//#define WEIGHT 0.0001f
#define WEIGHTI 429497u //0xFFFFFFFFu*WEIGHT
#define CHANCE 0.1f


// TYPEDEFS
typedef struct __align__(16)
{
	float z_min;		// Layer z_min [cm]
	float z_max;		// Layer z_max [cm]
	float mutr;			// Reciprocal mu_total [cm]
	float mua;			// Absorption coefficient [1/cm]
	float g;			// Anisotropy factor [-]
	float n;			// Refractive index [-]
}LayerStruct;

typedef struct __align__(16) 
{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float dx;		// (Global, normalized) x-direction
	float dy;		// (Global, normalized) y-direction
	float dz;		// (Global, normalized) z-direction
	float weight;			// Photon weight
	int layer;				// Current layer
}PhotonStruct;

typedef struct 
{
	unsigned long number_of_photons;
	unsigned int n_layers;
	float start_weight;
	LayerStruct* layers;
}SimulationStruct;

typedef struct
{
	float radius[13];	//float radius[13];		//YU-modified
	float NA[13];		//float NA[13];			//YU-modified
	float position[13];	//float position[13];	//YU-modified
	float angle[13];	//float angle[13];		//YU-modified
	float data[13];		//float data[13];		//YU-modified
}Fibers;

typedef struct
{
	Fibers* f;                          
	PhotonStruct* p;					// Pointer to structure array containing all the photon data
	unsigned int* thread_active;		// Pointer to the array containing the thread active status
	unsigned int* num_terminated_photons;	//Pointer to a scalar keeping track of the number of terminated photons
	curandState*  state;
}MemStruct;

typedef struct
{
	float* all;
	float* prob;
	float* cumf;
}G_Array;


__device__ __constant__ unsigned int num_photons_dc[1];	
__device__ __constant__ unsigned int n_layers_dc[1];		
__device__ __constant__ float start_weight_dc[1];	
__device__ __constant__ LayerStruct layers_dc[MAX_LAYERS];	