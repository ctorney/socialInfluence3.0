
#include <curand_kernel.h>
#include <stdio.h>
#include "params.h"

__global__ void d_initRands(curandState *state, int seed)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread gets same seed, a different sequence number, no offset 
    curand_init(seed, id, 0, &state[id]);
}

__global__ void d_updateStates(int* states, int* net, float sigma, int N, curandState* d_rands, int Ns)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bstart = blockIdx.x * blockDim.x;

    int deltan = 0;
    for (int e=0;e<Ns;e++)
    {
        int n2 = bstart + net[id*Ns + e];
        if (states[n2]>0.5)
            deltan++;
    }
    float s_exp = 1.0f - 2.0f*float(deltan)/float(Ns);

    float lp;
    if (curand_uniform(&d_rands[id])<0.5)
       lp = 1.0 + (sigma * log(curand_uniform(&d_rands[id])));
    else
       lp = 1.0 - (sigma * log(curand_uniform(&d_rands[id])));

    float pup = exp((1.0f/sigma)*(-2.0*lp));
    float pall = pup*powf(chi,s_exp);
    int newState;
    if (pall<1.0f)
        newState = 1;
    else
        newState = 0;

    __syncthreads();

    states[id] = newState;
}

__global__ void d_recordData(int* states, int* states2, int N, curandState* d_rands,  float* d_up, float* d_down, int* d_upcount, int t)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id!=0)
        return;

    for (int b=0;b<gridDim.x;b++)
    {
        int totalUp = 0;
        for (int i=0;i<N;i++)
            if (states2[b * N + i] > 0.5)
                totalUp++;


        int nowDown = 0;
        for (int i=0;i<N;i++)
            if ((states2[b * N + i] > 0.5)&&(states[b * N + i] < 0.5))
                nowDown++;

        int nowUp = 0;
        for (int i=0;i<N;i++)
            if ((states2[b * N + i] < 0.5)&&(states[b * N + i] > 0.5))
                nowUp++;


        d_upcount[totalUp]+=1;
        int c = d_upcount[totalUp];
        //           printf("%d %d %d %d\n",t, totalUp,nowDown, nowUp);
        d_down[totalUp] = (nowDown/(float)N)/(float)c + (c-1)*d_down[totalUp]/(float)c;
        d_up[totalUp] = (nowUp/(float)N)/(float)c + (c-1)*d_up[totalUp]/(float)c;


        for (int i=0;i<N;i++)
            states2[b * N + i] = states[b * N + i];
    }


}


__global__ void block_sum(const int *input, int *per_block_results, const size_t n)
{
    extern __shared__ int sdata[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into __shared__ memory
    int x = 0;
    if (i < n)
        x = input[i];
    sdata[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2; offset > 0;offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

void initRands(int numThreads, int numBlocks, curandState *state, int seed) 
{
    d_initRands<<< numBlocks, numThreads >>>(state, seed);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );
}
void advanceTimestep(int numThreads, int numBlocks, curandState *rands, float sigma, int* states, int* net, int N, int Ns, int t)
{
    d_updateStates<<< numBlocks, numThreads >>>(states, net, sigma, N, rands, Ns);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );

}
void recordData(int numThreads, int numBlocks, int* states, int* states2, curandState *rands, float* d_up, float* d_down, int* d_upcount, int t)
{
    d_recordData<<< numBlocks, numThreads >>>(states, states2, numThreads, rands, d_up, d_down, d_upcount, t);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );
}
void countStates(int numThreads, int numBlocks, int* states, int* blockTotals, int N_ALL)
{
    block_sum<<< numBlocks, numThreads, numThreads * sizeof(int) >>>(states, blockTotals, N_ALL);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );

}
