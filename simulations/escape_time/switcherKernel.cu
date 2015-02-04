
#include <curand_kernel.h>
#include <stdio.h>
#include "params.h"

__device__ int getIndex(int t_x, int t_y)
{
    // calculate full index from a grid position 
    int indx = __mul24(t_y,blockDim.x) + t_x;
    return __mul24(blockDim.y, __mul24(blockIdx.x, blockDim.x)) + indx;

}
__device__ int getIndex(int t_x)
{
    // calculate full index from a grid position 
    return __mul24(blockDim.y, __mul24(blockIdx.x, blockDim.x)) + t_x;

}
        

__global__ void d_initRands(curandState *state, int seed)
{
    int id = getIndex(threadIdx.x, threadIdx.y);

    /* Each thread gets same seed, a different sequence 
     *        number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

__global__ void d_updateStates(int* states, int* net, float sigma, int N, curandState* d_rands, int Ns, int t)
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


    //float pup = exp((1.0f/sigma)*(abs(lp-1000.0f)-abs(lp+1000.0f)));
    float pup = exp((1.0f/sigma)*(-2.0*lp));
    float pall = pup*powf(chi,s_exp);
    int newState;
    if (pall<1.0f)
        newState = 1;
    else
        newState = 0;

    __syncthreads();

    if (t==threadIdx.x)
        states[id] = newState;
    bool debug = 0;
    if ((debug)&&(threadIdx.x==t))
    {
        int sCount = 0;
        printf("%0.5f %0.5f %0.5f %d \n",s_exp, pup, pall, newState );
    }
}

__global__ void d_recordData(int* states, int* states2, curandState* d_rands,  int N_x, float* d_up, float* d_down, int* d_upcount, int* d_downcount, int t)
{

    int group_id = threadIdx.y * N_x + threadIdx.x;

    int N = N_x*N_x;

    if ((group_id==0)&&(blockIdx.x==0))
        for (int b=0;b<gridDim.x;b++)
        {
            if (t==0)
                for (int i=0;i<N;i++)
                    states2[b * N + i] = states[b * N + i];
            else
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



                //         res[blockIdx.y] = counter/float(t+1) + t*res[blockIdx.y]/float(t+1);



                // now for something crazy!!!
                // we're going to count all the uppies and then put them all in order
                totalUp=0;
                for (int i=0;i<N;i++)
                {
                    if (states[b * N + i] > 0.5)
                        totalUp++;
     //               states[b * N + i] = 0;
                }
 //               totalUp=32;
       /*         int nc = 0.875 * totalUp;
                float frac = float(totalUp-nc)/float(N-totalUp);
                for (int i=0;i<nc;i++)
                    states[b * N + i] = 1;
                for (int i=nc;i<N;i++)
                    if (curand_uniform(&d_rands[group_id])< frac)
                        states[b * N + i] = 1;
*/
       //         int i2 = totalUp + 0.5*(N-totalUp);
         //           states[b * N + i2] = 1;
//
                for (int i=0;i<N;i++)
                    states2[b * N + i] = states[b * N + i];
            }

    
        //res[t * gridDim.y + blockIdx.y] = counter;
      //  if (t==0)
  //          res[blockIdx.y] = counter;
     //   else
   //         res[blockIdx.y] = counter/float(t+1) + t*res[blockIdx.y]/float(t+1);
        }
}


__global__ void block_sum(const int *input, int *per_block_results, const size_t n)
{
    extern __shared__ int sdata[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into __shared__ memory
    int x = 0;
    if(i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2;
            offset > 0;
            offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        // wait until all threads in the block hav
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

void initRands(dim3 threadGrid, int numBlocks, curandState *state, int seed) 
{
    d_initRands<<< numBlocks, threadGrid >>>(state, seed);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );
}
void advanceTimestep(int numThreads, int numBlocks, curandState *rands, float sigma, int* states, int* net, int N, int Ns, int t)
{
    int r = rand() / ( RAND_MAX / (N) );
    d_updateStates<<< numBlocks, numThreads >>>(states, net, sigma, N, rands, Ns, r);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );

}
void recordData(dim3 threadGrid, int numBlocks, int* states, int* states2, curandState *rands, int N_x, float* d_up, float* d_down, int* d_upcount, int* d_downcount, int t)
{
     d_recordData<<< numBlocks, threadGrid >>>(states, states2, rands, N_x, d_up, d_down, d_upcount, d_downcount, t);
     if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );
}
void countStates(int numThreads, int numBlocks, int* states, int* blockTotals, int N_ALL)
{
    block_sum<<< numBlocks, numThreads, numThreads * sizeof(int) >>>(states, blockTotals, N_ALL);
    if (cudaSuccess != cudaGetLastError()) printf( "cuda error!\n" );

}
