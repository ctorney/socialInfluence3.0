
#include "main.h"

int main() 
{
    const gsl_rng_type * T = gsl_rng_default;
    gsl_rng * r = gsl_rng_alloc (T);
    gsl_rng_env_setup();

    // number of reps
    int numBlocks = 512;

    // number of individuals in a group
    int N = 64;
    int N_ALL = N * numBlocks;

    // number of social observations
    int Ns = 18;

    // *********************************
    //      declare host variables
    // *********************************
    float* h_up = new float [N+1];
    float* h_down = new float [N+1];
    int* h_upcount = new int [N+1];
    int* h_net = new int [Ns * N_ALL];
    int* h_states = new int[N_ALL];
    int* h_blockTotals = new int[numBlocks];
    int* h_blockTimes = new int[numBlocks];
    // *********************************


    // *********************************
    //      declare device variables
    // *********************************
    curandState *devRands;
    int *d_net, *d_states, *d_states2, *d_upcount, *d_blockTotals;
    float *d_up, *d_down;
    CUDA_CALL(cudaMalloc((void **)&devRands, N_ALL * sizeof(curandState)));
    CUDA_CALL(cudaMalloc((void**)&d_net, sizeof(int) *  (Ns*N_ALL) ));
    CUDA_CALL(cudaMalloc((void**)&d_states, sizeof(int) * N_ALL));
    CUDA_CALL(cudaMalloc((void**)&d_states2, sizeof(int) * N_ALL));
    CUDA_CALL(cudaMalloc((void**)&d_up, sizeof(float) *  (N + 1) ));
    CUDA_CALL(cudaMalloc((void**)&d_down, sizeof(float) *  (N + 1) ));
    CUDA_CALL(cudaMalloc((void**)&d_upcount, sizeof(int) *  (N + 1) ));
    CUDA_CALL(cudaMalloc((void**)&d_blockTotals, sizeof(int) * numBlocks));
    // *********************************


    int sigCount = 1;


    // initialize random numbers
    initRands(N, numBlocks, devRands, gsl_rng_uniform(r));

    // set-up output for reading into python
    const unsigned int shape[] = {N+1,2};
    float* results = new float[(N+1)*2];
    for (int i=0;i<(N+1)*2;i++)
       results[i]=0.0f;





        for (int G=0;G<sigCount;G++)
        {
            // reset all variables
            CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemset (d_states2, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemset (d_blockTotals, 0, sizeof(int) * (numBlocks)));
            CUDA_CALL(cudaMemset (d_up, 0, sizeof(float) * (N + 1)));
            CUDA_CALL(cudaMemset (d_down, 0, sizeof(float) * (N + 1)));
            CUDA_CALL(cudaMemset (d_upcount, 0, sizeof(int) * (N + 1)));

            float cluster = 0.85;
            generateNetworkSW(h_net, numBlocks, N, Ns, cluster,r);
            CUDA_CALL(cudaMemcpy (d_net, h_net, (N_ALL*Ns) * sizeof(int), cudaMemcpyHostToDevice));

            float sigma = 5.5 + 0.5 * float(G);
            char fileName[300];
            sprintf(fileName, "../output/potential%f-%f.npy", sigma, cluster);


            for (int b=0;b<numBlocks;b++)
                h_blockTimes[b] = -1;
            int maxTime = 2000;


            for (int t=0;t<maxTime;t++)
            {

                advanceTimestep(N, numBlocks, devRands, sigma, d_states, d_net, N, Ns, t);
                recordData(N, numBlocks, d_states, d_states2, devRands, d_up, d_down, d_upcount, t);
                /*
                   CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
                   int countUp = 0;
                   for (int i=0;i<N_ALL;i++)
                   if (h_states[i]>0)
                   countUp++;
                   cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<countUp<<endl;
                //            */
            }

        CUDA_CALL(cudaMemcpy(h_up, d_up, (N + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_down, d_down, (N + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_upcount, d_upcount, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i=0;i<N+1;i++)
        {
            results[2*i]=h_up[i];
            results[2*i+1]=h_down[i];
            cout<<i/float(N)<<" "<<h_up[i]<<" "<<h_down[i]<<" "<<h_upcount[i]<<endl;
        }
        cnpy::npy_save(fileName,results,shape,2,"w");


    }
    return 0;
}
void generateNetworkSW(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

    float av_cc = 0.0f;
    int he = int(0.5*edges);
    for (int bl=0;bl<blocks;bl++)
    {
        int allEdges[N*(edges+1)];
        for (int i=0;i<N;i++)
        {
            allEdges[i*(edges+1)]=edges;
            for (int j=0;j<he;j++)
                if (gsl_rng_uniform(r)<trans)
                allEdges[i*(edges+1)+j+1]= (i-(j+1)) % N ;
                else
                allEdges[i*(edges+1)+j+1]= gsl_rng_uniform_int(r,N);
            for (int j=he;j<edges;j++)
                if (gsl_rng_uniform(r)<trans)
                allEdges[i*(edges+1)+j+1]= (i+(j+1-he)) % N ;
                else
                allEdges[i*(edges+1)+j+1]= gsl_rng_uniform_int(r,N);

        }
        av_cc +=cluster_coeff(allEdges,N,edges); 
        for (int i=0;i<N;i++)
        {
         //   cout<<i<<":";
            for (int j=0;j<edges;j++){
                net[bl*N*edges + i*edges + j] =allEdges[i*(edges+1)+j+1];
           //     cout<<":"<<net[bl*N*edges + i*edges + j];
            }
           // cout<<endl;
        }




    }
        cout<<av_cc/float(blocks)<<endl;
}
float cluster_coeff(int* net, int N, int edges) 
{
    float av_cc = 0.0f;
    for (int i=0;i<N;i++)
    {
        float links = 0.0;
        for (int j=0;j<edges;j++)
            for (int k=0;k<edges;k++)
            {
                int n1 = net[i*(edges+1)+j+1];
                int n2 = net[i*(edges+1)+k+1];
                // does n1 look at n2?
                for (int l=0;l<edges;l++)
                    if (n2 == net[n1*(edges+1)+l+1])
                    {
                        links+=1.0;
                        break;
                    }

            }
        double e = 1.0 * double(edges);
        av_cc +=  links / (e * (e-1));

    }
    return av_cc/float(N);
}
