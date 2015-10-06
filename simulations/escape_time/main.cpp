
/*test.cc*/
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iostream>
#include <algorithm>
#include <time.h>
#include "cuda_call.h"
#include "switcherKernel.h"
#include "params.h"
#include "cnpy.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define VISUAL 0
#if VISUAL
#include "plotGrid.h"
#endif

using namespace std;
void generateNetwork(int* net, int blocks, int N, int edges, float trans, gsl_rng* r);
void generateNetworkSW(int* net, int blocks, int N, int edges, float trans, gsl_rng* r);
float cluster_coeff(int* net, int N, int edges);
int main() 
{

#if VISUAL
    plotGrid* pg = new plotGrid;
#endif
    const gsl_rng_type * T;
    gsl_rng * r;



    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    // number of reps
    int numBlocks = 128;//512;
    // length of grid
    int N = 1024;
    int N2 = 0.5 * N;
    int N4 = 0.5 * N2;
    int N_ALL = N * numBlocks;

    int Ns = 20;

    dim3 threadGrid(N);
    curandState *devRands;
    CUDA_CALL(cudaMalloc((void **)&devRands, N_ALL * sizeof(curandState)));

    srand(0);// (time(NULL));
    initRands(threadGrid, numBlocks, devRands, rand());


    int* d_net;
    CUDA_CALL(cudaMalloc((void**)&d_net, sizeof(int) *  (Ns*N_ALL) ));
    int* d_states;
    CUDA_CALL(cudaMalloc((void**)&d_states, sizeof(int) * N_ALL));
    int* d_states2;
    CUDA_CALL(cudaMalloc((void**)&d_states2, sizeof(int) * N_ALL));


    int* d_blockTotals;
    CUDA_CALL(cudaMalloc((void**)&d_blockTotals, sizeof(int) * numBlocks));

    int* h_net = new int [Ns * N_ALL];
    int* h_states = new int[N_ALL];
    int* h_blockTotals = new int[numBlocks];
    int* h_blockTimes = new int[numBlocks];
    int sigCount = 20;

    const unsigned int shape[] = {sigCount,2};

    float* results = new float[sigCount*2];


    for (int NL=4;NL<6;NL+=4)
    {

        for (int i=0;i<sigCount*2;i++)
            results[i]=0.0f;


        float sigma = 4.5;//2.5 + 0.25 * float(G);
        cout<<"#~~~~~~~~~~~~~~~~~~"<<endl<<"#sigma "<<sigma<<"Ns "<<Ns<<" chi "<<chi<<endl<<"#~~~~~~~~~~~~~~~~~~"<<endl;

        char fileName[30];
        sprintf(fileName, "./output/time-%d.npy", int(2.0*NL));
        for (int G=0;G<sigCount;G++)
        {
            float alpha  = 1.0-0.05 * float(G);
            //generateNetworkSW(h_net, numBlocks, N, Ns, alpha, r);
            generateNetwork(h_net, numBlocks, N, Ns, alpha, r);

             // generate network
            /*for (int b=0;b<numBlocks;b++)
            {
                for (int i=0;i<N;i++)
                    for (int j=0;j<Ns;j++)
                        h_net[b*N*Ns + i*Ns + j] = gsl_rng_uniform_int(r,N);
            }*/
            CUDA_CALL(cudaMemcpy (d_net, h_net, (N_ALL*Ns) * sizeof(int), cudaMemcpyHostToDevice));

 //           float sigma = 2.0 + 0.25 * float(G);


            for (int b=0;b<numBlocks;b++)
                h_blockTimes[b] = -1;
            int maxTime = 100000000;
            int checkTime = 1;


            CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemset (d_blockTotals, 0, sizeof(int) * (numBlocks)));



            int NESC = int(N * (0.5- 1.0/(sigma*(chi))) - 1);

            for (int t=0;t<maxTime;t++)
            {

                advanceTimestep(N, numBlocks, devRands, sigma, d_states, d_net, N, Ns, t);
                /*
                   CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
                   int countUp = 0;
                   for (int i=0;i<N_ALL;i++)
                   if (h_states[i]>0)
                   countUp++;
                   cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<countUp<<endl;
                //            */
#if VISUAL
                CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
                pg->draw(Nx, h_states);
#endif
                if (t%checkTime == 0 ) 
                {
                    countStates(N, numBlocks, d_states, d_blockTotals, N_ALL);

                    CUDA_CALL(cudaMemcpy(h_blockTotals, d_blockTotals, (numBlocks) * sizeof(int), cudaMemcpyDeviceToHost));
                    bool allDone = true;
                    for (int b=0;b<numBlocks;b++)
                    {
                        if (h_blockTotals[b]>NESC)
                        {
                            if (h_blockTimes[b]<0)
                                h_blockTimes[b]=t;
                        }
                        else
                        {
                            //           cout<<b<<" block done"<<endl;
                            allDone = false;
                        }
                    }
                    if (allDone)
                    {
                        break;
                        /*
                           for (int b=0;b<numBlocks;b++)
                           h_blockTimes[b] = -1;
                           CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
                           */
                    } 
                }

            }

            float avTime = 0.0f;
            int count=0;
            for (int b=0;b<numBlocks;b++)
                if (h_blockTimes[b]>0)
                {
                    avTime += (float)h_blockTimes[b];
                    count++;
                }
            results[G*2] = alpha;
            if (count>0)
                results[G*2+1] = avTime/(float)count;
            else
                results[G*2+1] = maxTime;
            if (avTime/(float)count > 100*checkTime)
                checkTime = checkTime * 10;
            if (checkTime > 10000)
                checkTime = 10000;
 //           checkTime = 10;

            cout<<results[G*2]<<" "<<results[G*2+1]<<endl;
            cnpy::npy_save(fileName,results,shape,2,"w");
        }
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
                if (net[bl*N*edges + i*edges + j]<0)
                    net[bl*N*edges + i*edges + j]+=N;
           //     cout<<":"<<net[bl*N*edges + i*edges + j];
            }
           // cout<<endl;
        }




    }
 //       cout<<av_cc/float(blocks)<<":";
}
void generateNetworkNEWMAN(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

    // from Newman's paper
    float av_cc = 0.0f;
    for (int bl=0;bl<blocks;bl++)
    {
        int allEdges[N*(edges+1)];
        for (int i=0;i<N;i++)
        {
            allEdges[i*(edges+1)]=0;
            for (int j=0;j<edges;j++)
                allEdges[i*(edges+1)+j+1]=-1;

        }
        int eInds[N*edges];
        int eList[N*edges];
        for (int i=0;i<N*edges;i++)
        {
            eInds[i]=i;
            eList[i]=-1;
        }
        int tInds[int(0.5*N*edges)];
        int tList[int(0.5*N*edges)];
        for (int i=0;i<int(0.5*N*edges);i++)
        {
            tInds[i]=i;
            tList[i]=-1;
        }
        gsl_ran_shuffle (r, eInds, int(N*edges), sizeof(int));
        gsl_ran_shuffle (r, tInds, int(0.5*N*edges), sizeof(int));

        int ce=0;
        int ct=0;
        for (int i=0;i<N;i++)
        {
            int e = edges;
            while (e>0)
            {
                if (e==1)
                {
                    eList[ce]=i;
                    e--;
                    ce++;
                }
                else
                {
                    float rn = gsl_rng_uniform(r);
                    if (rn<trans)
                    {
                        tList[ct]=i;
                        e-=2;
                        ct++;
                    }
                    else
                    {
                        eList[ce]=i;
                        e--;
                        ce++;

                    }


                }
            }

        }
        // we need to have a least a couple of edges to start off with
        if (ce==0)
        {
            int nct = gsl_rng_uniform_int(r,ct);
            int i = tList[nct];
            tList[nct] = -1;
            eList[0] = i;
            eList[1] = i;
        }
        // now do all the single edges
        for (int i=0;i<N*edges;i++)
        {
            int v = eList[eInds[i]];
            if (v<0) continue;
            allEdges[v*(edges+1)]++;
            allEdges[v*(edges+1) + allEdges[v*(edges+1)]] = gsl_rng_uniform_int(r,N);


        }
        // next do all the triangles
        for (int i=0;i<int(0.5*N*edges);i++)
        {
            int v = tList[tInds[i]];
            if (v<0) continue;
            int n1=gsl_rng_uniform_int(r,N);
            while (allEdges[n1*(edges+1)]==0)
            {
                n1++;
                if (n1==N)
                    n1=0;
 //               n1=gsl_rng_uniform_int(r,N);
            }
            int n1c = allEdges[n1*(edges+1)];
            int n2 = gsl_rng_uniform_int(r,n1c) + 1;
            n2 = allEdges[n1*(edges+1)+ n2];
            allEdges[v*(edges+1)]++;
            allEdges[v*(edges+1) + allEdges[v*(edges+1)]] = n1;
            allEdges[v*(edges+1)]++;
            allEdges[v*(edges+1) + allEdges[v*(edges+1)]] = n2;


        }
        av_cc +=cluster_coeff(allEdges,N,edges); 
        for (int i=0;i<N;i++)
        {
         //   cout<<i<<":";
            for (int j=0;j<edges;j++){
                net[bl*N*edges + i*edges + j] =allEdges[i*(edges+1)+j+1];
           //     cout<<":"<<net[bl*N*edges + i*edges + j];
            }
//            cout<<endl;
        }




    }
        cout<<av_cc/float(blocks)<<":";
}
void generateNetwork(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

//Jennifer Badham and Rob Stocker (2010)
//A Spatial Approach to Network Generation for Three Properties: Degree Distribution, Clustering Coefficient and Degree Assortativity
    float av_cc = 0.0f;
    int he = int(0.5*edges);
    int nx = powf(float(N),0.5);
    int neigh[8][2] = { { 1, 1 }, { 1, 0 }, { 1, -1 } , { 0, 1 }, { 0, -1 }, { -1, -1 } , { -1, 0 }, { -1, 1 } };
    for (int bl=0;bl<blocks;bl++)
    {
        int allEdges[N*(edges+1)];
        for (int ix=0;ix<nx;ix++)
        for (int iy=0;iy<nx;iy++)
        {
            int i = ix*nx + iy;
            int asigned[nx][nx];
            for (int iix=0;iix<nx;iix++)
                for (int iiy=0;iiy<nx;iiy++)
                    asigned[iix][iiy]=0;
            asigned[ix][iy]=1;

            allEdges[i*(edges+1)]=0;//edges;
            int j = 1;
            int offx=0;
            int offy=0;
            int dx[4] = {0, 1, 0, -1};
            int dy[4] = {1, 0, -1, 0};
            while (allEdges[i*(edges+1)]<edges)
            {
                j++;
                for (int k=0;k<j/2;k++)
                {
                    if (allEdges[i*(edges+1)]==edges)
                        break;

                    offx += dx[j % 4];
                    offy += dy[j % 4];
                    //            cout<<i<<" "<<(allEdges[i*(edges+1)])<<endl;
                    int jx = (ix+offx)%nx;
                    int jy = (iy+offy)%nx;

                    if ((gsl_rng_uniform(r)<trans))//&(asigned[jx][jy]==0))
                    {
                        asigned[jx][jy]=1;
                        allEdges[i*(edges+1)]++;
                        int nj = jx*nx + jy;
                        allEdges[i*(edges+1) + allEdges[i*(edges+1)]]= nj;
                    }
                    else
                    {
                        asigned[jx][jy]=1;
                        allEdges[i*(edges+1)]++;
                        allEdges[i*(edges+1) + allEdges[i*(edges+1)]]= gsl_rng_uniform_int(r,N);

                    }
                }
            }

        }
        av_cc +=cluster_coeff(allEdges,N,edges); 
        for (int i=0;i<N;i++)
        {
            //           cout<<i<<":";
            for (int j=0;j<edges;j++){
                net[bl*N*edges + i*edges + j] =allEdges[i*(edges+1)+j+1];
                if (net[bl*N*edges + i*edges + j]<0)
                    net[bl*N*edges + i*edges + j]+=N;
                //             cout<<":"<<net[bl*N*edges + i*edges + j];
            }
            //       cout<<endl;
        }




    }
    cout<<av_cc/float(blocks)<<":";
}
void generateNetwork1D(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

    //Jennifer Badham and Rob Stocker (2010)
    //A Spatial Approach to Network Generation for Three Properties: Degree Distribution, Clustering Coefficient and Degree Assortativity
    float av_cc = 0.0f;
    int he = int(0.5*edges);
    for (int bl=0;bl<blocks;bl++)
    {
        int allEdges[N*(edges+1)];
        for (int i=0;i<N;i++)
        {
            allEdges[i*(edges+1)]=0;//edges;
            int j = 1;
            while (allEdges[i*(edges+1)]<edges)
            {

                if (gsl_rng_uniform(r)<trans)
                {
                    allEdges[i*(edges+1)]++;
                    allEdges[i*(edges+1) + allEdges[i*(edges+1)]]= (i-(j)) % N ;
                }
                if ((gsl_rng_uniform(r)<trans)&(allEdges[i*(edges+1)]<edges))
                {
                    allEdges[i*(edges+1)]++;
                    allEdges[i*(edges+1) + allEdges[i*(edges+1)]]= (i+(j)) % N ;
                }
                j++;
            }

        }
        av_cc +=cluster_coeff(allEdges,N,edges); 
        for (int i=0;i<N;i++)
        {
            //   cout<<i<<":";
            for (int j=0;j<edges;j++){
                net[bl*N*edges + i*edges + j] =allEdges[i*(edges+1)+j+1];
                if (net[bl*N*edges + i*edges + j]<0)
                    net[bl*N*edges + i*edges + j]+=N;
           //     cout<<":"<<net[bl*N*edges + i*edges + j];
            }
           // cout<<endl;
        }




    }
        cout<<av_cc/float(blocks)<<":";
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
