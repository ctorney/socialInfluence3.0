
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
#include <ngraph.hpp>
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
using namespace NGraph;
void generateNetwork(int* net, int blocks, int N, int edges, float trans, gsl_rng* r);
void printGraph(Graph G);
float cluster_coeff(Graph A, int N);
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
    int numBlocks = 1;//256;
    // length of grid
    int N = 512;
    int N2 = 0.5 * N;
    int N4 = 0.5 * N2;
    int N_ALL = N * numBlocks;

    int Ns = 8;

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
    int sigCount = 15;

    const unsigned int shape[] = {sigCount,2};

    float* results = new float[sigCount*2];


    for (int NL=4;NL<6;NL+=4)
    {

        for (int i=0;i<sigCount*2;i++)
            results[i]=0.0f;


        cout<<"~~~~~~~~~~~~~~~~~~"<<endl<<NL<<endl<<"~~~~~~~~~~~~~~~~~~"<<endl;

        char fileName[30];
        sprintf(fileName, "../output/time-%d.npy", int(2.0*NL));
        for (int G=0;G<sigCount;G++)
        {
            generateNetwork(h_net, numBlocks, N, Ns, 1.5,r);
            return 0;
            CUDA_CALL(cudaMemcpy (d_net, h_net, (N_ALL*Ns) * sizeof(int), cudaMemcpyHostToDevice));

            float sigma = 2.5 + 0.5 * float(G);


            for (int b=0;b<numBlocks;b++)
                h_blockTimes[b] = -1;
            int maxTime = 10000000;
            int checkTime = 1;


            CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemset (d_blockTotals, 0, sizeof(int) * (numBlocks)));




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
   //                 cout<<t<<endl;
                    countStates(N, numBlocks, d_states, d_blockTotals, N_ALL);

                    CUDA_CALL(cudaMemcpy(h_blockTotals, d_blockTotals, (numBlocks) * sizeof(int), cudaMemcpyDeviceToHost));
                    bool allDone = true;
                    for (int b=0;b<numBlocks;b++)
                    {
 //                       cout<<"block total : "<<h_blockTotals[b]<<endl;
                        if (h_blockTotals[b]>0.5*N)
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
            results[G*2] =sigma;
            if (count>0)
                results[G*2+1] = avTime/(float)count;
            else
                results[G*2+1] = maxTime;
            if (avTime/(float)count > 100*checkTime)
                checkTime = checkTime * 10;
            if (checkTime > 10000)
                checkTime = 10000;
            checkTime = 10;

            cout<<results[G*2]<<" "<<results[G*2+1]<<endl;
            cnpy::npy_save(fileName,results,shape,2,"w");
        }
    }
    return 0;
}
void generateNetwork(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

    float rn;
    for (int b=0;b<blocks;b++)
    {
        Graph G;
        // create a random network
        for (int i=0;i<N;i++)
            while (G.out_neighbors(i).size()<edges)
            {
                int j = gsl_rng_uniform_int(r,N);
                G.insert_edge_noloop(i, j);
            }

        // add in transitivity
        for ( Graph::const_iterator a = G.begin(); a != G.end(); a++)
        {

            int ap = G.node(a);
            Graph::vertex_set aSo = G.out_neighbors(a);
            for (Graph::vertex_set::const_iterator b = aSo.begin(); b !=aSo.end(); b++)
                if (gsl_rng_uniform(r) < trans)
                {
                    //                   int b = *t;
                    Graph::vertex_set bSo = G.out_neighbors(*b);
                    //                   int sz = G.out_neighbors(b).size();
                    //                  if (sz==0) continue; 
                    // pick an out neighbour of a that is not b
                    Graph::vertex_set::const_iterator e = aSo.begin(); 
                    rn = gsl_rng_uniform_int(r,edges);
                    advance(e,rn);
                    while (G.node(e)==G.node(b))
                    {
                        e = aSo.begin(); 
                        rn = gsl_rng_uniform_int(r,edges);
                        advance(e,rn);

                    }
                    // now pick a neighbour of b at random
                    Graph::vertex_set::const_iterator c = bSo.begin(); 
                    rn = gsl_rng_uniform_int(r,edges);
                    advance(c,rn);
                    while (G.node(a)==G.node(c))
                    {
                        c = bSo.begin(); 
                        rn = gsl_rng_uniform_int(r,edges);
                        advance(c,rn);

                    }
                    // if a is already connected to c then do nothing
                    if (G.includes_edge(G.node(a),G.node(c)))
                        continue;

                    // now pick an in neighbour of c that's not b to rewire in compensation
                    int sz = G.in_neighbors(*c).size();
                    if (sz>1) 
                    {
                        Graph::vertex_set cSi = G.in_neighbors(*c);
                        Graph::vertex_set::const_iterator d = cSi.begin(); 
                        rn = gsl_rng_uniform_int(r,sz);
                        advance(d,rn);
                        while (G.node(d)==G.node(b))
                        {
                            d = cSi.begin(); 
                            rn = gsl_rng_uniform_int(r,sz);
                            advance(d,rn);

                        }
                        //               G.remove_edge(make_pair(*d,*c));
                        if (!G.includes_edge(G.node(d),G.node(e)))
                        {
                            G.remove_edge(G.node(d),G.node(c));//make_pair(a,*c));
                            G.insert_edge(G.node(d),G.node(e));//make_pair(a,*c));
                        }
                        //               G.insert_edge(make_pair(*d,*e));
                    }
                    //              cout<<ap<<":"<<*b<<":"<<*c<<":"<<*e<<":"<<endl;
                    //               printGraph(G);
                    G.insert_edge(G.node(a),G.node(c));//make_pair(a,*c));
                    G.remove_edge(G.node(a),G.node(e));//make_pair(a,*c));
                    //             G.remove_edge(make_pair(a,*e));



                }

            //               net[b*N*edges + i*edges + j++] = *t;
        }
                    cout<<cluster_coeff(G, N)<<endl;

        // save to a list
        for ( Graph::const_iterator p = G.begin(); p != G.end(); p++)
        {

            int j = 0;
            int i = G.node(p);
            Graph::vertex_set So = G.out_neighbors(p);
            for (Graph::vertex_set::const_iterator t = So.begin(); t !=So.end(); t++)
                net[b*N*edges + i*edges + j++] = *t;
        }
    }

    //  for (int b=0;b<blocks;b++)
    //  {
    //     for (int i=0;i<N;i++)
    //    {
    //       cout<<i<<"::";
    //      for (int j=0;j<edges;j++)    
    //         cout<< net[b*N*edges + i*edges + j] <<":";
    //     cout<<endl;
    // }
    // }
    // generate network

}
void printGraph(Graph G)
{
    cout<<"Printing graph ..."<<endl;

    for ( Graph::const_iterator a = G.begin(); a != G.end(); a++)
    {

        int ap = G.node(a);
        Graph::vertex_set aSo = G.out_neighbors(a);
        cout<<ap<<"::";
        for (Graph::vertex_set::const_iterator b = aSo.begin(); b !=aSo.end(); b++)
            cout<<*b<<":";
        cout<<endl;
    }

    cout<<"... done"<<endl;
}

float cluster_coeff(Graph G, int N) 
{
    float av_cc = 0.0f;
    int count = 0;
    for ( Graph::const_iterator a = G.begin(); a != G.end(); a++)
    {
        Graph::vertex_set aSo = G.out_neighbors(a);
        float links = 0.0;
        for (Graph::vertex_set::const_iterator b = aSo.begin(); b !=aSo.end(); b++)
            for (Graph::vertex_set::const_iterator c = aSo.begin(); c !=aSo.end(); c++)
                if (G.includes_edge(G.node(b),G.node(c)))
                    links+=1.0;
        //cout<<links<<endl;
        double  e = G.out_neighbors(a).size(); 
        av_cc +=  links / (e * (e-1));
        count++;
    }
    cout<<count<<endl;
    return av_cc/float(N);
}
