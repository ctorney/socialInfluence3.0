
#include "main.h"

int main() 
{
    const gsl_rng_type * T = gsl_rng_default;
    gsl_rng * r = gsl_rng_alloc (T);
    gsl_rng_env_setup();

    // number of reps
    int numBlocks = 1;//512;

    // number of individuals in a group
    int N = 64;
    int N_ALL = N * numBlocks;

    // number of social observations
    int Ns = 8;

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

            float cluster = 0.00;
            generateNetworkTri(h_net, numBlocks, N, Ns, cluster,r);
            return 0;

            // generate network
            /*for (int b=0;b<numBlocks;b++)
            {
                for (int i=0;i<N;i++)
                    for (int j=0;j<Ns;j++)
                        h_net[b*N*Ns + i*Ns + j] = gsl_rng_uniform_int(r,N);
            }*/
            CUDA_CALL(cudaMemcpy (d_net, h_net, (N_ALL*Ns) * sizeof(int), cudaMemcpyHostToDevice));

            float sigma = 4.0 + 0.5 * float(G);
            char fileName[30];
            sprintf(fileName, "../output/potential%f-%f.npy", sigma, cluster);


            for (int b=0;b<numBlocks;b++)
                h_blockTimes[b] = -1;
            int maxTime = 200;


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
void generateNetworkTri(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

    float av_cc = 0.0f;
    int allEdges[N*(edges+1)], a[3], b[N], c[N];
    for (int bl=0;bl<blocks;bl++)
    {

        int Ntri = N;
        for (int i=0;i<N;i++)
        {
            allEdges[i*(edges+1)]=0;
            b[i] = c[i] = i;
        }

        while (Ntri>=3)
        {
            gsl_ran_choose (r, a, 3, c, Ntri, sizeof(int));
            for (int i=0;i<3;i++)
                for (int j=0;j<3;j++)
                {
                    if (i==j)
                        continue;
                    int ni = b[a[i]];
                    int nj = b[a[j]];
                    allEdges[ni*(edges+1)]++;
                    allEdges[ni*(edges+1) + allEdges[ni*(edges+1)]] = nj;
                }

            for (int i=0;i<3;i++)
            {
                int ni = b[a[i]];
                // check to see if this individual still has two free edges spare
                // if not remove it from the list
                if (allEdges[ni*(edges+1)]>(edges-2))
                    b[a[i]] = b[--Ntri];
            }

        }
        for (int i=0;i<N;i++)
        {
            // fill any outstanding edges
            while (allEdges[i*(edges+1)]<edges)
            {
                    allEdges[i*(edges+1)]++;
                    allEdges[i*(edges+1) + allEdges[i*(edges+1)]] = gsl_rng_uniform_int(r,N);
            }
            // randomize according to transitivity metric
            for (int j=0;j<edges;j++)
                if (gsl_rng_uniform(r)>trans)
                    allEdges[i*(edges+1)+j+1]= gsl_rng_uniform_int(r,N);

        }



        av_cc +=cluster_coeff(allEdges,N,edges); 
        for (int i=0;i<N;i++)
        {
            cout<<i<<":";
            for (int j=0;j<edges;j++){
                net[bl*N*edges + i*edges + j] =allEdges[i*(edges+1)+j+1];
                cout<<":"<<net[bl*N*edges + i*edges + j];
            }
            cout<<endl;
        }




    }
        cout<<av_cc/float(blocks)<<endl;
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
void generateNetwork(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

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
            while (e)
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
                n1=gsl_rng_uniform_int(r,N);
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
  //          cout<<i<<":";
            for (int j=0;j<edges;j++){
                net[bl*N*edges + i*edges + j] =allEdges[i*(edges+1)+j+1];
    //            cout<<":"<<net[bl*N*edges + i*edges + j];
            }
      //      cout<<endl;
        }




    }
        cout<<av_cc/float(blocks)<<endl;
}
void generateNetwork2(int* net, int blocks, int N, int edges, float trans, gsl_rng* r)
{

    float rn;
    for (int bl=0;bl<blocks;bl++)
    {
        Graph G;
        // create a random network
        for (int i=0;i<N;i++)
    //        for (int k=0;k<edges;k++)
            while (G.out_neighbors(i).size()<edges)
            {
                int j = gsl_rng_uniform_int(r,N);
                G.insert_edge_noloop(i, j);
      //          G.insert_edge(i, j);
            }


        // add in transitivity
        float cc = 0.0f;//cluster_coeff1(G, N);
        int maxIters = 100000;
        while ((cc<trans)&&(maxIters>0))
        {
            maxIters--;
            int v = gsl_rng_uniform_int(r,N);
            Graph::vertex_set vSo = G.out_neighbors(v);
            Graph::vertex_set::const_iterator iter = vSo.begin(); 
            rn = gsl_rng_uniform_int(r,edges);
            advance(iter,rn);
            int n1=*iter;
            int n2=n1;
            while (n1==n2)
            {
                iter = vSo.begin(); 
                rn = gsl_rng_uniform_int(r,edges);
                advance(iter,rn);
                n2=*iter;
            }
            if (G.includes_edge(n1,n2))
                continue;
            // now pick a neighbour of n1 at random
            Graph::vertex_set n1So = G.out_neighbors(n1);
            iter = n1So.begin(); 
            rn = gsl_rng_uniform_int(r,edges);
            advance(iter,rn);
            int p1=*iter;
            while (p1==v)
            {
                iter = n1So.begin(); 
                rn = gsl_rng_uniform_int(r,edges);
                advance(iter,rn);
                p1=*iter;

            }
            // now pick an in-neighbour of n2 at random
            Graph::vertex_set n2Si = G.in_neighbors(n2);
            iter = n2Si.begin(); 
            int sz = G.in_neighbors(n2).size();
            if (sz<2)
                continue;
            rn = gsl_rng_uniform_int(r,sz);
            advance(iter,rn);
            int p2=*iter;
            while (p2==v)
            {
                iter = n2Si.begin(); 
                rn = gsl_rng_uniform_int(r,sz);
                advance(iter,rn);
                p2 = *iter;

            }
            if (p1==p2)
                continue;
            if (G.includes_edge(p2, p1))
                continue;


            bool reduceCC = false;
            Graph::vertex_set n1Si = G.in_neighbors(n1);
            for (Graph::vertex_set::const_iterator b = n1Si.begin(); b !=n1Si.end(); b++)
                if (G.includes_edge(G.node(b),p1))
                    if (cluster_coeff_v(G,G.node(b))>cc)
                        reduceCC = true;

            if (reduceCC) continue;
            for (Graph::vertex_set::const_iterator b = n2Si.begin(); b !=n2Si.end(); b++)
                if (G.includes_edge(G.node(b),p2))
                    if (cluster_coeff_v(G,G.node(b))>cc)
                        reduceCC = true;

            if (reduceCC) continue;

            G.remove_edge(n1, p1); 
            G.remove_edge(p2, n2); 
            G.insert_edge(n1, n2);
            G.insert_edge(p2, p1);
            float cc2 =0.0;// cluster_coeff1(G, N);
            if (cc2<cc)
            {
            G.insert_edge(n1, p1); 
            G.insert_edge(p2, n2); 
            G.remove_edge(n1, n2);
            G.remove_edge(p2, p1);

            }
            else
                cc=cc2;
  //          cout<<blocks<<":"<<bl<<":"<<cc<<endl;


            //               net[b*N*edges + i*edges + j++] = *t;
        }

        // save to a list
        for ( Graph::const_iterator p = G.begin(); p != G.end(); p++)
        {

            int j = 0;
            int i = G.node(p);
            Graph::vertex_set So = G.out_neighbors(p);
            for (Graph::vertex_set::const_iterator t = So.begin(); t !=So.end(); t++)
                net[bl*N*edges + i*edges + j++] = *t;
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
float cluster_coeff2(Graph G, int N) 
{
    float av_cc = 0.0f;
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
    }
    return av_cc/float(N);
}
float cluster_coeff_v(Graph G, int v) 
{
    Graph::vertex_set aSo = G.out_neighbors(v);
    float links = 0.0;
    for (Graph::vertex_set::const_iterator b = aSo.begin(); b !=aSo.end(); b++)
        for (Graph::vertex_set::const_iterator c = aSo.begin(); c !=aSo.end(); c++)
            if (G.includes_edge(G.node(b),G.node(c)))
                links+=1.0;
    //cout<<links<<endl;
    double  e = G.out_neighbors(v).size(); 
    return links / (e * (e-1));
}
