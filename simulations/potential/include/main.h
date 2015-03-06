
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
using namespace std;
using namespace NGraph;

void generateNetwork(int* net, int blocks, int N, int edges, float trans, gsl_rng* r);
void generateNetworkSW(int* net, int blocks, int N, int edges, float trans, gsl_rng* r);
void generateNetworkTri(int* net, int blocks, int N, int edges, float trans, gsl_rng* r);
void printGraph(Graph G);
float cluster_coeff1(Graph A, int N);
float cluster_coeff(int* net, int N, int edges);
float cluster_coeff_v(Graph G, int v);

#define VISUAL 0
#if VISUAL
#include "plotGrid.h"
#endif
