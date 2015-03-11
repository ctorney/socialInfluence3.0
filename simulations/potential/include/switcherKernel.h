
void initRands(int numThreads, int blockGrid, curandState *state, int seed); 
void advanceTimestep(int numThreads, int numBlocks, curandState *rands, float sigma, int* states, int* net, int N_x, int Ns, int t);
void recordData(int threadGrid, int blockGrid, int* states, int* states2, curandState *rands, float* d_up, float* d_down, int* d_upcount, int t);
void countStates(int numThreads, int numBlocks, int* states, int* blockTotals, int N_ALL);
