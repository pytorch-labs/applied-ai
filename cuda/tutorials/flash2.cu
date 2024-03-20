
// flash2

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
const int Tc, const int Tr, const int Bc, const int Br, const float sm_scale,
float* l, float* m, float* O)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;  // batch index
    int bidy = blockIdx.y;  // head index

    int qkv_offset = (bidx * gridDim.y * N * d) + (bidy*N*d);
    int lm_offset = (bidx * gridDim.y *N) + (bidy *N); //l and m offset

    extern __shared__ float sram[];
    int tile_size = Bc * d; size of Qi, Kj, Vj

    float* Qi = sram;
    float * Kj = &sram[tile_size];
    float* Vj = &sram[tile_size *2];
    float* S = &sram[tile_size *3];

    for (int j=0; j < Tc; j++) {

        // load Kj, Vj to sram
        for (int x=0; x < d; x++) {
            Kj[(tx*d)+x] = K[qkv_offset + (tile_size *j) + (tx*d) +x];
            Vj[(tx*d) + x] = V[qkv_offset +(tile_size *j) + (tx*d) +x];
        }
        __synchthreads();

    }
}


    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj
