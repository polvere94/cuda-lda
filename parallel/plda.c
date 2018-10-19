#include <stdio.h>

#define BDMX 16
#define BDMY 16

#define INDEX(ros,col,stride) (row * stride + col)

__device__ double transposeSmem(float* out, float* in, int n_row,int n_col){
	__shared__ float tile[BDMY][BDMX];

	unsigned int row = blockDim.y * blockIdx.y + threadIds.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIds.x;

	unsigned int offset =  INDEX(row, col, n_col);

	if(row < n_row && col < n_col){
		tile[threadIdx.y][threadIdx.x] = in[offset];
	}
	__synchthreads();
}
//calcolo prodotto tra matrici
int main(void){

	double *dev_data_points;

	cudaMalloc( (void**)&dev_data_points, NUM_POINTS*DIM*sizeof(double) );

	return 0;
}