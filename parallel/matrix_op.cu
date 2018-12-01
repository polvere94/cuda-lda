/*
@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/

#include "matrix_op.h"
#define SHARED_BLOCK_SIZE 16
#define INDEX(rows, cols, stride) (rows * stride + cols)

typedef struct Matrix_{
    float* data;
    float* mean;
} Matrix;


void print_matrix(float* in, int n, int m, char* label){
	int i,j;
	printf("\n%s\n",label);
	for(i =0;i<n;i++){
		for(j=0;j<m;j++){
			if(j==m-1)
				printf("%.06f;", in[(i*m)+j]);
			else
				printf("%.06f ", in[(i*m)+j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void add_matrix(float* in_a, float* in_b, float* out,int n, int m) {	
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if ((idx < n*m))
		out[idx] = in_a[idx] + in_b[idx];	
}

__global__ void add_vectors(float* in_a, float* in_b, float* out, int v_size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < v_size)
		out[idx] = in_a[idx] + in_b[idx];
}

void invert_device(cublasHandle_t handle, float* src_d, float* dst_d, int n){
    int batchSize = 1,*P, *INFO, lda = n, INFOh = 0;

    CHECK(cudaMalloc<int>(&P, n * batchSize * sizeof(int)));
    CHECK(cudaMalloc<int>(&INFO,batchSize * sizeof(int)));

    float *A[] = { src_d };
    float** A_d;
    CHECK(cudaMalloc<float*>(&A_d,sizeof(A)));
    CHECK(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize);

    CHECK(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0){
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        CHECK(cudaDeviceReset());
        exit(EXIT_FAILURE);
    }

    float* C[] = { dst_d };
    float** C_d;
    CHECK(cudaMalloc<float*>(&C_d,sizeof(C)));
    CHECK(cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice));

    cublasSgetriBatched(handle,n,(const float **)A_d,lda,P,C_d,n,INFO,batchSize);

    CHECK(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        CHECK(cudaDeviceReset());
        exit(EXIT_FAILURE);
    }

    CHECK(cudaFree(P));
    CHECK(cudaFree(INFO));
    cublasDestroy_v2(handle);
}


__global__ void mat_prod_shared(float* A, float* B, float* C,int N, int M,int P) {
	// indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// block shared memory
	__shared__ float A_s[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
	__shared__ float B_s[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

	// loop over blocks from block row of matrix A and block column of matrix B
	float sum = 0.0;
	int numBlocks = (P + SHARED_BLOCK_SIZE - 1) / SHARED_BLOCK_SIZE;
	for (int m = 0; m < numBlocks; m++) {

		// copy block from matrix to shared memory
		int c = m * SHARED_BLOCK_SIZE + threadIdx.x;
		int r = m * SHARED_BLOCK_SIZE + threadIdx.y;

		A_s[threadIdx.y][threadIdx.x] = A[INDEX(row, c, P)];
		B_s[threadIdx.y][threadIdx.x] = B[INDEX(r, col, M)];

		// ******************* BARRIER SYNC ************************
		__syncthreads();

		// length of this part of row-column product is SHARED_BLOCK_SIZE
		// except for last block when it may be smaller
		int K = (m == numBlocks - 1 ? P - m * SHARED_BLOCK_SIZE : SHARED_BLOCK_SIZE);

		// compute this part of row-column product
		for (int k = 0; k < K; k++) {
			sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
		}

		// ******************* BARRIER SYNC ************************
		__syncthreads();
	}

	// all done; store computed element in matrix c
	if (row < N && col < M)
		C[row * M + col] = sum;
}


__global__ void transposeSmem(float *in,float *out, int nrows, int ncols) {
		// static shared memory
	__shared__ float tile[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

	// coordinate matrice originale
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// indice lineare della mat nella global memory
	unsigned int offset = INDEX(row, col, ncols);

	// trasferimento dati dalla global memory alla shared memory
	if (row < nrows && col < ncols)
		tile[threadIdx.y][threadIdx.x] = in[offset];

	// thread synchronization
	__syncthreads();

	// indici del thread nel blocco trasposto
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// NOTE - need to transpose row and col on block and thread-block level:
	// 1. swap blocks x-y
	// 2. swap thread x-y assignment (irow and icol calculations above)
	// note col still has continuous threadIdx.x -> coalesced gst
	col = blockIdx.y * blockDim.y + icol;
	row = blockIdx.x * blockDim.x + irow;

	// indice lineare globale di memoria nella matrice trasposta
	unsigned int transposed_offset = INDEX(row, col, nrows);

	// NOTA: controlli invertiti nelle dim riga colonna
	if (row < ncols && col < nrows)
		out[transposed_offset] = tile[icol][irow]; // NOTE icol,irow not irow,icol
}


__global__ void transposeNaiveRow(float* in, float* out, int ny, int nx) {
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if ((ix < nx) && (iy < ny)) {
		out[(ix*ny) + iy] = in[(iy*nx) + ix];
	}
}

__global__ void matrix_mean(float* in, float* out, int in_row, int in_col){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i = 0;
	float val = 0;
	if((col < in_col)){
		val = 0;
		for(i=0; i<in_row; i++){
			val += in[i * in_col + col];
		}
		out[col] = val/in_row;
	}
}

__global__ void vector_prod(float* in_a, float* in_b, float* out, int n, int m){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row < n) && (col < m)){
		for(int i=0; i<m; i++){
			out[(row*m)+i] = in_a[row] * in_b[i];
		}		
	}
}

__global__ void diff_matr_vect(float* in_matr, float* in_vect, float* out_matr, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n*m)
		out_matr[idx] = in_matr[idx] - in_vect[idx%m];
}

__global__ void matrix_prod(float* in_a, float* in_b, float* out, int n, int m,int p){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row < n) && (col < m)){
		int i = 0;	
		float val = 0;
		for(i=0; i<p; i++){
			val +=  in_a[(row * p) + i] * in_b[(i * m) + col];
		}
		out[(row * m) + col] = val;
		
	}
}

__global__ void div_by_scalar(float* in_a, float scalar, float* out, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n*m)
		out[idx] = in_a[idx] / scalar;
}

__global__ void diff_vect(float* in_a, float* in_b, float* out, int v_size, int n_matrix) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < v_size)
		out[idx] = in_a[idx] - in_b[idx];
}



