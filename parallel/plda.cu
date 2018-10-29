/*
Compile instruction
-> nvcc -arch=sm_20 .\plda.cu  -lcublas -lcusolver

@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/

#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <math.h>
#include <string.h>

#define BDMX 16
#define BDMY 16

#define INDEX(ros,col,stride) (row * stride + col)
#define CHECK(call) { \
		const cudaError_t error = call; \
		if (error != cudaSuccess) { \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

void print_matrix(float* in, int n, int m, char* label){
	int i,j;
	printf("\n%s\n",label);
	for(i =0;i<n;i++){
		for(j=0;j<m;j++){
			if(j==m-1)
				printf("%.04f;", in[(i*m)+j]);
			else
				printf("%.04f ", in[(i*m)+j]);
		}
		printf("\n");
	}
	printf("\n");
}

typedef struct Matrix_{
    float* data;
    float* mean;
} Matrix;

void plot(Matrix* matrix, int n_matrix, int n_row, int n_col);
void invert(float* src, float* dst, int n);
void invert_device(float* src_d, float* dst_d, int n);
/*
 * kernel: somma di matrici
 */
__global__ void add_matrix(float* in_a, float* in_b, float* out,int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if ((idx < n*m))
		out[idx] = in_a[idx] + in_b[idx];
}

/*
 * kernel: somma dei vettori
 */
__global__ void add_vect(float* in_a, float* in_b, float* out, int v_size, int n_matrix) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < v_size)
		out[idx] = in_a[idx] + in_b[idx];
}

/*
 * kernel: differenza dei vettori
 */
__global__ void div_matr_scal(float* in_a, float scalar, float* out, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n*m)
		out[idx] = in_a[idx] / scalar;
}
/*
 * kernel: differenza dei vettori
 */
__global__ void diff_vect(float* in_a, float* in_b, float* out, int v_size, int n_matrix) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < v_size)
		out[idx] = in_a[idx] - in_b[idx];
}

/*
 * kernel: differenza tra vettore e matrice
 */
__global__ void diff_matr_vect( float* in_matr, float* in_vect, float* out_matr, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n*m)
		out_matr[idx] = in_matr[idx] - in_vect[idx%m];
}

#define IDX(i,j,n) (i*n+j)
// n e m dimensione matrice output
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

//Prodotto vettore colonna per vettore riga
// n e m dimensione matrice output
__global__ void vector_prod(float* in_a, float* in_b, float* out, int n, int m){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if((row < n) && (col < m)){
		int i = 0;	
		float val = 0;
		for(i=0; i<m; i++){
			val = in_a[row] * in_b[i];
			out[(m*row)+i] = val;
		}		
	}
}


/*
	Calcola il vettore media di una matrice rispetto alle colonne
	matrix_mean: in   vettore in input
				 out  vettore di output
				 in_row   numero righe della matrice di input
				 in_col  numero colonne della matrice di input (=colonne matrice output)

*/
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

/*
	Calcola il vettore media di una matrice rispetto alle colonne
	matrix_mean: in   vettore in input
				 out  vettore di output
				 in_row   numero righe della matrice di input
				 in_col  numero colonne della matrice di input (=colonne matrice output)

*/
__global__ void between_scatter_matrix(float* in, float* out, int in_row, int in_col){
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




// case 0 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float* in, float* out, int ny, int nx) {
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if ((ix < nx) && (iy < ny)) {
		out[(ix*ny) + iy] = in[(iy*nx) + ix];
	}
}


__global__ void mat_prod_shared(float* A, float* B, float* C) {
	// indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// block shared memory
	__shared__ float A_s[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_s[BLOCK_SIZE][BLOCK_SIZE];

	// loop over blocks from block row of matrix A and block column of matrix B
	float sum = 0.0;
	int numBlocks = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
	for (int m = 0; m < numBlocks; m++) {

		// copy block from matrix to shared memory
		int c = m * BLOCK_SIZE + threadIdx.x;
		int r = m * BLOCK_SIZE + threadIdx.y;

		A_s[threadIdx.y][threadIdx.x] = A[IDX(row, c, P)];
		B_s[threadIdx.y][threadIdx.x] = B[IDX(r, col, M)];

		// ******************* BARRIER SYNC ************************
		__syncthreads();

		// length of this part of row-column product is BLOCK_SIZE
		// except for last block when it may be smaller
		int K = (m == numBlocks - 1 ? P - m * BLOCK_SIZE : BLOCK_SIZE);

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


void read_data_file(Matrix* matrix, int n_matrix, int n_lines){
	char* filenames[3];
	filenames[0] = "data/dataset1.txt";
	filenames[1] = "data/dataset2.txt";
	filenames[2] = "data/dataset3.txt";
	FILE *file;
	float a,b,c,d=0;
	int h;
	int i=0;
	int col = 4; //fissato
	for(i=0; i<n_matrix; i++){		
		h = 0;
		file = fopen(filenames[i], "r");
		while (fscanf(file, "%f %f %f %f", &a, &b, &c, &d) != EOF) {
			matrix[i].data[(h*col)+0]=a; 
			matrix[i].data[(h*col)+1]=b;
			matrix[i].data[(h*col)+2]=c;
			matrix[i].data[(h*col)+3]=d;
			if(h==n_lines-1)
				break;
			h++;
		}
	}
}



void between_scatter_matrix(){


}
void within_scatter_matrix(){

}
/*
	Standard.
	prefix h_ stand for host memory variable
		   d_ stand for device memory variable
		   n_%	number of %
*/
int main(void){

	/********************************************
			Inizializzazione variabili
	*********************************************/
	cublasHandle_t handle;
	cublasCreate(&handle);
	int version;
	cublasGetVersion(handle, &version);
	printf("\nUsing CUBLAS Version: %d\n", version);

	//Fase 1) calcolo la media delle matrici
	int i;


	/* Lettura matrici dai file*/
	Matrix* matrix;
	const int n_matrix = 3;
	const int n_lines = 50;
	const int n_feature = 4;
	
	// Alloca memoria per ogni matrice
	matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));
	for(int i=0; i<n_matrix; i++){
		cudaMallocHost((void**)&matrix[i].data, n_feature*n_lines*sizeof(float));
		cudaMallocHost((void**)&matrix[i].mean, 1*n_feature*sizeof(float));
	}

	// Leggi il dataser dai file
	read_data_file(matrix, n_matrix, n_lines);


	float** d_data;
	float** d_means;
	d_data = (float**)malloc(n_matrix*sizeof(float*)); 
	d_means = (float**)malloc(n_matrix*sizeof(float*)); 

	for(i=0;i<n_matrix;i++){
		cudaMalloc((void**)&d_data[i], n_lines*n_feature*sizeof(float));
		cudaMalloc((void**)&d_means[i], 1*n_feature*sizeof(float));
	}
		
	int n_streams = n_matrix;
	cudaStream_t streams[3];
	for (i = 0; i < n_streams; ++i) {
		cudaStreamCreate(&streams[i]);
	}

	//dim3 threadPerBlock(n_feature,1);
	//dim3 blocksPerGrid(1, 1);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
			(N + blockDim.y - 1) / blockDim.y);

	float ms;
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent, 0);
	for (i = 0; i < n_streams; i++) {
		cudaMemcpyAsync(d_data[i], matrix[i].data, n_lines*n_feature*sizeof(float), cudaMemcpyHostToDevice,streams[i]);
		matrix_mean<<<threadPerBlock, threadPerBlock, 0, streams[i]>>>(d_data[i], d_means[i], n_lines, n_feature);
		cudaMemcpyAsync(matrix[i].mean, d_means[i], 1*n_feature*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
	}
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&ms, startEvent, stopEvent);
	printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);

	for (int i = 0; i < n_streams; i++) {
		cudaStreamSynchronize(streams[i]);
	}

	//Calcolo media totale (somma dei vettori media)
	
	float* d_tmeans;
	float* t_mean = (float*) malloc(1*n_feature*sizeof(float));
	cudaMalloc((void**)&d_tmeans, 1*n_feature*sizeof(float));

	cudaMemset(d_tmeans, 0, 1*n_feature*sizeof(float)); 	
	for(int i=0;i<n_matrix;i++){
		add_vect<<<1,n_feature>>>(d_tmeans, d_means[i], d_tmeans, 1*n_feature*sizeof(float), n_matrix);
	}

	cudaMemcpy(t_mean, d_tmeans, 1*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	//Divido per il numero di classi
	for(int i=0; i<n_feature; i++){
		t_mean[i]=t_mean[i] / n_matrix;
	}

	cudaMemcpy(d_tmeans, t_mean, 1*n_feature*sizeof(float), cudaMemcpyHostToDevice);
	

	print_matrix(t_mean, 1, n_feature, "Media totale");
	

	/**************************************
		Calcolo between-scatter matrix
	***************************************/
	dim3 thread_block(1,n_feature);
	dim3 blocks_grid(1, 1);


	float* h_sb_temp = (float*) malloc(1*n_feature*sizeof(float));
	float** d_temp1;
	d_temp1 = (float**)malloc(n_matrix*sizeof(float*)); 

	float** d_sb_temp;
	d_sb_temp = (float**)malloc(n_matrix*sizeof(float*)); 
	
	//Inizializzazione
	for(i=0;i<n_matrix;i++){
		cudaMalloc((void**)&d_temp1[i], 1*n_feature*sizeof(float));
		cudaMalloc((void**)&d_sb_temp[i], n_feature*n_feature*sizeof(float));
	}

	for(int i=0; i<n_matrix; i++){
		//Fare differenza tra media locale e media globale (sono due vettori)
		diff_vect<<<1, n_feature>>>(d_means[i], d_tmeans, d_temp1[i], n_feature, 1);
		//Devo trasporre la matrice risultante.
		vector_prod<<<blocks_grid, thread_block>>>(d_temp1[i], d_temp1[i], d_sb_temp[i], n_feature, n_feature);
		//Devo motliplicare le due matrici UNO x DUE
	}
	cudaDeviceSynchronize();
	cudaMemcpy(h_sb_temp, d_sb_temp[1], n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	

	//SOMMO TUTTE LE MATRICI USCENTI DALLE MOLTIPLICAZIONE ED OTTENGO SB
	float* d_sb;
	float* h_sb = (float*)malloc(n_feature*n_feature*sizeof(float));
	cudaMalloc((void**)&d_sb, n_feature*n_feature*sizeof(float));
	cudaMemset(d_sb, 0, n_feature*n_feature*sizeof(float)); 

	dim3 thread_block2(n_feature*n_feature,1);
	dim3 blocks_grid2(1, 1);
	for(int i=0; i<n_matrix; i++){
		add_matrix<<<blocks_grid2, thread_block2>>>(d_sb, d_sb_temp[i], d_sb, n_feature, n_feature);
	}

	cudaDeviceSynchronize();

	cudaMemcpy(h_sb, d_sb, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int i = 0;i<n_matrix;i++){
		cudaFree(&d_sb_temp[i]);
	}

	free(d_sb_temp);

	
	print_matrix(h_sb, n_feature, n_feature, "Between-scatter matrix");

		
	//TODO: liberare memoria delle operazioni di SB




	/*******************************************
			Calcolo della Within-scatter matrix
	********************************************/

	//Calcolo covarianza
	float** d_temp_sw;
	d_temp_sw = (float**)malloc(n_matrix*sizeof(float*));
	for(i=0;i<n_matrix;i++)
		cudaMalloc((void**)&d_temp_sw[i], n_lines*n_feature*sizeof(float));

	//Differenza tra la la matrice ed il vettore, riga per riga
	for(int i=0;i<n_matrix; i++){
		diff_matr_vect<<<1, n_lines * n_feature>>>(d_data[i], d_means[i], d_temp_sw[i], n_lines, n_feature);
	}
	cudaDeviceSynchronize();

	float** d_temp_sw2;
	float** d_temp_sw_t;
	d_temp_sw2 = (float**)malloc(n_matrix*sizeof(float*));
	for(i=0;i<n_matrix;i++)
		cudaMalloc((void**)&d_temp_sw2[i], n_feature*n_feature*sizeof(float));

	d_temp_sw_t = (float**)malloc(n_matrix*sizeof(float*));
	for(i=0; i<n_matrix; i++)
		cudaMalloc((void**)&d_temp_sw_t[i], n_feature*n_lines*sizeof(float));

	dim3 bg(1,1);
	dim3 th1(n_feature,n_lines);
	dim3 th2(n_feature,n_feature);
	for(int i=0;i<n_matrix; i++){
		transposeNaiveRow<<<bg,th1>>>(d_temp_sw[i], d_temp_sw_t[i], n_lines, n_feature);
		matrix_prod<<<bg,th2>>>(d_temp_sw_t[i], d_temp_sw[i], d_temp_sw2[i], n_feature, n_feature,n_lines);
		div_matr_scal<<<bg,n_feature*n_feature>>>(d_temp_sw2[i], n_lines-1, d_temp_sw2[i], n_feature, n_feature);
	}
	cudaDeviceSynchronize();
	
	float* d_sw;
	cudaMalloc((void**)&d_sw,n_feature*n_feature*sizeof(float));
	cudaMemset(d_sw,0,n_feature*n_feature*sizeof(float));
	
	for(int i=0; i<n_matrix; i++){
		add_matrix<<<1, n_feature*n_feature>>>(d_sw, d_temp_sw2[i], d_sw, n_feature, n_feature);
	}
	cudaDeviceSynchronize();
	float* h_sw = (float*)malloc(n_feature*n_feature*sizeof(float));
	cudaMemcpy(h_sw, d_sw, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	

	print_matrix(h_sw, n_feature, n_feature, "Whitin-scatter matrix");

	/***********************
		Ho trovato SW!!!
		Ora devo calcolare la matrice inversa di SW e poi usare SVD per trovare gli autovalori e autovettori associati
	************************/
	
   


    float h_invsw[n_feature*n_feature];
    float* a = h_sw;

    /***********
    Calcolo matrice inversa di SW
    ***********/
    invert(a, h_invsw,n_feature);

    print_matrix(h_invsw,n_feature,n_feature,"Matrice inversa di SW");


    float* d_invsw;
    float* d_invsw_by_sb;
    float* h_invsw_by_sb = (float*)malloc(n_feature*n_feature*sizeof(float));
	cudaMalloc((void**)&d_invsw,n_feature*n_feature*sizeof(float));
	cudaMalloc((void**)&d_invsw_by_sb,n_feature*n_feature*sizeof(float));
	cudaMemcpy(d_invsw, h_invsw, n_feature*n_feature*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 th3(n_feature,n_feature);
    matrix_prod<<<1,th3>>>(d_invsw, d_sb, d_invsw_by_sb, n_feature, n_feature, n_feature);
	cudaDeviceSynchronize();

    cudaMemcpy(h_invsw_by_sb, d_invsw_by_sb, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	
	
    print_matrix(h_invsw_by_sb, n_feature, n_feature, "inversa di SW per SB");



    /******************************************
		Calcolo autovalori e autovettori
    **********************************************/
	cusolverDnHandle_t cusolverH = NULL;
	const int lda = n_feature;
	const int m = n_feature;
	double V[lda*m]; // eigenvectors
    double W[m]; // eigenvalues

    //float A1[lda*m] = { 3.5, 0.5, 0, 0.5, 3.5, 0, 0, 0, 2.0};
    double A[lda*m];

    // Conversione di tipo, da float a double
    for(int i=0;i<m;i++){
    	for(int j=0;j<lda;j++){
    		A[i*m+j] = (double)h_invsw_by_sb[i*m+j];
    	}
    }

    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;

    // step 1: create cusolver/cublas handle
    cusolverDnCreate(&cusolverH);

    cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaMalloc ((void**)&devInfo, sizeof(int));
	cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnDsyevd_bufferSize(cusolverH,jobz,uplo, m, d_A,lda,d_W, &lwork);

    cudaMalloc((void**)&d_work, sizeof(double)*lwork);

	// step 4: compute spectrum
    cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo);
    cudaDeviceSynchronize();
   

    cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
   

    printf("\n\nAutovalori, ordine ascendente\n");
    printf("[");
    for(int i = 0 ; i < m ; i++){
        printf("%.04f ", W[i]);
    }
    printf("]");

    //print_matrix((float)W, 1, m, "Autovalori");


    // IMPORTANTE: il risultato è trasposto!  VET' * VAL * inv(VET')
    float* h_v = (float*)malloc(m*lda*sizeof(float));

     printf("\n\nAutovettori \n");
     printf("[");
    for(int i = 0 ; i < m ; i++){
    	for(int j=0; j<lda; j++){
    		h_v[i*lda+j] = V[j*m+i];
        	printf("%.04f ", h_v[i*lda+j]);
    	}
        printf(";\n");
    }
    printf("]");

    //print_matrix(V, lda, m, "Autovettori");

    //Devo moltiplicare i dati X gli n_matrix-1 autovettori non nulli (i maggiori)
    float* h_eigenvectors = (float*)malloc(m*(n_matrix-1)*sizeof(float));

    // 4x4
   	int rows = m;
   	int cols = lda;
    for(int i=0; i<rows; i++){
    	int h=0;
    	for(int j=0; j<2; j++){
    		h_eigenvectors[i*(n_matrix-1)+h] = h_v[i*cols+j];
    		h++;
    	}    	
    }

    print_matrix(h_eigenvectors,rows,n_matrix-1,"Autovavettori considerati");
    float* d_eigv;

    cudaMalloc((void**)&d_eigv,rows*(n_matrix-1)*sizeof(float));
    cudaMemcpy(d_eigv,h_eigenvectors,rows*(n_matrix-1)*sizeof(float),cudaMemcpyHostToDevice);

	Matrix* new_matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));
	for(int i=0; i<n_matrix; i++){
		new_matrix[i].data = (float*)malloc(n_lines*(n_matrix-1)*sizeof(float));
		new_matrix[i].mean = (float*)malloc( 1*(n_matrix-1)*sizeof(float));
	}
    float** d_new_data = (float**)malloc(n_matrix*sizeof(float*));
	for(i=0; i<n_matrix; i++)
		cudaMalloc((void**)&d_new_data[i], n_lines*(n_matrix-1)*sizeof(float));

	dim3 thh(n_matrix-1,n_lines);
	for(int i=0;i<n_matrix; i++){
		matrix_prod<<<1,thh>>>(matrix[i].data, d_eigv, d_new_data[i], n_lines, n_matrix-1, n_feature);
	}
	cudaDeviceSynchronize();

	for(int i=0; i<n_matrix; i++){
		//print_matrix(matrix[i].data,n_lines,n_matrix-1,"Vecchia matrice");
		cudaMemcpy(new_matrix[i].data,d_new_data[i],n_lines*(n_matrix-1)*sizeof(float),cudaMemcpyDeviceToHost);
		print_matrix(new_matrix[i].data,n_lines,n_matrix-1,"Nuova matrice");
	}

	 plot(matrix,n_matrix, 4, 2);


	// plot(new_matrix,n_matrix, n_lines, n_matrix-1);

	 //, V[3], V[3*m]
    plot(new_matrix,n_matrix, n_lines, n_matrix-1);


	return 0;
}

void invert(float* src, float* dst, int n){
    float* src_d, *dst_d;

    cudaMalloc<float>(&src_d,n * n * sizeof(float));
    cudaMemcpy(src_d,src, n * n * sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc<float>(&dst_d,n * n * sizeof(float));

    invert_device(src_d,dst_d,n);

    cudaMemcpy(dst,dst_d,n * n * sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(src_d), cudaFree(dst_d);
}

void invert_device(float* src_d, float* dst_d, int n){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    int batchSize = 1;

    int *P, *INFO;

    cudaMalloc<int>(&P, n * batchSize * sizeof(int));
    cudaMalloc<int>(&INFO,batchSize * sizeof(int));

    int lda = n;

    float *A[] = { src_d };
    float** A_d;
    cudaMalloc<float*>(&A_d,sizeof(A));
    cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice);

    cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize);

    int INFOh = 0;
    cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost);

    if(INFOh != 0)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    float* C[] = { dst_d };
    float** C_d;
    cudaMalloc<float*>(&C_d,sizeof(C));
    cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice);

    cublasSgetriBatched(handle,n,(const float **)A_d,lda,P,C_d,n,INFO,batchSize);

    cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost);

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);
}

void plot(Matrix* matrix, int n_matrix, int n_row, int n_col){
	int i,h, j = 0;
	#define NUM_COMMANDS 2
	char * commandsForGnuplot[] = {"set title 'LDA - Fisher Iris'", 
	"plot 'data/data.temp' u 1:2:3  title 'dati'  pointtype 5 linecolor  palette"};
	
	FILE * gnuplotPipe = _popen ("gnuplot -persistent", "w");

	FILE * temp = fopen("data/data.temp", "w");

	for(h=0; h<n_matrix; h++){
		for(i=0; i < n_row;i++){
			for(j=0;j<2;j++){
				fprintf(temp, "%lf ", matrix[h].data[i*n_col+j]); 
			}
			fprintf(temp, "%d \n",h);
		}
	}
/*
	FILE * temp_line = fopen("data/line.temp", "w");

	for(int i=-10;i<=25;i++){

		fprintf(temp_line, "%lf ", point_x*i);
		fprintf(temp_line, "%lf ", point_y*i);
		fprintf(temp_line,"\n");
	}
	 
			*/

	for (i=0; i < NUM_COMMANDS; i++){
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]);
	}

}