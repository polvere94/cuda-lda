#include <stdio.h>
#include <cuda.h>
#include "cublas_v2.h"

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

typedef struct Matrix_{
    float* data;
    float* mean;
} Matrix;

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

// case 0 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float* in, float* out, int ny, int nx) {
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if ((ix < nx) && (iy < ny)) {
		out[(ix*ny) + iy] = in[(iy*nx) + ix];
	}
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

/*
	Standard.
	prefix h_ stand for host memory variable
		   d_ stand for device memory variable
		   n_%	number of %
*/
int main(void){

	cublasHandle_t handle;
	cublasCreate(&handle);
	int version;
	cublasGetVersion(handle, &version);
	printf("\nUsing CUBLAS Version: %d\n", version);

	//Fase 1) calcolo la media delle matrici
	int i,j;


	/* Lettura matrici dai file*/
	Matrix* matrix;
	int n_matrix = 3;
	int n_lines = 50;
	int n_feature = 4;
	
	// Alloca memoria per ogni matrice
	matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));
	for(int i=0; i<n_matrix; i++){
		cudaMallocHost((void**)&matrix[i].data, n_feature*n_lines*sizeof(float));
		cudaMallocHost((void**)&matrix[i].mean, 1*n_feature*sizeof(float));

		/*matrix[i].data = (float*)malloc(n_lines*n_feature*sizeof(float));
		matrix[i].mean = (float*) malloc(1*n_feature*sizeof(float));*/
	}

	read_data_file(matrix, n_matrix, n_lines);


	float** d_data;
	float** d_means;
	d_data = (float**)malloc(n_matrix*sizeof(float*)); 
	d_means = (float**)malloc(n_matrix*sizeof(float*)); 

	//Inizializzazione
	for(i=0;i<n_matrix;i++){
		cudaMalloc((void**)&d_data[i], n_lines*n_feature*sizeof(float));
		cudaMalloc((void**)&d_means[i], 1*n_feature*sizeof(float));
	}
		
	int n_streams = n_matrix;
	cudaStream_t streams[3];
	for (i = 0; i < n_streams; ++i) {
		cudaStreamCreate(&streams[i]);
	}

	dim3 threadPerBlock(n_feature,1);
	dim3 blocksPerGrid(1, 1);

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
	

	printf("\nMEDIA TOTALE\n");
	for(j=0; j<n_feature; j++){
		printf("%.4f ", t_mean[j]);
	}
	printf("\n");
	


	/*Calcolo di Sb (Between class Matrix)
	 per ogni classe calcolare:
	
	SB(i) = Ns .* (m_s - m)*(m_s - m)';
	SB = SBi + SB(i+1)+ ... + SB(n-1)
	
				Calcolo Between-class scatter matrix
	*/
	dim3 thread_block(1,n_feature);
	dim3 blocks_grid(1, 1);


	float* h_sb_temp = (float*) malloc(1*n_feature*sizeof(float));
	float** d_temp1;
	d_temp1 = (float**)malloc(n_matrix*sizeof(float*)); 

	//float* d_sb_temp;
	float** d_sb_temp;
	d_sb_temp = (float**)malloc(n_matrix*sizeof(float*)); 
	
	//Inizializzazione
	for(i=0;i<n_matrix;i++){
		cudaMalloc((void**)&d_temp1[i], 1*n_feature*sizeof(float));
		cudaMalloc((void**)&d_sb_temp[i], n_feature*n_feature*sizeof(float));
	}
	//cudaMalloc((void**)&d_temp1, 1*n_feature*sizeof(float));
	//cudaMalloc((void**)&d_sb_temp, n_feature*n_feature*sizeof(float));
	
	for(int i=0; i<n_matrix; i++){
		//Fare differenza tra media locale e media globale (sono due vettori)
		diff_vect<<<1, n_feature>>>(d_means[i], d_tmeans, d_temp1[i], n_feature, 1);
		//Devo trasporre la matrice risultante.
		vector_prod<<<blocks_grid, thread_block>>>(d_temp1[i], d_temp1[i], d_sb_temp[i], n_feature, n_feature);
		//Devo motliplicare le due matrici UNO x DUE
	}
	cudaMemcpy(h_sb_temp, d_sb_temp[1], n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	//Divido per il numero di classi

	//matrice n_feature X n_feature
	for(int j=0; j<n_feature; j++){
		for(int i=0; i<n_feature; i++){
			printf("%.3f ",h_sb_temp[(j*n_feature)+i]);
		}
		printf("\n");
	}
	printf("\n\n\n");
	//SOMMO TUTTE LE MATRICI USCENTI DALLE MOLTIPLICAZIONE ED OTTENGO SB
	float* d_sb;
	float* h_sb = (float*)malloc(n_feature*n_feature*sizeof(float));
	cudaMalloc((void**)&d_sb,n_feature*n_feature*sizeof(float));
	cudaMemset(d_sb, 0, n_feature*n_feature*sizeof(float)); 

	dim3 thread_block2(n_feature*n_feature,1);
	dim3 blocks_grid2(1, 1);
	for(int i=0; i<n_matrix; i++){
		add_matrix<<<blocks_grid2, thread_block2>>>(d_sb, d_sb_temp[i], d_sb, n_feature, n_feature);
	}

	cudaMemcpy(h_sb, d_sb, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();

	// Matrice n_feature X n_feature
	for(int j=0; j<n_feature; j++){
		for(int i=0; i<n_feature; i++){
			printf("%.3f ",h_sb[(j*n_feature)+i]);
		}
		printf("\n");
	}

	for(int i = 0;i<n_matrix;i++){
		cudaFree(&d_sb_temp[i]);
	}
	free(d_sb_temp);
	

	/********************
	variabile d_sb contiene Between-scatter matrix
	********************/


	/********************
	Calcolo della Within-scatter matrix

	((setosa - m_s'))'*((setosa - m_s')))./50
	sw = n.*( c_s + c_vi + c_ve); Somma di covarianze

	********************/

	//Calcolo covarianza
	float** d_temp_sw;
	d_temp_sw = (float**)malloc(n_matrix*sizeof(float*));
	for(i=0;i<n_matrix;i++)
		cudaMalloc((void**)&d_temp_sw[i], n_lines*n_feature*sizeof(float));
	//Differenza tra la la matrice ed il vettore, riga per riga
	for(int i=0;i<n_matrix; i++){
		diff_matr_vect<<<1, n_lines * n_feature>>>(d_data[i], d_means[i], d_temp_sw[i], n_lines, n_feature);
	}

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
		printf("QUI\n");
		matrix_prod<<<bg,th2>>>(d_temp_sw_t[i], d_temp_sw[i], d_temp_sw2[i], n_feature, n_feature,n_lines);
		//add_matrix<<<blocks_grid2, thread_block2>>>(d_sb, d_sb_temp[i], d_sb, n_feature, n_feature);
		//vector_prod<<<blocks_grid, thread_block>>>(d_temp1[i], d_temp1[i], d_sb_temp[i], n_feature, n_feature)
		div_matr_scal<<<bg,n_feature*n_feature>>>(d_temp_sw2[i], n_lines-1, d_temp_sw2[i], n_feature, n_feature);
	}
	cudaDeviceSynchronize();
	
	float* d_sw;
	cudaMalloc((void**)&d_sw,n_feature*n_feature*sizeof(float));
	cudaMemset(d_sw,0,n_feature*n_feature*sizeof(float));
	
	for(int i=0; i<n_matrix; i++){
		add_matrix<<<1, n_feature*n_feature>>>(d_sw, d_temp_sw2[i], d_sw, n_feature, n_feature);
	}

	float* h_temp_sw = (float*)malloc(n_feature*n_feature*sizeof(float));
	cudaMemcpy(h_temp_sw, d_sw, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	//Divido per il numero di classi

	// Matrice n_feature X n_feature
	printf("\n\n");
	for(int j=0; j<n_feature; j++){
		for(int i=0; i<n_feature; i++){
			printf("%.4f ",h_temp_sw[(j*n_feature)+i]);
		}
		printf("\n");
	}

	/***********************
		Ho trovato SW!!!
		Ora devo calcolare la matrice inversa di SW e poi usare SVD per trovare gli autovalori e autovettori associati
	************************/


	return 0;
}