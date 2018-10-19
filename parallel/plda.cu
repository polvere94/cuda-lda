#include <stdio.h>

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
__global__ void diff_vect(float* in_a, float* in_b, float* out, int v_size, int n_matrix) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < v_size)
		out[idx] = in_a[idx] - in_b[idx];
}

// n e m dimensione matrice output
__global__ void matrix_prod(float* in_a, float* in_b, float* out, int n, int m){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i = 0;
	float val = 0;
	if((row < n) && (col < m)){
		val = 0;
		for(i=0; i<m; i++){
			val += in_a[row * m + i] * in_b[i * m + col];
		}
		out[row * m + col] = val;
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


__global__ void transposeSmem(float* in, float* out, int n_row, int n_col){
	__shared__ float tile[BDMY][BDMX];

	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int offset =  INDEX(row, col, n_col);

	if(row < n_row && col < n_col){
		tile[threadIdx.y][threadIdx.x] = in[offset];
		printf("%f\n",tile[threadIdx.y][threadIdx.x] );
	}
	__syncthreads();

	unsigned int bidx, irow, icol;

	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	col = blockIdx.y * blockDim.y + icol;
	row = blockIdx.x * blockDim.x + irow;
	printf("Trasposta\n");
	unsigned int transposed_offset = INDEX(row,col, n_row);
	if(row < n_col && col < n_row)
		out[transposed_offset] = tile[icol][irow];

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


	cudaMalloc((void**)&d_tmeans, 1*n_feature*sizeof(float));

	float* d_means_v;
	for(int i=0; i<n_matrix; i++){
		//Fare differenza tra media locale e media globale (sono due vettori)
		diff_vect<<<1, n_feature>>>(d_means[i], d_tmeans,OUT!!!, 1, n_feature);
		//Devo trasporre la matrice risultante.

		//Devo motliplicare le due matrici UNO x DUE
	}

	//SOMMO TUTTE LE MATRICI USCENTI DALLE MOLTIPLICAZIONE ED OTTENGO SB


	return 0;
}