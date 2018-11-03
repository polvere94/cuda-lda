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
__global__ void add_vectors(float* in_a, float* in_b, float* out, int v_size, int n_matrix) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < v_size)
		out[idx] = in_a[idx] + in_b[idx];
}

/*
 * kernel: differenza dei vettori
 */
__global__ void div_by_scalar(float* in_a, float scalar, float* out, int n, int m) {
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

/*
	Calcola il vettore media di una matrice rispetto alle colonne
	matrix_mean: in   vettore in input
				 out  vettore di output
				 in_row   numero righe della matrice di input
				 in_col  numero colonne della matrice di input (=colonne matrice output)

*/




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


/**
	Input: 
		- int n_matrix,
		- media globale
		- medie locali
		- numero colonne
		- Gli passo un riferimento della device dove riprendere il risultato SB.
	Output:
		-d_sb. matrice quadrata, n_feature X n_feature.

*/
void between_scatter_matrix(int n_matrix, float** d_means, float* d_global_mean, int n_feature, float* d_sb){
	int i;
	float* d_sb_temp;
	float* d_temp1;
	dim3 thread_block2(n_feature*n_feature,1);
	dim3 blocks_grid2(1, 1);
	dim3 thread_block(1,n_feature);
	dim3 blocks_grid(1, 1);

	CHECK(cudaMemset(d_sb, 0, n_feature*n_feature*sizeof(float))); 

	for(i=0; i<n_matrix; i++){
		// Allocazione memoria device
		CHECK(cudaMalloc((void**)&d_temp1, 1*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_sb_temp, n_feature*n_feature*sizeof(float)));

		//Fare differenza tra media locale e media globale (sono due vettori)
		diff_vect<<<1, n_feature>>>(d_means[i], d_global_mean, d_temp1, n_feature, 1);

		//Devo motliplicare le due matrici UNO x DUE
		vector_prod<<<blocks_grid, thread_block>>>(d_temp1, d_temp1, d_sb_temp, n_feature, n_feature);
		
		add_matrix<<<blocks_grid2, thread_block2>>>(d_sb, d_sb_temp, d_sb, n_feature, n_feature);

		CHECK(cudaFree(d_temp1));
		CHECK(cudaFree(d_sb_temp));
	}
	
	printf("Fine SB\n");
}

void within_scatter_matrix(int n_matrix,float** d_data, float** d_means,int n_lines, int n_feature,float* d_sw){
	int i;
	float* d_temp_sw;
	float* d_temp_sw2;
	float* d_temp_sw_t;	
	dim3 bg(1,1);
	dim3 th1(n_feature,n_lines);
	dim3 th2(n_feature,n_feature);

	for(i=0;i<n_matrix; i++){

		CHECK(cudaMalloc((void**)&d_temp_sw, n_lines*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_temp_sw2, n_feature*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_temp_sw_t, n_feature*n_lines*sizeof(float)));

		//Differenza tra la la matrice ed il vettore, riga per riga
	
		diff_matr_vect<<<1, n_lines*n_feature>>>(d_data[i], d_means[i], d_temp_sw, n_lines, n_feature);

		transposeNaiveRow<<<bg,th1>>>(d_temp_sw, d_temp_sw_t, n_lines, n_feature);
		
		matrix_prod<<<bg,th2>>>(d_temp_sw_t, d_temp_sw, d_temp_sw2, n_feature, n_feature,n_lines);
		
		div_by_scalar<<<bg,n_feature*n_feature>>>(d_temp_sw2, n_lines-1, d_temp_sw2, n_feature, n_feature);
		
		add_matrix<<<1, n_feature*n_feature>>>(d_sw, d_temp_sw2, d_sw, n_feature, n_feature);

		cudaFree(d_temp_sw);
		cudaFree(d_temp_sw2);
		cudaFree(d_temp_sw_t);
	}
}

void new_projection(int n_matrix, float* h_v, int rows, int cols,Matrix* old_matrix, Matrix* new_matrix, int n_lines, int n_feature);


/*
	Standard.
	prefix h_ stand for host memory variable
		   d_ stand for device memory variable
		   n_%	number of %
*/
int main(int argc, char *argv[]){

	/********************************************
			Inizializzazione variabili
	*********************************************/
	FILE* ftiming = fopen("stats/timing.csv", "a");
	if (ftiming == NULL){
    	printf("Errore apertura file per il timing\n");
    	exit(1);
	}
	fprintf(ftiming, "\n");

	const int n_matrix = 3;
	const int n_feature = 4;
	int n_lines, i, version;
	Matrix* matrix;
	cublasHandle_t handle;

	// Variabili per il timing
	float ms;
	cudaEvent_t startEvent, stopEvent;

	float** d_data;
	float** d_means;


	if(argc!=2){
		printf("Numero di parametri errato");
		exit(-1);
	}
	n_lines = atoi(argv[1]);

	//Informazioni su CUBLAS
	cublasCreate(&handle);
	cublasGetVersion(handle, &version);
	printf("\nVersione CUBLAS: %d\n", version);

	
	/**************************************
			calcolo la media delle matrici
	**************************************/
	// Alloca memoria pinned per ogni classe (matrice)
	matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));

	d_data = (float**)malloc(n_matrix*sizeof(float*)); 
	d_means = (float**)malloc(n_matrix*sizeof(float*)); 
	for(int i=0; i<n_matrix; i++){
		//Allocazione memoria host
		CHECK(cudaMallocHost((void**)&matrix[i].data, n_feature*n_lines*sizeof(float)));
		CHECK(cudaMallocHost((void**)&matrix[i].mean, 1*n_feature*sizeof(float)));

		//Allocazione memoria device
		CHECK(cudaMalloc((void**)&d_data[i], n_lines*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_means[i], 1*n_feature*sizeof(float)));
	}

	// Lettura dei file dove sono presenti i dati
	read_data_file(matrix, n_matrix, n_lines);

		
	//Creazione degli stream per la sovrapposizione di operazioni I/O e GPU
	int n_streams = n_matrix;
	cudaStream_t streams[3];
	for (i = 0; i < n_streams; ++i) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	dim3 threadPerBlock(n_feature,1);
	dim3 blocksPerGrid(1, 1);

	
	CHECK(cudaEventCreate(&startEvent));
	CHECK(cudaEventCreate(&stopEvent));
	CHECK(cudaEventRecord(startEvent, 0));
	for (i = 0; i < n_streams; i++) {
		CHECK(cudaMemcpyAsync(d_data[i], matrix[i].data, n_lines*n_feature*sizeof(float), cudaMemcpyHostToDevice,streams[i]));
		matrix_mean<<<threadPerBlock, threadPerBlock, 0, streams[i]>>>(d_data[i], d_means[i], n_lines, n_feature);
		CHECK(cudaMemcpyAsync(matrix[i].mean, d_means[i], 1*n_feature*sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
	}
	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}
	CHECK(cudaEventRecord(stopEvent, 0));
	CHECK(cudaEventSynchronize(stopEvent));
	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	fprintf(ftiming, "%f;",ms);
	//CHECK(cudaDeviceSynchronize());
	
	
	
	//Calcolo media totale (somma dei vettori media)
	
	float* d_tmeans;
	//float* t_mean = (float*) malloc(1*n_feature*sizeof(float));
	CHECK(cudaMalloc((void**)&d_tmeans, 1*n_feature*sizeof(float)));
	CHECK(cudaMemset(d_tmeans, 0, 1*n_feature*sizeof(float))); 	
	
	for(int i=0;i<n_matrix;i++){
		add_vectors<<<1,n_feature>>>(d_tmeans, d_means[i], d_tmeans, n_feature*sizeof(float), n_matrix);
	}
	div_by_scalar<<<1,n_feature>>>(d_tmeans, n_matrix, d_tmeans, n_feature*sizeof(float),1);


	/**************************************
			Calcolo between-scatter matrix
	***************************************/
	CHECK(cudaEventRecord(startEvent, 0));
	float* d_sb;
	CHECK(cudaMalloc((void**)&d_sb, n_feature*n_feature*sizeof(float)));
	
	between_scatter_matrix(n_matrix, d_means, d_tmeans, n_feature,d_sb);

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stopEvent, 0));
	CHECK(cudaEventSynchronize(stopEvent));
	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	fprintf(ftiming, "%f;",ms);

	/*******************************************
			Calcolo della Within-scatter matrix
	********************************************/
	CHECK(cudaEventRecord(startEvent, 0));
	
	float* d_sw;
	CHECK(cudaMalloc((void**)&d_sw,n_feature*n_feature*sizeof(float)));
	CHECK(cudaMemset(d_sw,0,n_feature*n_feature*sizeof(float)));
	
	within_scatter_matrix(n_matrix,d_data, d_means, n_lines, n_feature,d_sw);
	
	CHECK(cudaEventRecord(stopEvent, 0));
	CHECK(cudaEventSynchronize(stopEvent));
	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	fprintf(ftiming, "%f;",ms);
    /********************************************
    		Calcolo matrice inversa di SW
    ********************************************/

    float* d_invsw;
    CHECK(cudaMalloc((void**)&d_invsw,n_feature*n_feature*sizeof(float)));
    invert_device(d_sw, d_invsw, n_feature);
   
    float* d_invsw_by_sb;
    float* h_invsw_by_sb = (float*)malloc(n_feature*n_feature*sizeof(float));
	CHECK(cudaMalloc((void**)&d_invsw_by_sb,n_feature*n_feature*sizeof(float)));
	//cudaMemcpy(d_invsw, h_invsw, n_feature*n_feature*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 th3(n_feature,n_feature);
    matrix_prod<<<1,th3>>>(d_invsw, d_sb, d_invsw_by_sb, n_feature, n_feature, n_feature);
	CHECK(cudaDeviceSynchronize());

   	CHECK(cudaMemcpy(h_invsw_by_sb, d_invsw_by_sb, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost));
	
	
    print_matrix(h_invsw_by_sb, n_feature, n_feature, "inversa di SW per SB");


    /******************************************
		Calcolo autovalori e autovettori
    **********************************************/
    // Creazione hanlde cursolver
	cusolverDnHandle_t cusolverH = NULL;
	const int lda = n_feature;
	const int m = n_feature;
	double V[lda*m]; //conterra gli autovettori
    double W[m]; // conterra gli autovalori

    double A[lda*m];

    // Conversione di tipo, da float a double
    for(int i=0;i<m;i++){
    	for(int j=0;j<lda;j++){
    		A[i*m+j] = (double)h_invsw_by_sb[i*m+j];
    	}
    }

    double *d_A;
    double *d_W;
    int *devInfo;
    double *d_work;
    int lwork = 0;

    cusolverDnCreate(&cusolverH);

    CHECK(cudaMalloc ((void**)&d_A, sizeof(double) * lda * m));
    CHECK(cudaMalloc ((void**)&d_W, sizeof(double) * m));
    CHECK(cudaMalloc ((void**)&devInfo, sizeof(int)));
	CHECK(cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice));

    // Tipologia operazione
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
 
 	// Sizing del buffer
    cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork);

    CHECK(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

	// Computazione
    cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost));
   

    printf("\n\nAutovalori, ordine ascendente\n");
    printf("[");
    for(int i = 0 ; i < m ; i++){
        printf("%.04f ", W[i]);
    }
    printf("]");

    //print_matrix((float)W, 1, m, "Autovalori");


    // IMPORTANTE: il risultato è trasposto!  VET' * VAL * inv(VET')
    float* h_vectors = (float*)malloc(m*lda*sizeof(float));

    printf("\n\nAutovettori \n");
    printf("[");
    for(int i = 0 ; i < m ; i++){
    	for(int j=0; j<lda; j++){
    		h_vectors[i*lda+j] = V[j*m+i];
        	printf("%.04f ", h_vectors[i*lda+j]);
    	}
        printf(";\n");
    }
    printf("]");

    Matrix* new_matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));
    for(int i=0; i<n_matrix; i++){
		new_matrix[i].data = (float*)malloc(n_lines*(n_matrix-1)*sizeof(float));
		new_matrix[i].mean = (float*)malloc(1*sizeof(float));
	}
	fclose(ftiming);
	// Calcolo nuova proiezione
    new_projection(n_matrix, h_vectors, m, lda,matrix,new_matrix,n_lines, n_feature);
	//plot(new_matrix,n_matrix, n_lines, n_matrix-1);

    for(i=0;i<n_matrix;i++){
    	free(new_matrix[i].data);
    	free(new_matrix[i].data);
    	free(matrix[i].mean);
    	free(matrix[i].mean);
    }
    free(new_matrix);
     free(matrix);
	return 0;
}

void new_projection(int n_matrix, float* h_v, int rows, int cols, Matrix* old_matrix, Matrix* new_matrix, int n_lines, int n_feature){
	//Devo moltiplicare i dati X gli n_matrix-1 autovettori non nulli (i maggiori)
	int i;
	float* d_eigv;
	float** d_new_data;
    float* h_eigenvectors;
	dim3 thh(n_matrix-1,n_lines);

	// allocazione memoria host
    h_eigenvectors = (float*)malloc(rows*(n_matrix-1)*sizeof(float));
	d_new_data = (float**)malloc(n_matrix*sizeof(float*));

	// allocazione memoria device
    CHECK(cudaMalloc((void**)&d_eigv,rows*(n_matrix-1)*sizeof(float)));
    for(i=0; i<n_matrix; i++)
		CHECK(cudaMalloc((void**)&d_new_data[i], n_lines*(n_matrix-1)*sizeof(float)));

    // 4x4
    // prende solo gli n_matrix-1 autovettori (con autovalore maggiore)
    for(int i=0; i<rows; i++){
    	int h=0;
    	for(int j=0; j<2; j++){
    		h_eigenvectors[i*(n_matrix-1)+h] = h_v[i*cols+j];
    		h++;
    	}    	
    }

    print_matrix(h_eigenvectors,rows,n_matrix-1,"Autovavettori considerati");
  
    CHECK(cudaMemcpy(d_eigv,h_eigenvectors,rows*(n_matrix-1)*sizeof(float),cudaMemcpyHostToDevice));

	//TODO mettere stream
	for(int i=0;i<n_matrix; i++){
		matrix_prod<<<1,thh>>>(old_matrix[i].data, d_eigv, d_new_data[i], n_lines, n_matrix-1, n_feature);
	}

	CHECK(cudaDeviceSynchronize());

	for(int i=0; i<n_matrix; i++){
		CHECK(cudaMemcpy(new_matrix[i].data, d_new_data[i], n_lines*(n_matrix-1)*sizeof(float) ,cudaMemcpyDeviceToHost));
		//print_matrix(new_matrix[i].data, n_lines, n_matrix-1, "Nuova matrice");
	}
}

void invert(float* src, float* dst, int n){
    float* src_d, *dst_d;

    CHECK(cudaMalloc<float>(&src_d,n * n * sizeof(float)));
    CHECK(cudaMemcpy(src_d,src, n * n * sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMalloc<float>(&dst_d,n * n * sizeof(float)));

    invert_device(src_d,dst_d,n);

    CHECK(cudaMemcpy(dst,dst_d,n * n * sizeof(float),cudaMemcpyDeviceToHost));

    CHECK(cudaFree(src_d));
    CHECK(cudaFree(dst_d));
}

void invert_device(float* src_d, float* dst_d, int n){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    int batchSize = 1;

    int *P, *INFO;

    CHECK(cudaMalloc<int>(&P, n * batchSize * sizeof(int)));
    CHECK(cudaMalloc<int>(&INFO,batchSize * sizeof(int)));

    int lda = n;

    float *A[] = { src_d };
    float** A_d;
    CHECK(cudaMalloc<float*>(&A_d,sizeof(A)));
    CHECK(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize);

    int INFOh = 0;
    CHECK(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
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
	for (i=0; i < NUM_COMMANDS; i++){
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]);
	}

}