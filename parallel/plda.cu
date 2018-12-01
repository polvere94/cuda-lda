/*
Comando di compilazione
--> nvcc -arch=sm_20 .\plda.cu  -lcublas -lcusolver -o plda

Esempio di esecuzione
--> .\plda 1024 0 0
numero righe: 1024


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
#include "matrix_op.cu"
#include <errno.h>
#include <time.h>
#include "reader.c"

#define N_MATRIX 3
#define N_FEATURE 128
#define BLOCK_SIZE 64
#define BLOCK_SIZE_32 32

void read_data_file(Matrix* matrix, int n_matrix, int n_lines, int n_feature);
void between_scatter_matrix(int n_matrix, float** d_means, float* d_global_mean, int n_feature, float* d_sb);
void within_scatter_matrix(int n_matrix,float** d_data, float** d_means,int n_lines, int n_feature,float* d_sw);
void new_projection(float* ev_v, int ev_rows, int ev_cols,int n_matrix, Matrix* old_matrix, Matrix* new_matrix, int n_lines, int n_feature);
void plot(Matrix* matrix, int n_matrix, int n_row, int n_col,Matrix* original_matrix, int or_n_row, int or_n_col);

/*
	Prefissi standard:
			- h_ 	variabili in memoria host
		    - d_ 	variabili in memoria device
		   	- n_%	cardinalità di %
*/
int main(int argc, char *argv[]){

	/********************************************
			Inizializzazione variabili
	*********************************************/
	FILE* ftiming;
	const int n_matrix = N_MATRIX;
	const long n_feature = N_FEATURE;
	int n_lines, i;
	float** d_data, **d_means;
	Matrix* matrix;

	// Creazione hanlde cursolver
	cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    
	if(argc!=2){
		printf("Numero di parametri errato");
		exit(-1);
	}
	n_lines = atoi(argv[1]);

	ftiming = fopen("stats/timing.csv", "a");
	if (ftiming == NULL){
    	printf("Errore apertura file per il timing\n");
    	exit(-1);
	}
	fprintf(ftiming, "\n");

	/******************************
			Inizio timing
	*******************************/
	float et;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);

	
	/**************************************
			calcolo la media delle matrici
	**************************************/
	// Alloca memoria pinned per ogni classe (matrice)
	matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));

	d_data = (float**)malloc(n_matrix*sizeof(float*)); 
	d_means = (float**)malloc(n_matrix*sizeof(float*)); 
	
	for(i=0; i<n_matrix; i++){

		//	allocazione memoria host
		CHECK(cudaMallocHost((void**)&matrix[i].data, n_lines*n_feature*sizeof(float)));
		CHECK(cudaMallocHost((void**)&matrix[i].mean, n_feature*sizeof(float)));

		//	allocazione memoria device
		CHECK(cudaMalloc((void**)&d_data[i], n_lines*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_means[i], n_feature*sizeof(float)));
	}
	//	lettura dei file dove sono presenti i dati
	read_data_file(matrix, n_matrix, n_lines, n_feature);

	//	printf("Dati letti\n");
	//	print_matrix(matrix[2].data,n_lines,n_feature,"Matrice");	
	
	//Creazione degli stream per la sovrapposizione di operazioni I/O e GPU

	 // Calculate the time taken by fun() 
   /* clock_t t; 
    t = clock();     */

	//	creazione degli stream
	int n_streams = n_matrix;
	cudaStream_t streams[N_MATRIX];
	for (i=0; i<n_streams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	dim3 threadPerBlock(n_feature,1);
	dim3 blocksPerGrid(n_feature, 1);



	for (i=0; i<n_streams; i++) {
		CHECK(cudaMemcpyAsync(d_data[i], matrix[i].data, n_lines*n_feature*sizeof(float), cudaMemcpyHostToDevice,streams[i]));
		matrix_mean<<<(n_feature+1)/BLOCK_SIZE_32, BLOCK_SIZE_32, 0, streams[i]>>>(d_data[i], d_means[i], n_lines, n_feature);
		CHECK(cudaMemcpyAsync(matrix[i].mean, d_means[i], n_feature*sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
	}
	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}
	/*for(int u=0;u<n_matrix;u++)
		print_matrix(matrix[u].mean,1,n_feature,"media");
	*/

	//fprintf(ftiming, "%f;",ms);


	CHECK(cudaDeviceSynchronize());
	
	/********************************************************
			Calcolo media totale (somma dei vettori media)
	**********************************************************/
	float* d_tmeans;
	
	CHECK(cudaMalloc((void**)&d_tmeans, 1*n_feature*sizeof(float)));
	CHECK(cudaMemset(d_tmeans, 0, 1*n_feature*sizeof(float))); 	
	
	for(int i=0;i<n_matrix;i++){
		add_vectors<<<((n_feature)+1)/BLOCK_SIZE_32,BLOCK_SIZE_32>>>(d_tmeans, d_means[i], d_tmeans, n_feature*sizeof(float));
	}
	CHECK(cudaDeviceSynchronize());


	div_by_scalar<<<((n_feature)+1)/BLOCK_SIZE_32,BLOCK_SIZE_32>>>(d_tmeans, n_matrix, d_tmeans, n_feature*sizeof(float),1);

	/*float* t_mean = (float*) malloc(1*n_feature*sizeof(float));
	CHECK(cudaMemcpy(t_mean,d_tmeans,n_feature*sizeof(float),cudaMemcpyDeviceToHost));
	print_matrix(t_mean,1,n_feature,"media");*/


	/**************************************
			Calcolo between-scatter matrix
	***************************************/
	float* d_sb;
	CHECK(cudaMalloc((void**)&d_sb, n_feature*n_feature*sizeof(float)));
	
	between_scatter_matrix(n_matrix, d_means, d_tmeans, n_feature, d_sb);
	
	// per stampare la matrice SB

	/*float* temp = (float*) malloc(n_feature*n_feature*sizeof(float));
	CHECK(cudaMemcpy(temp, d_sb, n_feature*n_feature*sizeof(float),cudaMemcpyDeviceToHost));
	print_matrix(temp,n_feature,n_feature,"SB");
	free(temp);*/


	/*******************************************
			Calcolo della Within-scatter matrix
	********************************************/
	float* d_sw;
	CHECK(cudaMalloc((void**)&d_sw,n_feature*n_feature*sizeof(float)));
	CHECK(cudaMemset(d_sw,0,n_feature*n_feature*sizeof(float)));
	
	within_scatter_matrix(n_matrix,d_data, d_means, n_lines, n_feature,d_sw);
	

	// per stampare la matrice SW

	/*float* temp = (float*) malloc(n_feature*n_feature*sizeof(float));
	CHECK(cudaMemcpy(temp,d_sw,n_feature*n_feature*sizeof(float),cudaMemcpyDeviceToHost));
	print_matrix(temp,n_feature,n_feature,"SW");
	free(temp);*/


    /********************************************
    		Calcolo matrice inversa di SW
    ********************************************/
    float* d_invsw;
    CHECK(cudaMalloc((void**)&d_invsw,n_feature*n_feature*sizeof(float)));
    invert_device(handle,d_sw, d_invsw, n_feature);
   
    float* d_invsw_by_sb;
    float* h_invsw_by_sb = (float*)malloc(n_feature*n_feature*sizeof(float));
	CHECK(cudaMalloc((void**)&d_invsw_by_sb,n_feature*n_feature*sizeof(float)));
	//cudaMemcpy(d_invsw, h_invsw, n_feature*n_feature*sizeof(float), cudaMemcpyHostToDevice);
	
    matrix_prod<<<((n_feature*n_feature)+1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_invsw, d_sb, d_invsw_by_sb, n_feature, n_feature, n_feature);
	


   	CHECK(cudaMemcpy(h_invsw_by_sb, d_invsw_by_sb, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost));
	
	CHECK(cudaDeviceSynchronize());
    //print_matrix(h_invsw_by_sb, n_feature, n_feature, "inversa di SW per SB");

   

    /******************************************
		Calcolo autovalori e autovettori
    **********************************************/
	double V[n_feature*n_feature]; //	conterra gli autovettori
    double W[n_feature]; //	conterra gli autovalori
    double A[n_feature*n_feature];

    //	conversione di tipo, da float a double
    for(i = 0; i < n_feature; i++){
    	for(int j = 0; j < n_feature; j++){
    		A[i*n_feature+j] = (double)h_invsw_by_sb[i*n_feature+j];
    	}
    }

    double *d_A, *d_W, *d_work;
    int *devInfo, lwork = 0;

    CHECK(cudaMalloc((void**)&d_A, sizeof(double) * n_feature * n_feature));
    CHECK(cudaMalloc((void**)&d_W, sizeof(double) * n_feature));
    CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	CHECK(cudaMemcpy(d_A, A, sizeof(double) * n_feature * n_feature, cudaMemcpyHostToDevice));

    // tipologia di operazione
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
 
 	// sizing del buffer
    cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, n_feature, d_A, n_feature, d_W, &lwork);
    CHECK(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

	// Computazione
    cusolverDnDsyevd(cusolverH, jobz, uplo, n_feature, d_A, n_feature, d_W, d_work, lwork, devInfo);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(W, d_W, n_feature*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(V, d_A, n_feature*n_feature*sizeof(double), cudaMemcpyDeviceToHost));
   
    //print_matrix((float)W, 1, n_feature, "\n\nAutovalori, ordine ascendente\n");


    // IMPORTANTE: il risultato è trasposto!  VET' * VAL * inv(VET')
    float* h_vectors = (float*)malloc(n_feature*n_feature*sizeof(float));

    //printf("\n\nAutovettori \n");
    //printf("[");
    for(int i = 0 ; i<n_feature ; i++){
    	for(int j=0; j<n_feature; j++){
    		h_vectors[i*n_feature+j] = V[j*n_feature+i];
        	//printf("%.04f ", h_vectors[i*n_feature+j]);
    	}
        //printf(";\n");
    }
    //printf("]");

    // Allocazione della matrice risultato
    Matrix* new_matrix = (Matrix*) malloc(n_matrix*sizeof(Matrix));
    for(int i=0; i<n_matrix; i++){
		CHECK(cudaMallocHost((void**)&new_matrix[i].data, n_lines*(n_matrix-1)*sizeof(float)));
		CHECK(cudaMallocHost((void**)&new_matrix[i].mean, 1*n_feature*sizeof(float)));
	}

	// Calcolo nuova proiezione
    new_projection(h_vectors, n_feature, n_feature, n_matrix, matrix, new_matrix, n_lines, n_feature);
   
    
    /**********************
    		timing
 	***********************/
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);



    //fprintf(ftiming, "tempo %lf\n", time_taken);
    printf("Tempo: %lf\n", et);
	//fclose(ftiming);

    

    // Plotting dei risultati
	plot(new_matrix, n_matrix, n_lines, n_matrix-1, matrix, n_lines,n_feature);

	// Free della memoria allocata
  	for(i=0;i<n_matrix;i++){
    	cudaFreeHost(matrix[i].data);
    	cudaFreeHost(matrix[i].mean);
    	cudaFreeHost(new_matrix[i].data);
    	cudaFreeHost(new_matrix[i].mean);
    }
    cudaFreeHost(new_matrix);
    cudaFreeHost(matrix);
    
	CHECK(cudaDeviceReset());

	return 0;
}


/*
	TODO: il problema del crash è qui
*/
void new_projection(float* ev_v, int ev_rows, int ev_cols,int n_matrix, Matrix* old_matrix, Matrix* new_matrix, int n_lines, int n_feature){
	//Devo moltiplicare i dati X gli n_matrix-1 autovettori non nulli (i maggiori)
	
	int n_streams = N_MATRIX, i, h, j;
	float *d_eigv, **d_new_data, *h_eigenvectors;
	
	int new_n_feature = n_matrix-1;
	int new_matrix_cols = new_n_feature;
	int new_matrix_rows = n_lines;
	int old_matrix_cols = n_feature;
	int old_matrix_rows = n_lines;

	
	// allocazione memoria host
    h_eigenvectors = (float*)malloc(ev_rows*new_matrix_cols*sizeof(float));
	d_new_data = (float**)malloc(n_matrix*sizeof(float*));

	// allocazione memoria device
    CHECK(cudaMalloc((void**)&d_eigv, ev_rows*ev_cols*sizeof(float)));
    for(i=0; i<n_matrix; i++)
		CHECK(cudaMalloc((void**)&d_new_data[i], n_lines*new_matrix_cols*sizeof(float)));

    // prende solo gli n_matrix-1 autovettori (con autovalore maggiore)
    for(i=0; i<ev_rows; i++){
    	h=0; 
    	for(j=ev_cols-1; j>=ev_cols-3; j--){
    		h_eigenvectors[i*new_matrix_cols+h] = ev_v[i*ev_cols+j];
    		h++;
    	}
    }

   	//print_matrix(h_eigenvectors, ev_rows, n_matrix-1, "Autovavettori considerati");
  
    CHECK(cudaMemcpy(d_eigv, h_eigenvectors, ev_rows*new_matrix_cols*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

	int rows_mat_a = n_lines;
	//int cols_mat_a = n_feature;
	int row_mat_b = ev_rows;
	int cols_mat_b = new_matrix_cols;

	dim3 blockDim(SHARED_BLOCK_SIZE,SHARED_BLOCK_SIZE);
	dim3 gridDim((new_n_feature + blockDim.x - 1) / blockDim.x, (n_lines + blockDim.y - 1) / blockDim.y);

	cudaStream_t streams[N_MATRIX];
	for (i=0; i<n_streams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	//TODO verificare dimensioni matrici
	for(i=0; i<n_matrix; i++){
		mat_prod_shared<<<gridDim, blockDim, 0, streams[i]>>>(old_matrix[i].data, d_eigv, d_new_data[i], rows_mat_a, cols_mat_b, row_mat_b);
		//matrix_prod<<<gridDim, blockDim, 0, streams[i]>>>(old_matrix[i].data, d_eigv, d_new_data[i],rows_mat_a, cols_mat_b, row_mat_b);
		CHECK(cudaMemcpyAsync(new_matrix[i].data, d_new_data[i], n_lines*new_matrix_cols*sizeof(float) ,cudaMemcpyDeviceToHost,streams[i]));
	}

	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}

	cudaFree(d_new_data);
	cudaFree(d_eigv);
	free(h_eigenvectors);
}

void plot(Matrix* matrix, int n_matrix, int n_row, int n_col, Matrix* original_matrix, int or_n_row, int or_n_col){
	#define NUM_COMMANDS 5

	int i,h, j = 0;

	char * commandsForGnuplot[] = {
		"set multiplot layout 2,1 rowsfirst",
		"set label 'LDA'", 
		"plot 'data/data_2.temp' u 1:2:3  title 'dati'  pointtype 5 linecolor  palette",
		"set label 'LDA'", 
		"plot 'data/data_1.temp' u 1:2:3  title 'dati'  pointtype 5 linecolor  palette"
	};

	FILE* gnuplotPipe = _popen ("gnuplot -persistent", "w");

	FILE* temp_1 = fopen("data/data_1.temp", "w");

	for(h=0; h<n_matrix; h++){
		for(i=0; i<n_row;i++){
			for(j=0;j<4;j=j+3){
				fprintf(temp_1, "%lf ", matrix[h].data[i*n_col+j]); 
			}
			fprintf(temp_1, "%d \n",h);
		}
	}
	

	FILE* temp_2 = fopen("data/data_2.temp", "w");
	for(h=0; h<n_matrix; h++){
		for(i=0; i<n_row;i++){
			for(j=0;j<4;j=j+3){
				fprintf(temp_2, "%lf ", original_matrix[h].data[i*or_n_col+j]); 
			}
			fprintf(temp_2, "%d \n",h);
		}
	}
	
	for (i=0; i < NUM_COMMANDS; i++){
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]);
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
	int n_streams = n_matrix;
	dim3 thread_block2(BLOCK_SIZE,1);
	dim3 blocks_grid2((n_feature*n_feature)+1/BLOCK_SIZE, 1);

	dim3 thread_block(1,n_feature);
	dim3 blocks_grid(1, 1);

	CHECK(cudaMemset(d_sb, 0, n_feature*n_feature*sizeof(float))); 

	float **d_temp1 = (float**)malloc(N_MATRIX*sizeof(float*));
	float **d_sb_temp = (float**)malloc(N_MATRIX*sizeof(float*));
	for(i=0;i<N_MATRIX;i++){
		// Allocazione memoria device
		CHECK(cudaMalloc((void**)&d_temp1[i], 1*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_sb_temp[i], n_feature*n_feature*sizeof(float)));
	}

	cudaStream_t streams[N_MATRIX];
	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	for(i=0; i<N_MATRIX; i++){
		//Fare differenza tra media locale e media globale (sono due vettori)
		
		diff_vect<<<(n_feature+1)/BLOCK_SIZE_32, BLOCK_SIZE_32,0,streams[i]>>>(d_means[i], d_global_mean, d_temp1[i], n_feature, 1);
		
		//Devo motliplicare le due matrici UNO' X UNO
		vector_prod<<<(n_feature+1)/BLOCK_SIZE_32, BLOCK_SIZE_32, 0, streams[i]>>>(d_temp1[i], d_temp1[i], d_sb_temp[i], n_feature, n_feature);
		
		add_matrix<<<blocks_grid2, thread_block2, 0, streams[i]>>>(d_sb, d_sb_temp[i], d_sb, n_feature, n_feature);		
	}

	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}
	/*
		float *temp = (float*) malloc(n_feature*n_feature*sizeof(float));
		CHECK(cudaMemcpy(temp,d_sb,n_feature*n_feature*sizeof(float),cudaMemcpyDeviceToHost));
		print_matrix(temp, n_feature, n_feature,"vector_prod");*/
	
	for(i=0;i<N_MATRIX;i++){
		CHECK(cudaFree(d_temp1[i]));
		CHECK(cudaFree(d_sb_temp[i]));
	}
	free(d_temp1);
	free(d_sb_temp);
}

void within_scatter_matrix(int n_matrix,float** d_data, float** d_means,int n_lines, int n_feature,float* d_sw){
	int i, n_streams;
	float **d_temp_sw, **d_temp_sw2, **d_temp_sw_t;	
	dim3 bg(1,1);
	dim3 th1(n_feature,n_lines);
	dim3 th2(n_feature,n_feature);

	int rows_d_temp_sw2 = n_feature;
	int cols_d_temp_sw2 = n_feature;
	int rows_d_temp_sw_t = n_feature;
	//int cols_d_temp_sw_t = n_lines;
	int rows_d_temp_sw = n_lines;
	int cols_d_temp_sw = n_feature;

	//Stream??
	n_streams = N_MATRIX;
	cudaStream_t streams[N_MATRIX];
	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	d_temp_sw = (float**)malloc(N_MATRIX*sizeof(float*));
	d_temp_sw2 = (float**)malloc(N_MATRIX*sizeof(float*));
	d_temp_sw_t = (float**)malloc(N_MATRIX*sizeof(float*));
	
	for(i=0;i<N_MATRIX;i++){
		CHECK(cudaMalloc((void**) &d_temp_sw[i], n_lines*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**) &d_temp_sw2[i], n_feature*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**) &d_temp_sw_t[i], n_feature*n_lines*sizeof(float)));
	}

	// execution configuration
	dim3 block(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
	dim3 grid((n_feature + block.x - 1) / block.x, (n_lines + block.y - 1) / block.y);
	dim3 blockDim(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
	dim3 gridDim((cols_d_temp_sw2+ blockDim.x - 1)/blockDim.x, (rows_d_temp_sw2 + blockDim.y - 1)/blockDim.y);

	//TODO: come sistemo gli stream??
	for(i=0;i<N_MATRIX; i++){
		//Differenza tra la la matrice ed il vettore, riga per riga
		diff_matr_vect<<<((n_feature*n_feature)+1)/BLOCK_SIZE, BLOCK_SIZE, 0, streams[i]>>>(d_data[i], d_means[i], d_temp_sw[i], n_lines, n_feature);

		transposeSmem<<<grid, block, 0, streams[i]>>>(d_temp_sw[i], d_temp_sw_t[i], n_lines, n_feature);

		mat_prod_shared<<<gridDim, blockDim,0,streams[i]>>>(d_temp_sw_t[i], d_temp_sw[i], d_temp_sw2[i], rows_d_temp_sw_t, cols_d_temp_sw, rows_d_temp_sw);
	}

	

	for(i=0; i<N_MATRIX; i++){
		div_by_scalar<<<((n_feature*n_feature)+1)/BLOCK_SIZE,BLOCK_SIZE,0,streams[i]>>>(d_temp_sw2[i], n_lines-1, d_temp_sw2[i], n_feature, n_feature);
	
		add_matrix<<<((n_feature*n_feature)+1)/BLOCK_SIZE,BLOCK_SIZE,0,streams[i]>>>(d_sw, d_temp_sw2[i], d_sw, n_feature, n_feature);
	}

	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}

	cudaFree(d_temp_sw);
	cudaFree(d_temp_sw2);
	cudaFree(d_temp_sw_t);
}