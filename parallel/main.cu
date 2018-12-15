/*
Comando di compilazione
--> nvcc -arch=sm_20 .\plda.cu .\main.cu -lcublas -lcusolver -o plda

Esempio di esecuzione
--> .\plda 1024
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
#include "plda.h"
#include "reader.c"


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

	// Creazione hanlde cursolver e cublas
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
	// Allocazione della memoria pinned per ogni classe (matrice)
	matrix = (Matrix*)malloc(n_matrix*sizeof(Matrix));

	d_data = (float**)malloc(n_matrix*sizeof(float*)); 
	d_means = (float**)malloc(n_matrix*sizeof(float*)); 
	
	for(i=0; i<n_matrix; i++){

		//	Allocazione memoria host come pinned
		CHECK(cudaMallocHost((void**)&matrix[i].data, n_lines*n_feature*sizeof(float)));
		CHECK(cudaMallocHost((void**)&matrix[i].mean, n_feature*sizeof(float)));

		//	Allocazione memoria device
		CHECK(cudaMalloc((void**)&d_data[i], n_lines*n_feature*sizeof(float)));
		CHECK(cudaMalloc((void**)&d_means[i], n_feature*sizeof(float)));
	}
	//	lettura dei dati dai file
	read_data_file(matrix, n_matrix, n_lines, n_feature);

	//	printf("Dati letti\n");
	//	print_matrix(matrix[2].data,n_lines,n_feature,"Matrice");	
	
	//Creazione degli stream per la sovrapposizione di operazioni I/O e GPU
	int n_streams = n_matrix;
	cudaStream_t streams[N_MATRIX];
	for (i=0; i<n_streams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	for (i=0; i<n_streams; i++) {
		CHECK(cudaMemcpyAsync(d_data[i], matrix[i].data, n_lines*n_feature*sizeof(float), cudaMemcpyHostToDevice,streams[i]));
		matrix_mean<<<(n_feature+1)/BLOCK_SIZE_32, BLOCK_SIZE_32, 0, streams[i]>>>(d_data[i], d_means[i], n_lines, n_feature);
		CHECK(cudaMemcpyAsync(matrix[i].mean, d_means[i], n_feature*sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
	}

	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}
	
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

	//	Per stampare a video la media totale

	/*float* t_mean = (float*) malloc(1*n_feature*sizeof(float));
	CHECK(cudaMemcpy(t_mean,d_tmeans,n_feature*sizeof(float),cudaMemcpyDeviceToHost));
	print_matrix(t_mean,1,n_feature,"media");*/


	/**************************************
			Calcolo between-scatter matrix
	***************************************/
	float* d_sb;
	CHECK(cudaMalloc((void**)&d_sb, n_feature*n_feature*sizeof(float)));
	
	between_scatter_matrix(n_matrix, d_means, d_tmeans, n_feature, d_sb);
	
	// Per stampare a video la matrice SB

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
	

	//	Per stampare a video la matrice Within-scattter matrix

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
	
	//	Prodotto tra  inversa di SW e SB
    matrix_prod<<<((n_feature*n_feature)+1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_invsw, d_sb, d_invsw_by_sb, n_feature, n_feature, n_feature);
	
	//	Per stampare a video la matrice inversa di SW per SB

    //print_matrix(h_invsw_by_sb, n_feature, n_feature, "inversa di SW per SB");

   
    /*********************************************
		Calcolo autovalori e autovettori
    **********************************************/
	float V[n_feature*n_feature]; //	Conterra gli autovettori
    float W[n_feature]; //	Conterra gli autovalori
   
    float *d_W, *d_work;
    int *devInfo, lwork = 0;

    CHECK(cudaMalloc((void**)&d_W, n_feature*sizeof(float)));
    CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	
    //	Tipologia di operazione
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
 
 	//	Sizing del buffer
    cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, n_feature, d_invsw_by_sb, n_feature, d_W, &lwork);
    CHECK(cudaMalloc((void**)&d_work, lwork*sizeof(float)));

	// Computazione
    cusolverDnSsyevd(cusolverH, jobz, uplo, n_feature, d_invsw_by_sb, n_feature, d_W, d_work, lwork, devInfo);
   
    CHECK(cudaMemcpy(W, d_W, n_feature*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(V, d_invsw_by_sb, n_feature*n_feature*sizeof(float), cudaMemcpyDeviceToHost));
   
    //print_matrix((float)W, 1, n_feature, "\n\nAutovalori, ordine ascendente\n");

    //	Il risultato è trasposto  VET' * VAL * inv(VET')
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

	/********************************
    		Calcolo nuova proiezione
 	*********************************/
    new_projection(h_vectors, n_feature, n_feature, n_matrix, matrix, new_matrix, n_lines, n_feature);
   
    
    /**********************
    		Timing
 	***********************/
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);
    printf("Tempo: %lf\n", et);

   	/*****************************
    		Plot dei nuovi dati
 	******************************/
	plot(new_matrix, n_matrix, n_lines, n_matrix-1, matrix, n_lines,n_feature);

	// Free della memoria allocata
  	for(i=0; i<n_matrix; i++){
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
