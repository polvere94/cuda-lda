/*
Comando di compilazione
--> nvcc -arch=sm_20 .\plda.cu  -lcublas -lcusolver -o plda

Esempio di esecuzione
--> .\plda 1024 0 0
numero righe: 1024


@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/
#include "plda.h"
#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <math.h>
#include <string.h>
#include "matrix_op.h"


/*
	Esegue il calcolo 
*/
void new_projection(float* ev_v, int ev_rows, int ev_cols,int n_matrix, Matrix* old_matrix, Matrix* new_matrix, int n_lines, int n_feature){
	//Devo moltiplicare i dati X gli n_matrix-1 autovettori non nulli (i maggiori)
	int new_n_feature, new_matrix_cols;
	int n_streams, i, h, j;
	float *d_eigv, **d_new_data, *h_eigenvectors;
	
	n_streams = N_MATRIX;
	new_n_feature = n_matrix-1;
	new_matrix_cols = new_n_feature;

	
	// Allocazione memoria host
    h_eigenvectors = (float*)malloc(ev_rows*new_matrix_cols*sizeof(float));
	d_new_data = (float**)malloc(n_matrix*sizeof(float*));

	// Allocazione memoria device
    CHECK(cudaMalloc((void**)&d_eigv, ev_rows*new_matrix_cols*sizeof(float)));
    for(i=0; i<n_matrix; i++)
		CHECK(cudaMalloc((void**)&d_new_data[i], n_lines*new_matrix_cols*sizeof(float)));

    // Prendo solo gli n_matrix-1 autovettori (con autovalore maggiore)
    for(i=0; i<ev_rows; i++){
    	h=0; 
    	for(j=ev_cols-1; j>=(ev_cols-new_matrix_cols); j--){
    		h_eigenvectors[i*new_matrix_cols+h] = ev_v[i*ev_cols+j];
    		h++;
    	}
    }
   	//print_matrix(h_eigenvectors, ev_rows, n_matrix-1, "Autovavettori considerati");
  
    CHECK(cudaMemcpy(d_eigv, h_eigenvectors, ev_rows*new_matrix_cols*sizeof(float), cudaMemcpyHostToDevice));

	int rows_mat_a = n_lines;
	int row_mat_b = ev_rows;
	int cols_mat_b = new_matrix_cols;

	dim3 blockDim(SHARED_BLOCK_SIZE,SHARED_BLOCK_SIZE);
	dim3 gridDim((new_n_feature + blockDim.x - 1) / blockDim.x, (n_lines + blockDim.y - 1) / blockDim.y);

	cudaStream_t streams[N_MATRIX];
	for (i=0; i<n_streams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	
	for(i=0; i<n_matrix; i++){
		
		//	Prodotto tra matrice dati e autovettori
		mat_prod_shared<<<gridDim, blockDim, 0, streams[i]>>>(old_matrix[i].data, d_eigv, d_new_data[i], rows_mat_a, cols_mat_b, row_mat_b);
		
		CHECK(cudaMemcpyAsync(new_matrix[i].data, d_new_data[i], n_lines*new_matrix_cols*sizeof(float) ,cudaMemcpyDeviceToHost,streams[i]));
	}

	for (i=0; i<n_streams; i++) {
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
			if((n_matrix-1)>1){
				for(j=0;j<2;j=j++){
					fprintf(temp_1, "%lf ", matrix[h].data[i*n_col+j]); 
				}
			}else{
				fprintf(temp_1, "%lf ", matrix[h].data[i*n_col]); 
				fprintf(temp_1,"0 "); 
			}
			fprintf(temp_1, "%d \n",h);
		}
	}
	

	FILE* temp_2 = fopen("data/data_2.temp", "w");
	for(h=0; h<n_matrix; h++){
		for(i=0; i<n_row;i++){
			for(j=0;j<2;j++){
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

	for(i=0; i<n_matrix; i++){
		//Fare differenza tra media locale e media globale (sono due vettori)
		
		diff_vect<<<(n_feature+1)/BLOCK_SIZE_32, BLOCK_SIZE_32,0,streams[i]>>>(d_means[i], d_global_mean, d_temp1[i], n_feature);
		
		//Devo motliplicare le due matrici UNO' X UNO
		vector_prod<<<blocks_grid, thread_block, 0, streams[i]>>>(d_temp1[i], d_temp1[i], d_sb_temp[i], n_feature, n_feature);
		
		add_matrix<<<blocks_grid2, thread_block2, 0, streams[i]>>>(d_sb, d_sb_temp[i], d_sb, n_feature, n_feature);		
	}

	for (i = 0; i < n_streams; i++) {
		CHECK(cudaStreamSynchronize(streams[i]));
	}
	
	/*float *temp = (float*) malloc(n_feature*sizeof(float));
	CHECK(cudaMemcpy(temp,d_global_mean,n_feature*sizeof(float),cudaMemcpyDeviceToHost));
	print_matrix(temp, 1, n_feature,"vector_prod");*/


	//	free della memoria
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

	//	creazione degli stream
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

	// free della memoria
	cudaFree(d_temp_sw);
	cudaFree(d_temp_sw2);
	cudaFree(d_temp_sw_t);
}