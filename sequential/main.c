/*
Istruzione di compilazione
-> gcc -Wall  matrix_op.c lda.c main.c -o lda

@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "reader.c"
#include "lda.h"


int main(int argc, char *argv[]){

	if(argc!=2){
		printf("Numero di parametri errato");
		exit(-1);
	}
	/********************************************
			Inizializzazione variabili
	*********************************************/
    clock_t t; 
	int unit = 1;
	int i;
	int n_row = atoi(argv[1]);
	int n_col = FEATURE_NUMBER;
	Matrix* M;
	float* tmean_vect;
	
	

	/****************************
			Inizio timing
	*****************************/
    t = clock(); 

	
	tmean_vect = init_matrix(unit, n_col);
	// Alloca memoria per ogni matrice
	M = (Matrix*) malloc(N_MATRIX*sizeof(Matrix));
	for(i=0; i<N_MATRIX; i++){
		M[i].data= (float*)malloc(n_row*n_col*sizeof(float));
		M[i].mean= (float*)malloc(1*n_col*sizeof(float));
	}

	/***************************************
			Lettura del dataset
	****************************************/
	read_data_file(M, N_MATRIX, n_row, FEATURE_NUMBER);

	/********************************************************
			Calcolo della media totale (somma dei vettori media)
	**********************************************************/
	for(i=0; i<N_MATRIX; i++){
		mean(M[i].data, n_row, n_col, M[i].mean);
	}
	sum_vectors_mean(M, N_MATRIX, n_col, tmean_vect);

	//Calcolo media globale
	for(int y=0; y<n_col; y++){
		tmean_vect[y] = tmean_vect[y] / N_MATRIX;
	}
	

	/****************************************************
				Calcolo Between-class scatter matrix
	*****************************************************/
	float* bw = between_scatter_matrix(M, N_MATRIX, n_row, n_col, tmean_vect);

	/****************************************************
				Calcolo Within-class scatter matrix
	
	*****************************************************/
	float* sw = within_scatter_matrix(M,N_MATRIX,n_row,n_col);


	int size_sw = n_col;
	int size_sb = n_col;

	/****************************************
			Calcolo matrice inversa di SW
			e prodotto tra invSW e SB
	*****************************************/
	float* inv_sw = inv_matrix(sw,size_sw);
	float* invsw_by_sb = init_matrix(size_sw,size_sw);
	prod(inv_sw, bw, size_sw, size_sb, size_sw, size_sb, invsw_by_sb);

	//print_matrix(invsw_by_sb, size_sw, size_sw);

	/*************************************************************
				Calcolo degli autovalori e autovettori
	**************************************************************/
	float* w =(float*)malloc(size_sw*sizeof(float));
	float* v =(float*)malloc(size_sw*size_sw*sizeof(float));

	Lsvd(invsw_by_sb, size_sw, w, v);


	/*************************************************************
				Trasformazione dello spazio di iniziale 
				tramite (n_matrix-1) autovettori con i rispettivi
				autovalori massimi
	**************************************************************/
	float* h_eigenvectors;
	int rows, cols, h, j;
	rows = size_sw;
   	cols = size_sw;

   	h_eigenvectors = (float*)malloc(size_sw*(N_MATRIX-1)*sizeof(float));

    for(i=0; i<rows; i++){
    	h=0;
    	for(j=0; j<(N_MATRIX-1); j++){
    		h_eigenvectors[i*(N_MATRIX-1)+h] = v[i*cols+j];
    		h++;
    	}    	
    }
    
    Matrix* new_matrix = (Matrix*) malloc(N_MATRIX*sizeof(Matrix));
	for(i=0; i<N_MATRIX; i++){
		new_matrix[i].data = (float*)malloc(n_row*(N_MATRIX-1)*sizeof(float));
		new_matrix[i].mean = (float*)malloc(1*1*sizeof(float));

		prod(M[i].data, h_eigenvectors, n_row, rows, n_col,N_MATRIX-1, new_matrix[i].data);	
	}


    /**********************************
    			Timing finale
    **********************************/
    t = clock() - t;
    double time_taken = ((double)t)/(double)(CLOCKS_PER_SEC/1000); // in seconds
    printf("%lf\n", time_taken); 
  	

	/*************************************************************
				Plot dei risultati
	**************************************************************/
	plot(new_matrix, N_MATRIX, n_row, (N_MATRIX-1),M, n_row, n_col);

	for(i=0; i<N_MATRIX; i++){
		free(M[i].data);
		free(M[i].mean);
		free(new_matrix[i].data);
		free(new_matrix[i].mean);
	}
	free(M);
	free(new_matrix);
	free(h_eigenvectors);
	free(w);
	free(v);	


	return 0;
}