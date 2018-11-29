/*
Compile instruction
-> gcc -Wall .\lda.c

@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/
#include "lda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define FEATURE_NUMBER 128
#define N_MATRIX 3

float* between_scatter_matrix(Matrix* matrix, int n_matrix, int n_row, int n_col, float* mean){
	float* accumulatore_sb, *SBc, *v_transpose, *res2;
	int i;
	
	accumulatore_sb = init_matrix(n_col, n_col);
	SBc = init_matrix(1, n_col);
	v_transpose = init_matrix(n_col, 1);
	res2 = init_matrix(n_col, n_col);

	for(i=0; i<n_matrix; i++){
		//	calcolo differenza tra vettore media matrice e vettore media totale
		diff_vector(matrix[i].mean, mean, n_col, SBc);
		
		//	calcolo della matrice trasposta
		transpose(SBc,1, n_col, v_transpose);

		//	prodotto tra matrice trasposta e matrice NON trasposta
		prod(v_transpose, SBc, n_col, 1, 1 ,n_col, res2);

		//	somma del risultato in un accumulatore di risultati di tutte le matrici
		sum_matrix(accumulatore_sb, res2, n_col, n_col);
	}

	free(SBc);
	free(v_transpose);
	free(res2);
	return accumulatore_sb;
}

float* within_scatter_matrix(Matrix* matrix, int n_matrix, int n_row, int n_col){
	float *ACC, *res, *ONE, *ONE_T;
	int y,x,h;

	ACC = init_matrix(n_col, n_col);
	memset_matrix(ACC, n_col, n_col);
	res = init_matrix(n_col, n_col);
	ONE = init_matrix(n_row, n_col);
	ONE_T = init_matrix(n_col, n_row);
	
	for(h =0; h<n_matrix; h++){
		
		//	differenza tra la la matrice dei dati ed il rispettivo vettore media
		diff_matrix_vector(matrix[h].data, matrix[h].mean, n_row, n_col, ONE);

		//	calcolo della matrice trasposta
		transpose(ONE, n_row, n_col, ONE_T);

		//	prodotto tra matrice trasposta e matrice NON trasposta
		prod(ONE_T, ONE, n_col, n_row, n_row, n_col, res); //Risultato (n_col X n_col)
		
		// divide i risultati per (n_row-1)
		for(y=0; y<n_col; y++){
			for(x=0; x<n_col; x++){
				res[(y*n_col)+x] = res[(y*n_col)+x] / (n_row-1);
			}
		}

		//	somma del risultato in un accumulatore di risultati di tutte le matrici
		sum_matrix(ACC, res, n_col, n_col);
	}

	free(res);
	free(ONE);
	free(ONE_T);
	return ACC;
}

void plot(Matrix* matrix, int n_matrix, int n_row, int n_col, Matrix* original_matrix, int or_n_row, int or_n_col){
	int i,h, j = 0;

	#define NUM_COMMANDS 5
	char * commandsForGnuplot[] = {
		"set multiplot layout 2,1 rowsfirst",
		"set label 'LDA - Fisher Iris'", 
		"plot 'data/data_2.temp' u 1:2:3  title 'dati'  pointtype 5 linecolor  palette",
		"set label 'LDA - Fisher Iris'", 
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