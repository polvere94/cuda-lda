#include "matrix_op.h"

/*
	Dependent on the dataset
	Dipendenti dal dataset

*/
#define N_MATRIX 3
#define N_FEATURE 128
#define BLOCK_SIZE 64
#define BLOCK_SIZE_32 32


/*
	Effettua il calcolo della Between Scatter Matrix
	
	n_matrix:		numero di matrici
	d_means:		medie delle matrici (device memory)
	d_global_mean:	vettore media totale (device memory)
	n_feature:		numero di colonne delle matrici
	d_sb:			matrice di output (n_feature X n_feature)
*/
void between_scatter_matrix(int n_matrix, float** d_means, float* d_global_mean, int n_feature, float* d_sb);

/*
	Effettua il calcolo della Within Scatter Matrix
	
	n_matrix:	numero di matrici
	matrix:		matrici in ingresso (device memory)
	d_means:	medie delle matrici (device memory)
	n_lines:	numero di righe delle matrici
	n_feature:	numero di colonne delle matrici
	d_sw:		matrice di output (n_feature X n_feature)
*/
void within_scatter_matrix(int n_matrix,float** d_data, float** d_means,int n_lines, int n_feature,float* d_sw);

/*
	Effettua il calcolo della nuova proiezione

	ev_v:		matrice di autovettori
	ev_rows:	numero di righe di ev_v
	ev_cols:	numero di colonne di ev_v
	n_matrix:	numero di matrici
	old_matrix:	vettore delle matrici di partenza
	new_matrix:	vettore delle matrici di output
	n_lines:	numero di righe di new_matrix
	n_feature:	numero di colonne di new_matrix
*/

void new_projection(float* ev_v, int ev_rows, int ev_cols,int n_matrix, Matrix* old_matrix, Matrix* new_matrix, int n_lines, int n_feature);

/*
	Crea un grafico doppio dei dati e della nuova proiezione
	
	matrix:				vettore delle nuove matrici
	n_matrix:			numero di matrici considerate
	n_row:				numero righe di una singola matrice
	n_col:				numero colonne di una singola matrice
	original_matrix:	vettore delle matrici di partenza
	or_n_row:			numero di righe matrice di partenza
	or_n_col:			numero di colonne matrice di partenza
*/
void plot(Matrix* matrix, int n_matrix, int n_row, int n_col,Matrix* original_matrix, int or_n_row, int or_n_col);