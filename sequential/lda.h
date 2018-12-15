#include "matrix_op.h"

#define N_MATRIX 3
#define FEATURE_NUMBER 128
/*
	Effettua il calcolo della Beetween Scatter Matrix

	matrix:		matrici in ingresso
	n_matrix:	numero di matrici
	n_row:		numero di righe di una singola matrice
	n_col:		numero colonne di una singola matrice
	means:		vettore delle medie delle matrici
*/
float* between_scatter_matrix(Matrix* matrix, int n_matrix, int n_row, int n_col, float* mean);

/*
	Effettua il calcolo della Within Scatter Matrix

	matrix:		matrici in ingresso
	n_matrix:	numero di matrici
	n_row:		numero di righe di una singola matrice
	n_col:		numero colonne di una singola matrice
*/
float* within_scatter_matrix(Matrix* matrix, int n_matrix, int n_row, int n_col);

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
void plot(Matrix* matrix, int n_matrix, int n_row, int n_col, Matrix* original_matrix, int or_n_row, int or_n_col);