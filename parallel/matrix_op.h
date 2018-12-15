/*
Header file contenente intestazioni delle operazioni su matrici

@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/
#ifndef MATRIX_OP_H
#define MATRIX_OP_H
#define SHARED_BLOCK_SIZE 16
#define INDEX(rows, cols, stride) (rows * stride + cols)

#include "cublas_v2.h"

#define CHECK(call) { \
		const cudaError_t error = call; \
		if (error != cudaSuccess) { \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

/*
	Struttura di una matrice utilizzata in alcune funzioni
	data:	contiene i dati della matrice
	mean:	è un vettore che contiene la media della matrice
*/
typedef struct Matrix_{
    float* data;
    float* mean;
} Matrix;

void print_matrix(float* in, int n, int m, char* label);

/*
	Calcola la matrice inversa della matrice quadrata in input utilizzando
	le funzioni di cuBLAS

	handle:		handle già inizializzato di cuBLAS
	src_d:		matrice da invertire
	dst_d:		matrice invertita in output	
	n:			dimensione matrice (numero righe o colonne)
*/
void invert_device(cublasHandle_t handle, float* src_d, float* dst_d, int n);

/*
	Somma due vettori elemento per elemento
	
	in_a:		primo vettore in input
	in_b:		secondo vettore in input
	out:		vettore somma in output
	v_size:		dimensione dei vettori
*/
__global__ void add_vectors(float* in_a, float* in_b, float* out, int v_size);

/*
	Esegue la somma di due matrici

	in_a:	prima matrice
	in_b:	seconda matrice
	out:	matrice risultato
	n:		numero di righe delle matrici
	m:		numero di colonne delle matrici
*/
__global__ void add_matrix(float* in_a, float* in_b, float* out,int n, int m);

/*
	Calcola il protto riga per colonna utilizzando la shared memory

	A:	prima matrice
	B:	seconda matrice
	C:	matrice risultato
	N:	numero di righe matrice C
	M:	numero di colonne matrice C
	P:	numero colonne di A e righe di B
*/
__global__ void mat_prod_shared(float* A, float* B, float* C,int N, int M, int P);

/*
	Calcola la trasposta della matrice in input utilizzando la shared memory

	in:		matrice in input
	out:	matrice trasposta in output	
	nrows:	numero di righe matrice in
	ncols:	numero di colonne matrice in
*/
__global__ void transposeSmem(float *in,float *out, int nrows, int ncols);

/*
	Calcola il vettore media di una matrice rispetto alle colonne
	
	in:		vettore in input
	out:	vettore di output
	in_row:	numero righe della matrice di input
	in_col: numero colonne della matrice di input (=colonne matrice output)
*/
__global__ void matrix_mean(float* in, float* out, int in_row, int in_col);

/*
	Prodotto vettore colonna per vettore riga

	in_a:	vettore colonna (l'algoritmo lo considera vettore colonna)
	in_b:	vettore riga
	out:	matrice in output
	n:		numero di righe di out
	m:		numero di colonne di out
*/
__global__ void vector_prod(float* in_a, float* in_b, float* out, int n, int m);

/*
 	Caclola la differenza tra vettore riga e matrice

 	in_matri:	matrice 
 	in_vect:	vettore riga 
	out_matr:	matrice in output
	n:			numero righe di out_matr
	m:			numero colonne di out_matr
 */
__global__ void diff_matr_vect(float* in_matr, float* in_vect, float* out_matr, int n, int m);

// n e m dimensione matrice output
/*
	Calcola il prodotto riga per colonna delle matrici in ingresso

	in_a:	prima matrice
	in_b:	seconda matrice
	out:	matrice risultato
	n:		numero righe di out
	m:		numero di colonne di out
*/

__global__ void matrix_prod(float* in_a, float* in_b, float* out, int n, int m, int p);

/*
	Esegue divisione della matrice per uno scalare

	in_a: 		matrice 
	scalar:		valore scalare
	out:		matrice risultato
	n:			numero righe della matrice
	m:			numero colonne della matrice

 */
__global__ void div_by_scalar(float* in_a, float scalar, float* out, int n, int m);
/*
	Calcola la differenza dei vettori riga in ingresso

	in_a:		vettore uno
	in_b:		vettore due
	out:		vettore di output
	v_size:		dimensione dei vettori
 */
__global__ void diff_vect(float* in_a, float* in_b, float* out, int v_size);

#endif