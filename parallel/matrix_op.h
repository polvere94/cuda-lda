/*
Header file contenente intestazioni di operazioni su matrici
@author: Francesco Polvere
@email: francesco.polvere@studenti.unimi.it
*/

#define CHECK(call) { \
		const cudaError_t error = call; \
		if (error != cudaSuccess) { \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

typedef struct Matrix_ Matrix;

void print_matrix(float* in, int n, int m, char* label);

/*
	Inverte la matrice
*/
void invert_device(float* src_d, float* dst_d, int n);

__global__ void add_vectors(float* in_a, float* in_b, float* out, int v_size, int n_matrix);
__global__ void add_matrix(float* in_a, float* in_b, float* out,int n, int m);
__global__ void transposeNaiveRow(float* in, float* out, int ny, int nx);

/*
	Calcola il vettore media di una matrice rispetto alle colonne
	matrix_mean: in   vettore in input
				 out  vettore di output
				 in_row   numero righe della matrice di input
				 in_col  numero colonne della matrice di input (=colonne matrice output)
*/
__global__ void matrix_mean(float* in, float* out, int in_row, int in_col);

/*Prodotto vettore colonna per vettore riga
 n e m dimensione matrice output*/
__global__ void vector_prod(float* in_a, float* in_b, float* out, int n, int m);

/*
 * kernel: differenza tra vettore e matrice
 */
__global__ void diff_matr_vect(float* in_matr, float* in_vect, float* out_matr, int n, int m);

// n e m dimensione matrice output
__global__ void matrix_prod(float* in_a, float* in_b, float* out, int n, int m,int p);

/*
 * kernel: differenza dei vettori
 */
__global__ void div_by_scalar(float* in_a, float scalar, float* out, int n, int m);
/*
 * kernel: differenza dei vettori
 */
__global__ void diff_vect(float* in_a, float* in_b, float* out, int v_size, int n_matrix);

