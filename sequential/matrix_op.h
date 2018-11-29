#ifndef MATRIX_OP_H
#define MATRIX_OP_H


/*
	Struttura di una matrice
	contiene un vettore dei dati (data)
	ed un vettore media contenente la media delle colonne (mean)
*/
typedef struct Matrix_{
    float* data;
    float* mean;
} Matrix;

/*
	Effettua una scansione del vettore ed ottiene
	l'indice del valore massimo nel vettore d'ingresso

	vector:	vettore in ingresso
	size:	dimesione
	return:	indice del valore massimo presente nel vettore
*/
int find_max(float* vector, int size);

/*
	Esegue il calcolo della matrice inversa della 
	matrice in ingresso

	IN:			matrice in ingresso
	matsize:	dimensione della matrice (quadrata)
*/
float* inv_matrix(float* IN, int matsize);

/*
	Inizializza una matrice di n_row e n_col

	n_row:	numero di righe
	n_col:	numero di colonne
*/
float* init_matrix(int n_row, int n_col);

/*
	Riempie la matrice di ingresso di dimensione n_row e n_col
	con degli 0

	a:		matrice di ingresso
	n_row:	numero di righe della matrice
	n_col:	numero di colonne dela matrice
*/
void memset_matrix(float* a, int n_row,int n_col);

/*
	Calcola la media della matrice in ingresso M di dimensione 
	n_row e n_col e restituisce il vettore di medie

	M:		matrice di ingresso
	n_row:	numero di righe della matrice
	n_col: 	numero di colonne della matrice
*/
void mean(float *M, int n_row, int n_col, float *means);

/*
	Somma i vettori media delle matrici tra di loro e restituisce
	un vettore somma dell medie

	M:			matrici in ingresso 
	n_matrix: 	numero delle matrici
	n_col:		numero di colonne (lunghezza del vettore M.mean)
	C:			vettore risultato
*/
void sum_vectors_mean(Matrix *M, int n_matrix, int n_col, float *C);

/*
	Esegue la somma di due matrici coincidenti

	m1:		matrice uno
	m2:		matrice due
	n_row: 	numero righe
	n_col:	numero colonne
*/
void sum_matrix(float *m1, float *m2, int n_row, int n_col);

/*
	Calcola la differenza tra due vettori

	vector: 	vettore uno
	vector2:	vettore due
	n_col:		numero di colonne
	C:			vettore risultato
*/
void diff_vector(float *vector, float *vector2, int n_col, float *C);

/*
	Esegue la differenza tra le righe della matrice ed un vettore 
	e restituisce una matrice 
	
	matrix:		matrice in ingresso
	vector2: 	vettore in ingresso
	n_row:		numero di righe della matrice
	n_col:		numero colonne della matrice = numero colonne del vettore
	C:			matrice risultato
*/
void diff_matrix_vector(float *matrix, float *vector2,int n_row, int n_col, float *C);

/*
	Calcola il prodotto riga per colonna di due matrici

	first:		matrice uno
	second:		matrice due
	row1:		numero righe matrice uno
	col1:		numero colonne matrice uno
	row2:		numero righe matrice due
	col2:		numero colonne matrice due
	res:		matrice risultato

*/
void prod(float* first, float* second, int row1, int row2, int col1, int col2, float* res);

/*
	Effettua la trasposta della matrice in ingresso

	a:		matrice in ingresso
	n_row:	numero di righe
	n_col:	numero di colonne
	res:	matrice risultante
*/
void transpose(float* a,int n_row, int n_col, float* res);

/*
	Stampa formattata su standard output la matrice in ingresso

	a:		matrice in ingresso
	n_row:	numero di righe
	n_col:	numero di colonne
*/
void print_matrix(float* a, int n_row, int n_col);

/*
	Prende la matrice in input rappresentata tramite 
	singolo puntatore e restituisce
	la sua versione rappresentanta come doppio puntatore

	a:		matrice in ingresso
	n_row	numero di righe
	n_col:	numero di colonne

	return:	matrice tramite doppio puntatorie
*/
float** from_linear_to_double(float* a, int n_row, int n_col);

/*
	Prende la matrice in input rappresentata tramite 
	doppio puntatore e restituisce la sua versione 
	rappresentanta puntatore singolo

	a:		matrice in ingresso
	n_row	numero di righe
	n_col:	numero di colonne

	return:	matrice tramite puntatore singolo

*/
float* from_double_to_linear(float** a, int n_row, int n_col);

/*
	Cacola la Singular Value Decomposition (SVD) della 
	matrice quadrata in ingresso

	a:		matrice di ingresso
	size:	dimensione della matrice (n_row oppure n_col)
*/
void Lsvd(float* a, int size, float* w, float* v);

#endif