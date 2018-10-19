#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "lib/svd.c"

#define FEATURE_NUMBER 3
#define N_MATRIX 3
#define DATASET_NAME1 "dataset1.txt"
#define DATASET_NAME2 "dataset2.txt"
#define DATASET_NAME3 "dataset3.txt"

typedef struct Matrix_{
    float** data;
    float** mean;
} Matrix;

int find_max(float* vector, int size){
	int max = 0,i;
	int index = 0;
	for(i=0;i<size;i++){
		if(vector[i]>=max){
			index=i;
			max=vector[i];
		}
	}
	return index;
}
void plot(Matrix* matrix, int n_matrix, int n_row, int n_col, int k,float point_x, float point_y);
float determinant(float **a,float size);
float** cofactor(float** matrix,int size);
float** transpose_cofactor(float** matrix,float** matrix_cofactor,int size);
float** init_matrix(int n_row,int n_col);
void free_matrix(float** matrix, int n_row);
void memset_matrix(float** matrix,int n_row,int n_col);
void diff_matrix_vector(float **vector, float **vector2,int n_row, int n_col, float **C);
void sum_vectors(Matrix *M, int n_matrix, int n_col, float **C);
void sum_matrix(float **m1, float **m2, int n_row, int n_col);
void prod(float** first, float** second, int row1, int row2, int col1, int col2, float** res);
void transpose(float** matrix,int row, int col, float** res);
void print_matrix(float** matrix, int row, int col);
void diff_vector(float **vector, float **vector2, int n_col, float **C);
void mean(float **M, int n_row, int n_col, float **means);

int main(void){
	int unit = 1;
	int n_row = 50;
	int n_col = 4;
	Matrix* M;
	float** tmean_vect;
	
	FILE *file;
	float aa,bb,cc,dd = 0;

	const char *filenames[3];
	filenames[0] = "data/dataset1.txt";
	filenames[1] = "data/dataset2.txt";
	filenames[2] = "data/dataset3.txt";
	// Alloca memoria per ogni matrice
	M = (Matrix*) malloc(N_MATRIX*sizeof(Matrix));
	for(int i = 0;i<N_MATRIX;i++){
		M[i].data = init_matrix(n_row, n_col);
		M[i].mean = init_matrix(unit, n_col);
	}
	
	tmean_vect= init_matrix(unit, n_col);

	for(int i = 0; i<N_MATRIX; i++){
		 // Lettura da file del dataset
	    file = fopen(filenames[i], "r");
	    int h=0;
		while (fscanf(file, "%f %f %f %f", &aa, &bb, &cc, &dd) != EOF) {
	  		M[i].data[h][0]=aa;
	  		M[i].data[h][1]=bb;
	  		M[i].data[h][2]=cc;
	  		M[i].data[h][3]=dd;
	  		if(h==n_row-1)
	  			break;
	  		h++;
		}
/*

		M[i].data[0][0]=1+i;
		M[i].data[0][1]=2;
		M[i].data[1][0]=3;
		M[i].data[1][1]=4+i;
		M[i].data[2][0]=1;
		M[i].data[2][1]=1;
*/
		print_matrix(M[i].data,n_row, n_col);

		//Calcola la media della matrice i-esima
		mean(M[i].data, n_row, n_col, M[i].mean);

		printf("Media matrice %d",i);
		print_matrix(M[i].mean, unit, n_col);
	}

	printf("Somma vettori matrici");
	sum_vectors(M, N_MATRIX, n_col, tmean_vect);
	
	//Calcolo media globale
	for(int y=0; y<n_col; y++)
			tmean_vect[0][y] = tmean_vect[0][y] / N_MATRIX;

	print_matrix(tmean_vect,1,n_col);


	//Calcolo di Sb (Between class Matrix)
	// SBs = Ns .* (m_s - m)*(m_s - m)';
	/*

				Calcolo Between-class scatter matrix
	
	*/
	
	float **accumulatore_sb = init_matrix(n_col, n_col);
	float **SBc = init_matrix(unit, n_col);
	float **v_transpose = init_matrix(n_col, unit);
	float **res2 = init_matrix(n_col, n_col);

	for(int i=0; i<N_MATRIX; i++){
		diff_vector(M[i].mean, tmean_vect, n_col, SBc);
		print_matrix(SBc, 1, n_col);
		
		//Righe pari al numero di colonne matrice A  e colonne pari al numero di colonne matrice B
		memset_matrix(v_transpose,n_col,n_col);

		transpose(SBc,1, n_col, v_transpose);

		printf("Prodotto tra:");
		print_matrix(v_transpose, n_col, 1);
		print_matrix(SBc, 1, n_col);

		prod(v_transpose, SBc, n_col, unit, unit ,n_col, res2);
		sum_matrix(accumulatore_sb, res2, n_col, n_col);
	}
	/*free_matrix(res2, n_col);
	free_matrix(v_transpose, n_col);
	free_matrix(SBc, unit);*/


	print_matrix(accumulatore_sb, n_col, n_col);



	// Calcolo di Sw (Whithin scatter Matrix)
	//sw = n.*( c_s + c_vi + c_ve); Somma di covarianze
	/*
	
					Calcolo Whithin-class scatter matrix
	
	*/
	float **ACC = init_matrix(n_col, n_col);
	memset_matrix(ACC, n_col, n_col);
	float **res = init_matrix(n_col, n_col);
	float **ONE = init_matrix(n_row, n_col);
	float **TWO = init_matrix(n_col, n_row);
	int y,x = 0;
	for(int h =0; h<N_MATRIX; h++){
		printf("Differenza tra\n");
		print_matrix(M[h].data, n_row, n_col);
		printf("e\n");
		print_matrix(M[h].mean, 1, n_col);

		//Differenza tra la la matrice ed il vettore, riga per riga
		diff_matrix_vector(M[h].data, M[h].mean, n_row, n_col, ONE);

		//Fai il prodotto della trasposta della riga per la riga stessa
		transpose(ONE, n_row, n_col, TWO);
		prod(TWO, ONE, n_col, n_row, n_row, n_col, res); //Risultato (n_col X n_col)
		
		for(y=0; y<n_col; y++){
			for(x=0; x<n_col; x++){
				res[x][y] = res[x][y] / (n_row-1);
			}
		}

		printf("Matrice covarianza %d", h);
		print_matrix(res, n_col, n_col);
		sum_matrix(ACC, res, n_col, n_col);
		printf("FINE\n");
	}
	print_matrix(ACC, n_col,n_col);
	/*for(int i =0; i<N_MATRIX; i++){
		free(M[i].data);
		free(M[i].mean);
	}*/



	int size_sw = n_col;
	int size_sb = n_col;
	float** sw = ACC;
	float** bw = accumulatore_sb;
	printf("\n\n\n");
	printf("SW:\n");
	print_matrix(sw, size_sw, size_sw);
	printf("SB:\n");
	print_matrix(bw, size_sb, size_sb);
	float d = determinant(sw, size_sw);
	printf("Determinante delal matrice = %f",d);
	if (d==0){
	   	printf("\nInverse of Entered Matrix is not possible\n");
		return 0;
	}
	printf("\n Matrice inversa sw\n");
	float** inv_sw = cofactor(sw,size_sw);
	print_matrix(inv_sw,size_sw,size_sw);

	printf("Prodotto inversa(sw) e sb\n");
	float **inv_by_sb = init_matrix(size_sw,size_sw);
	prod(inv_sw, bw, size_sw, size_sb, size_sw, size_sb, inv_by_sb);

	print_matrix(inv_by_sb, size_sw, size_sw);

	float* w =(float*) malloc(size_sw*sizeof(float));
	float** v = init_matrix(size_sw,size_sw);
	//Trovo autovettori e autovalori
	dsvd(inv_by_sb, size_sw, size_sw, w,v);

	printf("Autovettore");
	print_matrix(v,size_sw,size_sw);
	printf("Autovalore\n");
	printf("|");
	for(int i = 0; i < size_sw; i++)
		printf("%f | ",w[i]);


	


	
	int i_max = find_max(w, size_sw);
	printf("Autovettore con autovalore massimo\n");
	for(int i = 0;i< size_sw;i++ ){
		printf("%f ",v[i][i_max]);
	}

	 plot(M, N_MATRIX, n_row, n_col,2,v[0][i_max],v[1][i_max]);

	  for(int i =0; i<N_MATRIX; i++){
		free(M[i].data);
		free(M[i].mean);
	}

	return 0;
}

/* Metodo della matrice dei cofattori
Calcolo dei cofattori della matrice */
float** cofactor(float** matrix,int size){
	float **matrix_cofactor = (float**)malloc(size*sizeof(float *));
	for (int j = 0; j < size; j++)
			matrix_cofactor[j] = (float *) calloc(size, sizeof(float));

	float **m_cofactor = (float**)malloc(size*sizeof(float *));
	for (int j = 0; j < size; j++)
		m_cofactor[j] = (float *) calloc(size, sizeof(float));
     
    int p,q,m,n,i,j;
    for (q=0;q<size;q++){
        for (p=0;p<size;p++){
            m=0;
            n=0;
            for (i=0;i<size;i++){
                for (j=0;j<size;j++){
                    if (i != q && j != p){
    	                m_cofactor[m][n]=matrix[i][j];
                        if (n<(size-2)){
        	                n++;
                        }else{
                            n=0;
                            m++;
                        }
                     }
                 }
             }
             matrix_cofactor[q][p]=pow(-1,q + p) * determinant(m_cofactor,size-1);
         }
     }
     return transpose_cofactor(matrix,matrix_cofactor,size);
}

/*Finding transpose of cofactor of matrix*/ 
float** transpose_cofactor(float** matrix,float** matrix_cofactor,int size){
	int i,j;
    float d;
	float **m_inverse = (float**)malloc(size*sizeof(float *));
	for (int j = 0; j < size; j++)
		m_inverse[j] = (float *) calloc(size, sizeof(float));
   
 
 	 d=determinant(matrix,size);

    for (i=0;i<size;i++){
        for (j=0;j<size;j++){
            m_inverse[i][j]=matrix_cofactor[j][i]/d;
        }
    }
   	return m_inverse;
}

float** init_matrix(int n_row,int n_col){
	int j = 0;
	float **res;
	res = (float**)malloc(n_row*sizeof(float *));
	if (res == NULL) { 
		perror("Errore malloc: ");
		exit(-1);
	}
	for (j = 0; j < n_row; j++){
		res[j] =(float*) malloc(n_col* sizeof(float));
		if (res[j] == NULL) { 
			perror("Errore malloc interna: ");
			exit(-1);
		}
	}	
	return res;
}

void memset_matrix(float** matrix,int n_row,int n_col){
	int i;
	for (i=0; i<n_row; i++)
		memset(matrix[i], 0, n_col*sizeof(float));
}

void free_matrix(float** matrix, int n_row){
	int i;
	for (i=0; i<n_row; i++){
		free(matrix[i]);
	}
	free(matrix);
}


 
/*Per il calcolo del determinante della matrice */
float determinant(float **a,float size){
	float **b = (float**)malloc(size*sizeof(float *));
	for (int j = 0; j < size; j++)
		b[j] = (float *) calloc(size, sizeof(float));

	float s=1,det=0;
	int i,j,m,n,c;
	if (size==1){
	    return (a[0][0]);
	}else{
	    det=0;
	    for (c=0; c<size; c++){
	    	m=0;
	        n=0;
	        for (i=0;i<size;i++){
	            for (j=0; j<size; j++){
	                b[i][j]=0;
	                if (i != 0 && j != c){
	                   	b[m][n]=a[i][j];
	                   	if (n<(size-2))
	                       n++;
	                   	else{
	                    	n=0;
	                    m++;
	                    }
	                }
	            }
	        }
	        det=det + s * (a[0][c] * determinant(b,size-1));
	        s=-1 * s;
        }
    }
    return (det);
}
/**
Calcola la media sulle colonne di una matrice
*/
void mean(float **M, int n_row, int n_col, float **means){
	float sum;
	for(int j = 0; j < n_col; j++){
		sum = 0;
		for(int i = 0; i < n_row; i++){
			sum = sum + M[i][j];
		}
		means[0][j]=sum/n_row;
	}
}
/**
	Somma componenti delle matrici tra di loro [...]
*/
void sum_vectors(Matrix *M, int n_matrix, int n_col, float **C){
	memset(C[0],0,n_col*sizeof(float));
	for(int i = 0; i<n_matrix; i++){
		for(int j=0; j<n_col; j++){
			C[0][j] = C[0][j] + M[i].mean[0][j];
		}
	}
}

//TODO da problemi, forse buffer overflow
void sum_matrix(float **m1, float **m2, int n_row, int n_col){
	for(int i = 0; i<n_row; i++){
		for(int j=0; j<n_col; j++){
			m1[i][j] =  m1[i][j] + m2[i][j];
		}
	}
}

void diff_vector(float **vector, float **vector2, int n_col, float **C){
	memset(C[0],0,n_col*sizeof(float));
	for(int j=0; j<n_col; j++){
		C[0][j] = vector[0][j] - vector2[0][j];
	}
}

//Differenza tra matrice e vettore di uguali colonne
void diff_matrix_vector(float **vector, float **vector2,int n_row, int n_col, float **C){
	memset(C[0],0,n_col*sizeof(float));
	for(int i=0; i<n_row; i++){
		for(int j=0; j<n_col; j++){
			C[i][j] = vector[i][j] - vector2[0][j];
		}
	}
}

void prod(float** first, float** second, int row1, int row2, int col1, int col2, float** res){
	int c,d,k= 0;
	float sum = 0;
	for (c = 0; c < row1; c++) {
    	for (d = 0; d < col2; d++) {
	        for (k = 0; k < row2; k++) {
	          sum = sum + first[c][k]*second[k][d];
	        }
 			res[c][d] = sum;
	        sum = 0;
      }
    }
}

void transpose(float** matrix,int row, int col, float** res){
	int i,j = 0;
    for(i=0; i<row; i++)
        for(j=0; j<col; j++) {
        	res[j][i] = matrix[i][j];
        }
}

void print_matrix(float** matrix, int row, int col){
	printf("\n\n");
	for(int i = 0; i<row; i++){
		printf("|");
		for(int j = 0; j<col; j++){
			printf("%0.3f |", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

void plot(Matrix* matrix, int n_matrix, int n_row, int n_col, int k,float point_x, float point_y){
	int i,h, j = 0;

	#define NUM_COMMANDS 2
	char * commandsForGnuplot[] = {"set title 'LDA - Fisher Iris'", 
	"plot 'data/data.temp' u 1:2:3  title 'dati'  pointtype 5 linecolor  palette, 'data/line.temp' u 1:2  title 'proiezione' with line"};
	
	FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");

	FILE * temp = fopen("data/data.temp", "w");

	for(h=0; h<N_MATRIX; h++){
		for(i=0; i < n_row;i++){
			for(j=0;j<2;j++){
				fprintf(temp, "%lf ", matrix[h].data[i][j]); 
			}
			fprintf(temp, "%d \n",h);
		}
	}

	FILE * temp_line = fopen("data/line.temp", "w");

	for(int i=-10;i<25;i++){

		fprintf(temp_line, "%lf ", point_x*i);
		fprintf(temp_line, "%lf ", point_y*i);
		fprintf(temp_line,"\n");
	}
	 
			

	for (i=0; i < NUM_COMMANDS; i++){
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]);
	}

}