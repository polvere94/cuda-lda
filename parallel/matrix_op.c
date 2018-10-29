#include "matrix_op.h"
typedef struct Matrix_{
    float* data;
    float* mean;
} Matrix;

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


float* init_matrix(int n_row, int n_col){
	float *res;
	res = (float*)malloc(n_row*n_col*sizeof(float));
	if (res == NULL) { 
		perror("Errore malloc: ");
		exit(-1);
	}
	memset(res,0,n_row*n_col*sizeof(float));
	return res;
}

void memset_matrix(float* a, int n_row,int n_col){
	memset(a, 0, n_col*n_row*sizeof(float));
}

void free_matrix(float* a){
	free(a);
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
/*
Calcola la media sulle colonne di una matrice
*/
void mean(float *M, int n_row, int n_col, float *means){
	float sum;
	for(int j = 0; j < n_col; j++){
		sum = 0;
		for(int i = 0; i < n_row; i++){
			sum = sum + M[(i*n_col)+j];//M[i][j];
		}
		means[j]=sum/n_row;
	}
}
/*
	Somma componenti delle matrici tra di loro [...]
*/
void sum_vectors(Matrix *M, int n_matrix, int n_col, float *C){
	memset(C,0,n_col*sizeof(float));
	for(int i = 0; i<n_matrix; i++){
		for(int j=0; j<n_col; j++){
			/*C[0][j] = C[0][j]*/ C[j]= C[j] + M[i].mean[j];
		}
	}
}

//TODO da problemi, forse buffer overflow
void sum_matrix(float *m1, float *m2, int n_row, int n_col){
	for(int i = 0; i<n_row; i++){
		for(int j=0; j<n_col; j++){
			//m1[i][j] =  m1[i][j] + m2[i][j];
			m1[(i*n_col)+j] = m1[(i*n_col)+j] + m2[(i*n_col)+j];
		}
	}
}

void diff_vector(float *vector, float *vector2, int n_col, float *C){
	//memset(C[0],0,n_col*sizeof(float));
	for(int j=0; j<n_col; j++){
		C[j] = vector[j] - vector2[j];
	}
}

//Differenza tra matrice e vettore di uguali colonne
void diff_matrix_vector(float *vector, float *vector2,int n_row, int n_col, float *C){
	//memset(C,0,n_col*sizeof(float));
	for(int i=0; i<n_row; i++){
		for(int j=0; j<n_col; j++){
			C[(i*n_col)+j] = vector[(i*n_col)+j] - vector2[j];
			//C[i][j] = vector[i][j] - vector2[0][j];
		}
	}
}

void prod(float* first, float* second, int row1, int row2, int col1, int col2, float* res){
	int c,d,k= 0;
	float sum = 0;
	for (c = 0; c<row1; c++) {
    	for (d = 0; d<col2; d++) {
	        for (k = 0; k<row2; k++) {
	          sum = sum + first[(c*col1)+k]*second[(k*col2)+d];
	        }
 			res[(c*col2)+d] = sum;
	        sum = 0;
      }
    }
}

void transpose(float* a,int n_row, int n_col, float* res){
	int i,j = 0;
    for(i=0; i<n_row; i++)
        for(j=0; j<n_col; j++) {
        	res[(j*n_row)+i] = a[(i*n_col)+j];
        	//res[j][i] = a[i][j];
        }
}

void print_matrix(float* a, int row, int col){
	printf("\n\n");
	for(int i = 0; i<row; i++){
		printf("|");
		for(int j = 0; j<col; j++){
			printf("%0.3f |", a[(i*col)+j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

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

float** from_linear_to_double(float* a, int n_row, int n_col){
	int i,j;
	float** res;
	res = (float**)malloc(n_row*sizeof(float*));
	for(i=0; i<n_row; i++)
		res[i]=(float*)malloc(n_col*sizeof(float*));
	
	for(i=0; i<n_row; i++)
		for(j=0; j<n_col; j++)
			res[i][j] = a[(i*n_col)+j];
	return res;
}

float* from_double_to_linear(float** a, int n_row, int n_col){
	int i,j;
	float* res;
	res = (float*)malloc(n_row*n_col*sizeof(float));
	for(i=0; i<n_row; i++)
		for(j=0; j<n_col; j++)
			res[(i*n_col)+j] = a[i][j];
	return res;
}


float LDeterminant(float* a, int size){
	int i;
	float d;
	float **temp;
	temp = from_linear_to_double(a,size,size);
	d = determinant(temp, size);
	for(i=0;i<size;i++)
		free(temp[i]);
	free(temp);
	return d;
}

float* LCofactor(float* a, int size){
	int i;
	float** temp;
	temp = from_linear_to_double(a,size,size);
	float** cof =  cofactor(temp,size);
	float* res = from_double_to_linear(cof,size,size);
	for(i=0;i<size;i++){
		free(cof[i]);
		free(temp[i]);
	}
	free(temp);
	free(cof);
	return res;
}

void Lsvd(float* a, int size, float* w, float* v){

	float** temp_v = from_linear_to_double(v,size,size);
	

	float **temp_invsw_by_sb = from_linear_to_double(a,size,size);
	dsvd(temp_invsw_by_sb, size, size, w, temp_v);
	
	free(v);
	v = from_double_to_linear(temp_v,size,size);
}


