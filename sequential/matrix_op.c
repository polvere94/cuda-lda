#include "matrix_op.h"
#include "lib/svd.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

float* inv_matrix(float* IN, int matsize){
	int i,j,k;
	float **A, **I, *OUT,temp;

	A=(float **)malloc(matsize*sizeof(float *));            
    for(i=0;i<matsize;i++){
        A[i]=(float *)malloc(matsize*sizeof(float));
    }

	I=(float **)malloc(matsize*sizeof(float *));            
    for(i=0;i<matsize;i++)
        I[i]=(float *)malloc(matsize*sizeof(float));

    OUT=(float* )malloc(matsize*matsize*sizeof(float));            
    
    for(i=0;i<matsize;i++){
        for(j=0;j<matsize;j++){
            A[i][j] = IN[i*matsize+j];
        }
    }

	 for(i=0;i<matsize;i++)                                  
        for(j=0;j<matsize;j++)                              
            if(i==j)                                        
                I[i][j]=1;                                  
            else                                           
                I[i][j]=0;                                 
    
    for(k=0;k<matsize;k++) {
    	temp=A[k][k];                  
        for(j=0;j<matsize;j++){
            A[k][j]/=temp;                                  
            I[k][j]/=temp;                                  
        }                                                   
        for(i=0;i<matsize;i++){
            temp=A[i][k];                       
            for(j=0;j<matsize;j++){                                   
                if(i==k)
                    break;                      
                A[i][j]-=A[k][j]*temp;         
                I[i][j]-=I[k][j]*temp;         
            }
        }
    }

    for(i=0;i<matsize;i++){
        for(j=0;j<matsize;j++){
            OUT[i*matsize+j] = I[i][j];
        }
     }
    return OUT;
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
 
void mean(float *M, int n_row, int n_col, float *means){
	float sum;
	for(int j = 0; j < n_col; j++){
		sum = 0;
		for(int i = 0; i < n_row; i++){
			sum = sum + M[(i*n_col)+j];
		}
		means[j]=sum/n_row;
	}
}

void sum_vectors_mean(Matrix *M, int n_matrix, int n_col, float *C){
	memset(C,0,n_col*sizeof(float));
	for(int i = 0; i<n_matrix; i++){
		for(int j=0; j<n_col; j++){
			C[j]= C[j] + M[i].mean[j];
		}
	}
}

void sum_matrix(float *m1, float *m2, int n_row, int n_col){
	for(int i = 0; i<n_row; i++){
		for(int j=0; j<n_col; j++){
			m1[(i*n_col)+j] = m1[(i*n_col)+j] + m2[(i*n_col)+j];
		}
	}
}

void diff_vector(float *vector, float *vector2, int n_col, float *C){
	for(int j=0; j<n_col; j++){
		C[j] = vector[j] - vector2[j];
	}
}

void diff_matrix_vector(float *matrix, float *vector2,int n_row, int n_col, float *C){
	for(int i=0; i<n_row; i++){
		for(int j=0; j<n_col; j++){
			C[(i*n_col)+j] = matrix[(i*n_col)+j] - vector2[j];
		}
	}
}

void prod(float* first, float* second, int row1, int row2, int col1, int col2, float* res){
	int c,d,k= 0;
	float sum = 0;
	for (c = 0; c < row1; c++) {
    	for (d = 0; d < col2; d++) {
	        for (k = 0; k < row2; k++) {
	          sum = sum + first[c*col1+k]*second[k*col2+d];
	        }
 			res[c*col2+d] = sum;
	        sum = 0;
      }
    }
}

void transpose(float* a,int n_row, int n_col, float* res){
	int i,j = 0;
    for(i=0; i<n_row; i++)
        for(j=0; j<n_col; j++) {
        	res[(j*n_row)+i] = a[(i*n_col)+j];
        }
}

int find_max(float* vector, int size){
	int max = 0,i;
	int index = 0;
	for(i=0; i<size; i++){
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

void Lsvd(float* a, int size, float* w, float* v){

	float** temp_v = from_linear_to_double(v,size,size);
	
	float **temp_invsw_by_sb = from_linear_to_double(a,size,size);
	dsvd(temp_invsw_by_sb, size, size, w, temp_v);
	
	free(v);
	v = from_double_to_linear(temp_v,size,size);
}

void print_matrix(float* a, int n_row, int n_col){
	printf("\n\n");
	for(int i = 0; i<n_row; i++){
		printf("|");
		for(int j = 0; j<n_col; j++){
			printf("%0.3f |", a[(i*n_col)+j]);
		}
		printf("\n");
	}
	printf("\n\n");
}