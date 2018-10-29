#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "lib/svd.c"
#include "matrix_op.c"

#define FEATURE_NUMBER 3
#define N_MATRIX 3
#define DATASET_NAME1 "dataset1.txt"
#define DATASET_NAME2 "dataset2.txt"
#define DATASET_NAME3 "dataset3.txt"




void read_data_file(Matrix* matrix, int n_matrix, int n_lines){
	char* filenames[3];
	filenames[0] = "data/dataset1.txt";
	filenames[1] = "data/dataset2.txt";
	filenames[2] = "data/dataset3.txt";
	FILE *file;
	float a,b,c,d=0;
	int h;
	int i=0;
	int col = 4; //fissato
	for(i=0; i<n_matrix; i++){		
		h = 0;
		file = fopen(filenames[i], "r");
		while (fscanf(file, "%f %f %f %f", &a, &b, &c, &d) != EOF) {
			matrix[i].data[(h*col)+0]=a; 
			matrix[i].data[(h*col)+1]=b;
			matrix[i].data[(h*col)+2]=c;
			matrix[i].data[(h*col)+3]=d;
			if(h==n_lines-1)
				break;
			h++;
		}
	}
}

float* between_scatter_matrix(Matrix* matrix, int n_matrix, int n_row, int n_col, float* mean){
	int i;
	float *accumulatore_sb = init_matrix(n_col, n_col);
	float *SBc = init_matrix(1, n_col);
	float *v_transpose = init_matrix(n_col, 1);
	float *res2 = init_matrix(n_col, n_col);

	for(i=0; i<n_matrix; i++){
		diff_vector(matrix[i].mean, mean, n_col, SBc);
		print_matrix(SBc, 1, n_col);
		
		//Righe pari al numero di colonne matrice A  e colonne pari al numero di colonne matrice B
		memset_matrix(v_transpose,n_col,n_col);

		transpose(SBc,1, n_col, v_transpose);

		printf("Prodotto tra:");
		print_matrix(v_transpose, n_col, 1);
		print_matrix(SBc, 1, n_col);
		
		prod(v_transpose, SBc, n_col, 1, 1 ,n_col, res2);
		sum_matrix(accumulatore_sb, res2, n_col, n_col);
	}
	print_matrix(accumulatore_sb, n_col, n_col);
	free(SBc);
	free(v_transpose);
	free(res2);
	return accumulatore_sb;
}

float* within_scatter_matrix(Matrix* matrix, int n_matrix, int n_row, int n_col){
	float *ACC = init_matrix(n_col, n_col);
	memset_matrix(ACC, n_col, n_col);
	float *res = init_matrix(n_col, n_col);
	float *ONE = init_matrix(n_row, n_col);
	float *TWO = init_matrix(n_col, n_row);
	int y,x = 0;
	for(int h =0; h<n_matrix; h++){
		printf("Differenza tra\n");
		print_matrix(matrix[h].data, n_row, n_col);
		printf("e\n");
		print_matrix(matrix[h].mean, 1, n_col);

		//Differenza tra la la matrice ed il vettore, riga per riga
		diff_matrix_vector(matrix[h].data, matrix[h].mean, n_row, n_col, ONE);

		//Fai il prodotto della trasposta della riga per la riga stessa
		transpose(ONE, n_row, n_col, TWO);
		prod(TWO, ONE, n_col, n_row, n_row, n_col, res); //Risultato (n_col X n_col)
		
		for(y=0; y<n_col; y++){
			for(x=0; x<n_col; x++){
				res[(y*n_col)+x] = res[(y*n_col)+x] / (n_row-1);
			}
		}

		printf("Matrice covarianza %d", h);
		print_matrix(res, n_col, n_col);
		sum_matrix(ACC, res, n_col, n_col);
		printf("FINE\n");
	}
	print_matrix(ACC, n_col,n_col);
	free(res);
	free(ONE);
	free(TWO);
	return ACC;
}



int main(void){
	int unit = 1;
	int n_row = 50;
	int n_col = 4;
	Matrix* M;
	float* tmean_vect;
	
	tmean_vect = init_matrix(unit, n_col);

	// Alloca memoria per ogni matrice
	M = (Matrix*) malloc(N_MATRIX*sizeof(Matrix));
	for(int i=0; i<N_MATRIX; i++){
		M[i].data= (float*)malloc(n_row*n_col*sizeof(float));
		M[i].mean= (float*)malloc(1*n_col*sizeof(float));
	}

	read_data_file(M, N_MATRIX, n_row);

	for(int i=0;i<N_MATRIX;i++){
		//Calcola la media della matrice i-esima
		mean(M[i].data, n_row, n_col, M[i].mean);
	}
	//print_matrix(M[0].data,n_row,n_col);

	printf("Somma vettori matrici");
	sum_vectors(M, N_MATRIX, n_col, tmean_vect);
	
	//Calcolo media globale
	for(int y=0; y<n_col; y++)
		tmean_vect[y] = tmean_vect[y] / N_MATRIX;

	print_matrix(tmean_vect,1,n_col);
	

	/****************************************************
				Calcolo Between-class scatter matrix
	*****************************************************/
	float* bw = between_scatter_matrix(M,N_MATRIX,n_row,n_col,tmean_vect);

	/****************************************************
				Calcolo Within-class scatter matrix
	
	*****************************************************/
	float* sw = within_scatter_matrix(M,N_MATRIX,n_row,n_col);


	int size_sw = n_col;
	int size_sb = n_col;
	printf("\n\n\n");
	printf("SW:\n");
	print_matrix(sw, size_sw, size_sw);
	printf("SB:\n");
	print_matrix(bw, size_sb, size_sb);

	//float **temp_sw = from_linear_to_double(sw,size_sw,size_sw);
	float d = LDeterminant(sw, size_sw);//determinant(temp_sw, size_sw);
	printf("Determinante delal matrice = %f",d);
	if (d==0){
	   	printf("\nInverse of Entered Matrix is not possible\n");
		return 0;
	}
	printf("\n Matrice inversa sw:\n");
	float* inv_sw = LCofactor(sw, size_sw);

	printf("Prodotto tra inversa(sw) e sb:\n");
	float* invsw_by_sb = init_matrix(size_sw,size_sw);
	prod(inv_sw, bw, size_sw, size_sb, size_sw, size_sb, invsw_by_sb);

	print_matrix(invsw_by_sb, size_sw, size_sw);


	/*************************************************************
				Calcolo degli autovalori e autovettori
	**************************************************************/
	float* w =(float*)malloc(size_sw*sizeof(float));
	float* v =(float*)malloc(size_sw*size_sw*sizeof(float));

	Lsvd(invsw_by_sb, size_sw,w,v);

	//FINE
	printf("Autovettori");
	print_matrix(v,size_sw,size_sw);
	printf("Autovalori\n");
	printf("|");
	for(int i = 0; i < size_sw; i++)
		printf("%f | ",w[i]);


	int i_max = find_max(w, size_sw);
	printf("Autovettore con autovalore massimo\n");
	for(int i = 0;i< size_sw;i++ ){
		printf("%f ",v[i*size_sw+i_max]);
	}

	

	for(int i =0; i<N_MATRIX; i++){
		free(M[i].data);
		free(M[i].mean);
	}


	/*************************************************************
				Trasformazione dello spazio di partenza
	**************************************************************/
	 float* h_eigenvectors = (float*)malloc(m*(n_matrix-1)*sizeof(float));

    // 4x4
   	int rows = size_sw;
   	int cols = size_sw;
    for(int i=0; i<rows; i++){
    	int h=0;
    	for(int j=0; j<(n_matrix-1); j++){
    		h_eigenvectors[i*(n_matrix-1)+h] = v[i*cols+j];
    		h++;
    	}    	
    }


	/*************************************************************
				Plot dei risultati
	**************************************************************/
	//plot(M, N_MATRIX, n_row, n_col,2,v[0][i_max],v[1][i_max]);
	return 0;
}


/*void plot(Matrix* matrix, int n_matrix, int n_row, int n_col, int k,float point_x, float point_y){
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

}*/