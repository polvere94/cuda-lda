typedef struct Matrix_ Matrix;
int find_max(float* vector, int size);

float determinant(float **a,float size);
float** cofactor(float** matrix,int size);
float** transpose_cofactor(float** matrix,float** matrix_cofactor,int size);
float* init_matrix(int n_row, int n_col);
void memset_matrix(float* a, int n_row,int n_col);
float determinant(float **a,float size);
void mean(float *M, int n_row, int n_col, float *means);
void sum_vectors(Matrix *M, int n_matrix, int n_col, float *C);
void sum_matrix(float *m1, float *m2, int n_row, int n_col);
void diff_vector(float *vector, float *vector2, int n_col, float *C);
void diff_matrix_vector(float *vector, float *vector2,int n_row, int n_col, float *C);
void prod(float* first, float* second, int row1, int row2, int col1, int col2, float* res);
void transpose(float* a,int n_row, int n_col, float* res);
void print_matrix(float* a, int row, int col);


float** from_linear_to_double(float* a, int n_row, int n_col);
float* from_double_to_linear(float** a, int n_row, int n_col);
float LDeterminant(float* a, int size);
float* LCofactor(float* a, int size);