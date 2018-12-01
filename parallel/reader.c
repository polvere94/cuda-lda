const char* getfield(char* line, int num){
	const char* token;
	int i=0;
	for (token = strtok(line, ","); 
		token && *token; 
		token = strtok(line, ",")) {
		if (i==num)
			return token;
		i++;
	}
	return NULL;
}

void read_data_file(Matrix* matrix, int n_matrix, int n_lines,int n_feature){
	char line[1024], *token, *filenames[4];
	int h, lines_count, i=0;
	FILE *file;
	
	// 1 ha problemi
	filenames[0] = "../commons/data/DATA_4_MT.dat";
	filenames[1] = "../commons/data/DATA_2_MT.dat";
	filenames[2] = "../commons/data/DATA_3_MT.dat";
	filenames[3] = "../commons/data/DATA_4_MT.dat";

	for(i=0; i<n_matrix; i++){		
		file = fopen(filenames[i], "r");
		lines_count=0;
	    while (fgets(line, 1024, file) && n_lines > lines_count){
	    	token = NULL;
			token = strtok(line, ",");
			h=0;
			while(token != NULL){
				if(h==n_feature)break;
				matrix[i].data[(lines_count*n_feature)+h] = atof(token);
			    token = strtok(NULL, ",");
			    h++;
			}
			lines_count++;
		}
	}
}