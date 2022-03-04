#include <stdio.h>
#include <stdlib.h>
#include <complex>

typedef std::complex<float> complex_float;
#define ROW 300
#define COL 100

extern "C" {
void dotprod(complex_float* dotprod_in_matrix, float* dotprod_in_vector, complex_float* dotprod_out) {
	static complex_float a[ROW][COL];
    float b[COL];
    static complex_float c[ROW];

    int const FACTOR = 10;
#pragma HLS array_partition variable=a cyclic factor=FACTOR dim=2
#pragma HLS array_partition variable=b cyclic factor=2 dim=1

    // stream in first matrix
    int t = 0;
    for (int i = 0; i < ROW; i++)
        for (int j = 0; j <COL; j++) {
#pragma HLS PIPELINE II=1
            a[i][j] = dotprod_in_matrix[t];
            t++;
        }

    // stream in the vector
    for (int i = 0; i < COL; i++){
#pragma HLS PIPELINE II=1
        b[i] = dotprod_in_vector[i];
    }

    // dot product of matrix A and vector B
    complex_float sum;
	L1:for (int row = 0; row < ROW; ++row){
    L2:for (int col = 0; col < COL; ++col){
		#pragma HLS PIPELINE II=1
    		sum += a[row][col] * b[col];
    	}
    	c[row] = sum;
    	sum = 0;
    }

    // stream out result matrix

    for (int i = 0; i < ROW; i++)
    {
#pragma HLS PIPELINE II=1
        dotprod_out[i] = c[i];
    }
    return;
}
}
