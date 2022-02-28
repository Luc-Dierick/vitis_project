#include <stdio.h>
#include <stdlib.h>
#include <complex>
#define ROW 300
#define COL 100


typedef float float_t;
typedef  std::complex<float> complex_float;

extern "C" {
void update(complex_float* update_input, complex_float* update_kappa,
                     complex_float* update_output) {
//

    static complex_float a[ROW];
    static complex_float b[ROW][COL];
    static complex_float c[COL];
//    int /factor = 2;
//#pragma HLS array_partition variable=b cyclic factor=factor dim=2
//#pragma HLS array_partition variable=a cyclic factor=30 dim=1


    // stream in first matrix
    int t = 0;
    for(int i = 0; i < ROW; i++){
    	for(int j = 0; j < COL; j++){
#pragma HLS PIPELINE II=1

    		b[i][j] = update_kappa[t];
    		t++;
    	}
    }

    // stream in the vector
    for (int i = 0; i < ROW; i++) {
#pragma HLS PIPELINE II=1
        a[i] = update_input[i];
    }


    complex_float conj;
    L1:
    for (int row = 0; row < ROW; ++row) {
        L2:
        for (int col = 0; col < COL; ++col) {
#pragma HLS PIPELINE II=1
           conj.real(b[row][col].real() * a[row].real() + b[row][col].imag() * a[row].imag());
            conj.imag(
                    -b[row][col].real() * a[row].imag() - b[row][col].imag() * a[row].real());
            c[col] += conj;
        }
    }


    for (int j = 0; j < COL; j++) {
#pragma HLS PIPELINE II=1
//        c[j] = (j == (COL - 1)) ? 1 : 0;
        update_output[j] = c[j]; //c[j];
//        c[j].data = 0;

    }

}
}
