/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <cstdio>
#include "update.hpp"

using std::vector;

int main(int argc, char** argv) {
    int i,j;
    std::string binaryFile = "/shares/bulk/ldierick/workspace_luc/hls_test_system/Emulation-SW/update.xclbin";
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_vector_dotprod;
    cl::Kernel krnl_vector_update;
    cl::CommandQueue q;


    // Allocate Memory in Host Memory
    vector<complex_float,aligned_allocator<complex_float> > resVect(ROW);
    vector<complex_float,aligned_allocator<complex_float> > kappa(ROW*COL);
    vector<complex_float,aligned_allocator<complex_float> > kappaTimesResSW(COL,0);
    vector<complex_float,aligned_allocator<complex_float> > kappaTimesResHW(COL,0);

    /** Input Initiation */
    int t= 0;
       for(i = 0; i<ROW; i++){
           for(j = 0; j<COL; j++){
               kappa[t] = {i*1.0f,j*i*0.33f};
               t++;
           }
       }


       for(i = 0; i<ROW; i++){
       	   resVect[i] = {i*1.0f,i*0.33f};
       	  }
       /** End of Initiation */

    // Create test data and Software Result
      complex_float conj;
    for(int row = 0; row < ROW; ++row){
            for(int col = 0; col < COL; ++col){
                conj.real(kappa[row][col].real() * resVect[row].real() + kappa[row][col].imag() * resVect[row].imag());
                conj.imag(-kappa[row][col].real() * resVect[row].imag() - kappa[row][col].imag() * resVect[row].real());
                kappaTimesResSW[col] += conj;
            }

        }

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_update = cl::Kernel(program, "update", &err));
            std::cout <<"update kernel exists" <<std::endl;
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
	#include <cstdio>


    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR,sizeof(complex_float)*ROW,
                                          resVect.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR, sizeof(complex_float)*ROW*COL,
                                         kappa.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR , sizeof(complex_float)*COL,
                                            kappaTimesResHW.data(), &err));

    OCL_CHECK(err, err = krnl_vector_update.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_vector_update.setArg(1, buffer_in2));
    OCL_CHECK(err, err = krnl_vector_update.setArg(2, buffer_output));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_update));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0; i < COL; i++) {
        if (kappaTimesResSW[i] != kappaTimesResHW[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << kappaTimesResSW[i]
                      << " Device result = " << kappaTimesResHW2=[i] << std::endl;
            match = false;
            break;
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}


