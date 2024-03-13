#include "mex.h"
#include "..\Library\hostLibrary.cuh"

#include <stdio.h>
#include <iomanip>
#include <ctime>
#include <conio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#define OUT_FILE_PATH "D:\\CUDABifurcation3.0\\MatlabScripts\\CUDA_OUT\\mat.csv"

// Helper function to split string by delimiter
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Main MEX function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Validate number of inputs and outputs
    if (nrhs != 14) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumInputs", "14 inputs required.");
    }
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumOutputs", "Two outputs required.");
    }

    // Extract inputs from MATLAB to C++ types
    double tMax = mxGetScalar(prhs[0]);
    double NT = mxGetScalar(prhs[1]);
    int nPts = static_cast<int>(mxGetScalar(prhs[2]));
    double h = mxGetScalar(prhs[3]);
    double eps = mxGetScalar(prhs[4]);

    double* initialConditions = mxGetPr(prhs[5]);
    int amountOfInitialConditions = static_cast<int>(mxGetScalar(prhs[6]));

    double* ranges = mxGetPr(prhs[7]);

    double* dblPtr = mxGetPr(prhs[8]);
    size_t numElements = mxGetNumberOfElements(prhs[8]);
    int* indicesOfMutVars = new int[numElements];
    for (size_t i = 0; i < numElements; ++i) {
        indicesOfMutVars[i] = static_cast<int>(dblPtr[i]);
    }

    int writableVar = static_cast<int>(mxGetScalar(prhs[9]));
    double maxValue = mxGetScalar(prhs[10]);
    double transientTime = mxGetScalar(prhs[11]);
    double* values = mxGetPr(prhs[12]);
    int amountOfValues = static_cast<int>(mxGetScalar(prhs[13]));

    mexPrintf("Inputs validation & parsing completed... \n");
    mexPrintf("Starting CUDA... \n");

    // Call your function
    LS1D(
        tMax, //const double 
        NT, //const double 
        nPts, //const int 
        h, //const double 
        eps, //const double 
        initialConditions, //const double* 
        amountOfInitialConditions, //const int 
        ranges, //const double* 
        indicesOfMutVars, //const int*
        writableVar, //const int 
        maxValue, //const double 
        transientTime, //const double 
        values, //const double* 
        amountOfValues); //const int 

    mexPrintf("Computations done... \n");
    mexPrintf("Building matlab-covinient output \n");
    // Reading CSV file produced by bifurcation1DForH
    std::ifstream file(OUT_FILE_PATH); // Assuming the output file name is output.csv
    std::string line;
    std::vector<double> out1, out2;

    while (std::getline(file, line)) {
        std::vector<std::string> tokens = split(line, ',');
        if (tokens.size() == 2) { // Ensuring there are exactly two columns
            out1.push_back(std::stod(tokens[0]));
            out2.push_back(std::stod(tokens[1]));
        }
    }

    // Creating MATLAB arrays for output
    plhs[0] = mxCreateDoubleMatrix(out1.size(), 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(out2.size(), 1, mxREAL);
    double* out1Ptr = mxGetPr(plhs[0]);
    double* out2Ptr = mxGetPr(plhs[1]);

    // Copy data into MATLAB arrays
    std::copy(out1.begin(), out1.end(), out1Ptr);
    std::copy(out2.begin(), out2.end(), out2Ptr);

    delete[] indicesOfMutVars;
    mexPrintf("All done!!!");
}