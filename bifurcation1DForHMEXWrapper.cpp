#include "mex.h"
#include "hostLibrary.cuh"

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
    if (nrhs != 11) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumInputs", "Eleven inputs required.");
    }
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumOutputs", "Two outputs required.");
    }

    // Extract inputs from MATLAB to C++ types
    double tMax = mxGetScalar(prhs[0]);
    int nPts = static_cast<int>(mxGetScalar(prhs[1]));
    int amountOfInitialConditions = static_cast<int>(mxGetScalar(prhs[2]));
    double* initialConditions = mxGetPr(prhs[3]);
    double* ranges = mxGetPr(prhs[4]);
    int writableVar = static_cast<int>(mxGetScalar(prhs[5]));
    double maxValue = mxGetScalar(prhs[6]);
    double transientTime = mxGetScalar(prhs[7]);
    double* values = mxGetPr(prhs[8]);
    int amountOfValues = static_cast<int>(mxGetScalar(prhs[9]));
    int preScaller = static_cast<int>(mxGetScalar(prhs[10]));

    mexPrintf("Inputs validation & parsing completed... \n");
    mexPrintf("Starting CUDA... \n");
    // Call your function
    bifurcation1DForH(tMax, nPts, amountOfInitialConditions, initialConditions, ranges, writableVar, maxValue, transientTime, values, amountOfValues, preScaller);

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
    mexPrintf("All done!!!");
}