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

// Helper to create linspace
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> linspaced;

    if (num == 0) {
        return linspaced;
    }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // Ensure that end is exactly the last element

    return linspaced;
}

// Main MEX function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Validate number of inputs and outputs
    if (nrhs != 14) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumInputs", "14 inputs required.");
    }
    if (nlhs != 3) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumOutputs", "Three outputs required.");
    }

    // Extract inputs from MATLAB to C++ types
    double tMax = mxGetScalar(prhs[0]);
    int nPts = static_cast<int>(mxGetScalar(prhs[1]));
    double h = mxGetScalar(prhs[2]);
    int amountOfInitialConditions = static_cast<int>(mxGetScalar(prhs[3]));
    double* initialConditions = mxGetPr(prhs[4]);
    double* ranges = mxGetPr(prhs[5]);

    double* dblPtr = mxGetPr(prhs[6]);
    size_t numElements = mxGetNumberOfElements(prhs[6]);
    int* indicesOfMutVars = new int[numElements];
    for (size_t i = 0; i < numElements; ++i) {
        indicesOfMutVars[i] = static_cast<int>(dblPtr[i]);
    }

    int writableVar = static_cast<int>(mxGetScalar(prhs[7]));
    double maxValue = mxGetScalar(prhs[8]);
    double transientTime = mxGetScalar(prhs[9]);
    double* values = mxGetPr(prhs[10]);
    int amountOfValues = static_cast<int>(mxGetScalar(prhs[11]));
    int preScaller = static_cast<int>(mxGetScalar(prhs[12]));
    double eps = mxGetScalar(prhs[13]);

    mexPrintf("Inputs validation & parsing completed... \n");
    mexPrintf("Starting CUDA... \n");
    // Call your function
    bifurcation2D(
        tMax,								//  const double	
        nPts,								//  const int		
        h,									//  const double	
        amountOfInitialConditions,			//  const int		
        initialConditions,					//  const double* 
        ranges,								//  double* 
        indicesOfMutVars,					//  int* 
        writableVar,						//  const int		
        maxValue,							//  const double	
        transientTime,						//  const double	
        values,								//  const double* 
        amountOfValues,						//  const int		
        preScaller,							//  const int		
        eps);						        //  const double	

    mexPrintf("Computations done... \n");
    mexPrintf("Building matlab-covinient output \n");
    // Reading CSV file produced by bifurcation1DForH
    std::ifstream file(OUT_FILE_PATH);
    std::string line;

    // Read the first two lines for x and y ranges
    double xRange[2], yRange[2];
    for (int i = 0; i < 2; ++i) {
        std::getline(file, line);
        std::vector<std::string> tokens = split(line, ' ');
        if (i == 0) {
            xRange[0] = std::stod(tokens[0]);
            xRange[1] = std::stod(tokens[1]);
        }
        else if (i == 1) {
            yRange[0] = std::stod(tokens[0]);
            yRange[1] = std::stod(tokens[1]);
        }
    }

    // Creating x and y linspace arrays
    std::vector<double> x = linspace(xRange[0], xRange[1], nPts);
    std::vector<double> y = linspace(yRange[0], yRange[1], nPts);

    // Process the rest of the data, ignoring the last column
    std::vector<std::vector<double>> dataMatrix;
    while (std::getline(file, line)) {
        std::vector<std::string> tokens = split(line, ',');
        std::vector<double> rowData;
        for (size_t i = 0; i < tokens.size() - 1; ++i) { // Ignore last column
            rowData.push_back(std::stod(tokens[i]));
        }
        dataMatrix.push_back(rowData);
    }

    // Convert dataMatrix into a MATLAB matrix (2D array)
    mxArray* matlabDataMatrix = mxCreateDoubleMatrix(dataMatrix.size(), dataMatrix[0].size(), mxREAL);
    double* matDataPtr = mxGetPr(matlabDataMatrix);
    for (size_t i = 0; i < dataMatrix.size(); ++i) {
        for (size_t j = 0; j < dataMatrix[i].size(); ++j) {
            matDataPtr[j * dataMatrix.size() + i] = dataMatrix[i][j]; // Column-major order filling
        }
    }

    // Creating MATLAB arrays for output
    plhs[0] = mxCreateDoubleMatrix(x.size(), 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(y.size(), 1, mxREAL);
    double* out1Ptr = mxGetPr(plhs[0]);
    double* out2Ptr = mxGetPr(plhs[1]);
    // Preparing the output
    // Copy data into MATLAB arrays
    std::copy(x.begin(), x.end(), out1Ptr);
    std::copy(y.begin(), y.end(), out2Ptr);
    plhs[2] = matlabDataMatrix;

    delete[] indicesOfMutVars;

    mexPrintf("All done!!!");
}