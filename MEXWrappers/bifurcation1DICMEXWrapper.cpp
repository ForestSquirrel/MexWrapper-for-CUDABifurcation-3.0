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
    if (nrhs != 13) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumInputs", "13 inputs required.");
    }
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumOutputs", "Two outputs required.");
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

    mexPrintf("Inputs validation & parsing completed... \n");
    mexPrintf("Starting CUDA... \n");
    // Call your function
    bifurcation1DIC(
        tMax,							// Время моделирования системы  const double	
        nPts,							// Разрешение диаграммы  const int		
        h,								// Шаг интегрирования const double	
        amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе ) const int		
        initialConditions,				// Массив с начальными условиями const double* 
        ranges,							// Диаппазон изменения переменной const double* 
        indicesOfMutVars,				// Индекс изменяемой переменной в массиве values const int* 
        writableVar,					// Индекс уравнения, по которому будем строить диаграмму  const int		
        maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся" const double	
        transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы const double	
        values,							// Параметры  const double* 
        amountOfValues,					// Количество параметров const int		
        preScaller);					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка) const int		

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