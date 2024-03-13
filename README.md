# MATLAB interface for CUDABifurcation3.0
Provides a CUDAHandler class which is a MATLAB interface for [CUDABifurcation 3.0](https://github.com/KiShiVi/CUDABifurcation3.0/tree/newBranch)

## Requirements
- Matlab R2022b
- [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?)
- MSVS 2019

NVCC compiler comes with CUDA Toolkit, but its best to doublecheck. Run ``cmd``
```bash
nvcc --version
```
It should produce the following
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:15:10_Pacific_Standard_Time_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
```
## Project structure
The project follows structure described below.

``Compiled`` folder is used to store compiled MEX API files for any system.

``Matlab`` folder is there the class is located along with some other supporting stuff.

``Library`` folder containd CUDA Library itself.

``MEXWrappers`` folder is there all .cpp raw MEX API wrappers are located. 

```bash
Bif3.0Interface
|
|--Library
|  |--cudaLibrary.cu
|  |--cudaMacros.cu 
|  |--hostLibrary.cu
|  |--cudaLibrary.cuh
|  |--cudaMacros.cuh
|  |--hostLibrary.cuh
|
|--MEXWrappers
|  |--Wrapper1.cpp
|  |--Wrapper2.cpp
|  |--WrapperX.cpp
|  |--Wrapper11.cpp
|
|--Matlab
|  |--class to handle wrappers
|  |--some other functions
|  |--helper files
|
|--Compiled
|  |--SystemName1
|  |  |--<SystemName1>_<WeapperName1>.mexw64
|  |  |--<SystemName1>_<WeapperNameX>.mexw64
|  |  |--<SystemName1>_<WeapperName11>.mexw64
|  |
|  |--SystemNameN
|     |--<SystemNameN>_<WeapperName1>.mexw64
|     |--<SystemNameN>_<WeapperNameX>.mexw64
|     |--<SystemNameN>_<WeapperName11>.mexw64
```
## Usage
**FDS still need to be hardcoded in CUDA before using CUDAHandler**

Initialize CUDAHandler with desired system ``SystemName``
```matlab
duffCUDA = CUDAHandler("SystemName");
```

``CUDAHandler`` constructor will check if precompiled MEXs exist in ``Compiled/SystemName``folder, if so it will use them and you can proceed. 
If ``Compiled/SystemName``folder doesn't exist, it will be created and compilation of all MEXs will start, simple log is available in MATLAB console. 

Now with ``CUDAHandler`` object created, you can proceed to supported types of analysis:
* 1D bifurcation (for any system parameter/initial conditions/integration step)
```matlab
[xData, yData] = <class obj>.bifurcation1D(args);
[xData, yData] = <class obj>.bifurcation1DIC(args);
[xData, yData] = <class obj>.bifurcation1DForH(args);
```
* 2D bifurcation (for any pair of system parameters/initial conditions)
```matlab
[xData, yData, cData] = <class obj>.bifurcation2D(args);
[xData, yData, cData] = <class obj>.bifurcation2DIC(args);
```
* 1D Lyapunov exponents (for any system parameter/initial conditions)
```matlab
[xData, yData] = <class obj>.LLE1D(args);
[xData, yData] = <class obj>.LLE1DIC(args);
```
* 2D Lyapunov exponents (for any pair of system parameters/initial conditions)
```matlab
[xData, yData, cData] = <class obj>.LLE2D(args);
[xData, yData, cData] = <class obj>.LLE2DIC(args);
```
* 1D LS metrics (for any system parameter)
```matlab
[xData, yData] = <class obj>.LS1D(args);
```
* 2D LS metrics (for any pair of system parameters)
```matlab
[xData, yData, cData] = <class obj>.LS2D(args);
```
``args`` are semi-different for different types of analisys, though the mutual ones are present too. For this reason ``CUDAHandler`` supports global arguments, which can be used by all functions.
You can easily access them by running
```matlab
<class obj>.gOpts();
```
This will print to console all global args along with descriptions and expected data types, this can be directly copy-pasted to your source files.
```matlab
<class obj>.tMax =                 % Double scalar > 0                         % Simultaion time
<class obj>.NT =                   % Double scalar > 0                         % Normaliztion time for LLE
<class obj>.nPts =                 % Double scalar > 0                         % Number of point to analyze system in
<class obj>.h =                    % Double scalar > 0                         % Integration step
<class obj>.LLE_eps =              % Double scalar > 0                         % Epsilon value for LLE
<class obj>.intCon =               % Double vector 1 x N                       % Initial conditions
<class obj>.ranges =               % Double vector 1 x 2 for 1D & 2 x 2 for 2D % Ranges to vary mutVariables
<class obj>.indicesOfMutVars =     % Double vector 1 x N >=0                   % Indices of mutVariabes
<class obj>.writableVar            % Double scalar >= 0                        % Indice of state variable to conduct analysys on
<class obj>.maxValue =             % Double scalar >= 0                        % Value to determine if system has diverged
<class obj>.transient =            % Double scalar >= 0                        % Time to simulate transient
<class obj>.values =               % Double vector 1 x N                       % System parameters
<class obj>.preScaller =           % Double scalar >= 0                        % Modifier to reduce computations (every <preScaller> point is computed)
<class obj>.DBSCAN_eps =           % Double scalar >= 0                        % Epsilon value for DBSCAN
<class obj>.LS_eps  =              % Double scalar >= 0                        % Epsilon value for LS
```
You can, obviously, provide ``args`` direclty to functions. Since ``CUDAHandler`` provides MATALB with clear function identities autocompletes, are a way to go!

![Autocomplete](https://i.imgur.com/m7QYtgJ.png)

More details can be found in demo file ``demo.mlx``

## Documentation
Documentation is available directly in matlab
```matlab
help CUDAHandler
help CUDAHandler.<fuction name>
```
And you obviously can click hyperlink to access MATLAb buil-in documentation engine.

![Doc](https://i.imgur.com/82a61aW.png)

## About demo
``demo.mlx`` is a demonstration of working with ``CUDAHandler``. You can find syntax breakdowns along with some other usefull tips there.

Demo uses precompiled MEXs for Duffing oscilator. The FDS is Heun's method
```cpp
// DUFFING Heun
// x[0] = t
// x[1] = dx/dt
// x[2] = dy/dt
// a[0] - damp
// a[1] - alpha
// a[2] - beta
// a[3] - amp
// a[4] - w

// Temporary variables for the predictor step
double tempX, tempY;

// Predictor step using Euler's method
tempX = x[1] + h * x[2];
tempY = x[2] + h * (-a[0] * x[2] - a[1] * x[1] - a[2] * pow(x[1], 3) + a[3] * sin(a[4] * x[0]));

// Corrector step (average of slopes at the start and end)
x[1] = x[1] + h * 0.5 * (x[2] + tempY);
x[2] = x[2] + h * 0.5 * (-a[0] * x[2] - a[1] * x[1] - a[2] * pow(x[1], 3) + a[3] * sin(a[4] * x[0])
                        - a[0] * tempY - a[1] * tempX - a[2] * pow(tempX, 3) + a[3] * sin(a[4] * (x[0] + h)));

x[0] = x[0] + h; // Update time
```
## Changes in [CUDABifurcation 3.0](https://github.com/KiShiVi/CUDABifurcation3.0/tree/newBranch) and MEX Wrappers
- Minor paths adjustments to be in line with restructured project 
