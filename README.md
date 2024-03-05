# MexWrapper-for-CUDABifurcation-3.0
Collection of MEX C++ API Wrappers of questionable design &amp; utility for [CUDABifurcation 3.0](https://github.com/KiShiVi/CUDABifurcation3.0/tree/newBranch)


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
## Usage
First you need to compile sources with nvcc, this can be done directly from MATLAB using ``mexcuda``
```matlab
mexcuda -output <NameOfFunctionToBeCreated> <NameOfWrapper.cpp> cudaLibrary.cu cudaMacros.cu hostLibrary.cu
```
Example to create function ``bifurcation1DForH``
```matlab
mexcuda -output bifurcation1DForH bifurcation1DForHMEXWrapper.cpp cudaLibrary.cu cudaMacros.cu hostLibrary.cu
```
In MATLAB console you will see
```bash
Building with 'NVIDIA CUDA Compiler'.
MEX completed successfully.
```
Last message indicates that compilation was successfull.

Now u can use the function you just compiled as usuall.

```matlab
[xData, yData] = bifurcation1DForH(...
600,...												% const double tMax,
1000,...											% const int nPts,
3,...												% const int amountOfInitialConditions,
[0.1, 0.1, 0.1],...							        % const double* initialConditions,
[0.001, 0.5],...								    % const double* ranges,
0,...												% const int writableVar,
100,...												% const double maxValue,
1000,...											% const double transientTime,
[0.5, 0.2, 0.2, 5.7],...						    % const double* values,
4,...												% const int amountOfValues,
1);

plot(xData, yData, 'r.', 'MarkerSize', 2);
xlabel('c','interpreter','latex','FontSize', 24);
ylabel('X','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');
```
There are 2 types of wrappers, for 1D analysis & for 2D analysis
```matlab
[xData, yData] = fcn1d(); % 1d usage
[xData, yData, cData] = fcn2d(); % 2 d usage
```
All outputs are ready to plot, for 1D ``xData`` and ``yData`` are ``1 x N `` arrays and can be passed directly to ``plot``
```matlab
plot(xData, y Data)
```
For 2D ``xData`` and ``yData`` are ``1 x N `` arrays and ``cData`` is ``N x N`` matrix they can be passed directly to ``imagesc``
```matlab
imagesc(xData, yData, cData)
```
**FDS still need to be hardcoded in CUDA before compiling**

For documentation on [CUDABifurcation 3.0](https://github.com/KiShiVi/CUDABifurcation3.0/tree/newBranch) you can refer to its dev, one day we will see the __readme.md__ there.
## Notes
You can enable verbose nvcc output if necessary. This will enable MATLAB & NVCC to output logs of compilation process to MATLAB console
```matlab
mexcuda -v -output <NameOfFunctionToBeCreated> <NameOfWrapper.cpp> cudaLibrary.cu cudaMacros.cu hostLibrary.cu
```
You can try compiling CUDA library with nvcc from ``cmd`` and then compile wrappers with ``mex`` linking obj. But why bother if ``mexcuda`` handles everything.
```bash
nvcc -c cudaLibrary.cu -o cudaLibrary.obj -ccbin "D:\Visual Studio 2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
nvcc -c cudaMacros.cu -o cudaMacros.obj -ccbin "D:\Visual Studio 2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
nvcc -c hostLibrary.cu -o hostLibrary.obj -ccbin "D:\Visual Studio 2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
```
``-ccbin`` path should be adjusted according to your system
Than you are going to use ``mex`` and link precompiled objs
```matlab
mex -largeArrayDims -output <NameOfFunctionToBeCreated> <NameOfWrapper.cpp> cudaLibrary.obj cudaMacros.obj hostLibrary.obj -L"<Path to NVIDIA GPU computing Toolkit binaries>" -lcudart
```
## Changes in [CUDABifurcation 3.0](https://github.com/KiShiVi/CUDABifurcation3.0/tree/newBranch)
- hostLibrary ``#DEBUG`` is deprecataed
- All console logs are now handled by ``mexPrintf``
- Some minor ``static_cast`` adustments because of downgraded Toolkit version
