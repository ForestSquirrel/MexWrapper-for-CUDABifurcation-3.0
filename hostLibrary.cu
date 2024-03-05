// --- ������������ ���� ---
#include "hostLibrary.cuh"
#include "mex.h"

// --- ���� ��� ���������� �������������� ������ ---
#define OUT_FILE_PATH "D:\\CUDABifurcation3.0\\MatlabScripts\\CUDA_OUT\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// --- ���������, ���������� ������� ������� � ������� ���������� ��������� ---


__host__ void distributedSystemSimulation(
	const double	tMax,							// ����� ������������� �������
	const double	h,								// ��� ��������������
	const double	hSpecial,						// ��� �������� ����� ��������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues)					// ���������� ����������	
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h;

	int amountOfThreads = hSpecial / h;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.8;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)	

	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_data = new double[amountOfPointsInBlock * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_data,				amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );
	
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof(double),			cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------


	mexPrintf("Distributed System Simulation\n");


		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil( ( 1024.0f * 8.0f ) / ( ( amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (amountOfThreads + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		distributedCalculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(
				amountOfPointsForSkip,
				amountOfThreads,
				h,
				hSpecial,
				d_initialConditions,
				amountOfInitialConditions,
				d_values,
				amountOfValues,
				tMax / hSpecial,
				writableVar,
				d_data
				);

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_data, d_data, amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(20);

		for (size_t j = 0; j < amountOfPointsInBlock; ++j)
			if (outFileStream.is_open())
			{
				outFileStream << h * j << ", " << h_data[j] << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}


		gpuErrorCheck(cudaFree(d_data));
		gpuErrorCheck(cudaFree(d_initialConditions));
		gpuErrorCheck(cudaFree(d_values));

		delete[] h_data;
}


// ----------------------------------------------------------------------------
// --- ����������� �������, ��� ������� ���������� �������������� ��������� ---
// ----------------------------------------------------------------------------

__host__ void bifurcation1D(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// ��������� ��������� ����������
	const int*		indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------
	
	double* h_outPeaks		= new double	[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int*	h_amountOfPeaks = new int		[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int*	d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	int*	d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_data,				nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_ranges,				2 * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_indicesOfMutVars,	1 * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );

	gpuErrorCheck( cudaMalloc( ( void** )&d_outPeaks,			nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_amountOfPeaks,		nPtsLimiter * sizeof( int ) ) );
	
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof(double),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof(int),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof(double),			cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = ( size_t )ceil( ( double )nPts / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------


	mexPrintf("Bifurcation 1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);


	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, ( amountOfInitialConditions + amountOfValues ) * sizeof(double) * blockSize >> > 
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter );
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> > 
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_outPeaks,					// �������� ������, ���� ����� �������� �������� �����
				nullptr,					// ���������� �������� ����� �� �����
				0);							// ��� �������������� �� �����

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks,		d_outPeaks,			nPtsLimiter * amountOfPointsInBlock * sizeof(double),	cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks,	d_amountOfPeaks,	nPtsLimiter * sizeof(int),								cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
				}
				else
				{
					mexPrintf("\nOutput file open error\n");
					exit(1);
				}

		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck( cudaFree( d_data ) );
	gpuErrorCheck( cudaFree( d_ranges ) );
	gpuErrorCheck( cudaFree( d_indicesOfMutVars ) );
	gpuErrorCheck( cudaFree( d_initialConditions ) );
	gpuErrorCheck( cudaFree( d_values ) );
				   			 
	gpuErrorCheck( cudaFree( d_outPeaks ) );
	gpuErrorCheck( cudaFree( d_amountOfPeaks ) );

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



/**
 * �������, ��� ������� ���������� �������������� ��������� �� ����.
 */
__host__ void bifurcation1DForH(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// �������� ��������� ����
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� ����� � ����� ����� ---
	int amountOfPointsInBlock = tMax / (ranges[0] < ranges[1] ? ranges[0] : ranges[1]) / preScaller;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	int* d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)&d_data, nPtsLimiter* amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)&d_outPeaks, nPtsLimiter* amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("Bifurcation 1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA_H << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				transientTime,				// ����� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				tMax,						// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA_H << <gridSize, blockSize >> >
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_outPeaks,					// �������� ������, ���� ����� �������� �������� �����
				nullptr,					// ���������� �������� ����� �� �����
				0);							// ��� �������������� �� �����

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
				}
				else
				{
					mexPrintf("\nOutput file open error\n");
					exit(1);
				}

		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



// -----------------------------------------------------------------------------------------
// --- �������, ��� ������� ���������� �������������� ���������. (�� ��������� ��������) ---
// -----------------------------------------------------------------------------------------

__host__ void bifurcation1DIC(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// ��������� ��������� ����������
	const int*		indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller)						// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ���������� (���� � �� ���������� ��� ������ �������) ---
	// ---------------------------------------------------------------------------------------------------
	
	// ���������: ����� �� ����� ���� ������, ��� (amountOfPointsInBlock / 2), �.�. ����� ���� �� ����� ����� ���� ���
	double* h_outPeaks		= new double	[ceil(nPtsLimiter * amountOfPointsInBlock * sizeof(double) / 2.0f)];
	int*	h_amountOfPeaks = new int		[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int*	d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	double* d_outPeaks;				// ��������� �� ������ � GPU � ��������������� ������ ���. ���������
	int*	d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_data,				nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_ranges,				2 * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_indicesOfMutVars,	1 * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );

	gpuErrorCheck( cudaMalloc( ( void** )&d_outPeaks,			nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_amountOfPeaks,		nPtsLimiter * sizeof( int ) ) );
	
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof(double),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof(int),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof(double),			cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = ( size_t )ceil( ( double )nPts / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("Bifurcation 1DIC\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512: blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, ( amountOfInitialConditions + amountOfValues ) * sizeof(double) * blockSize >> > 
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				1,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter );
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> > 
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_outPeaks,					// �������� ������, ���� ����� �������� �������� �����
				nullptr,					// ���������� �������� ����� �� �����
				0);							// ��� �������������� �� �����

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks,		d_outPeaks,			nPtsLimiter * amountOfPointsInBlock * sizeof(double),	cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks,	d_amountOfPeaks,	nPtsLimiter * sizeof(int),								cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
				}
				else
				{
					mexPrintf("\nOutput file open error\n");
					exit(1);
				}

		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------
	gpuErrorCheck( cudaFree( d_data ) );
	gpuErrorCheck( cudaFree( d_ranges ) );
	gpuErrorCheck( cudaFree( d_indicesOfMutVars ) );
	gpuErrorCheck( cudaFree( d_initialConditions ) );
	gpuErrorCheck( cudaFree( d_values ) );
				   			 
	gpuErrorCheck( cudaFree( d_outPeaks ) );
	gpuErrorCheck( cudaFree( d_amountOfPeaks ) );

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}



// ------------------------------------------------------------------------
// --- �������, ��� ������� ��������� �������������� ��������� (DBSCAN) ---
// ------------------------------------------------------------------------

__host__ void bifurcation2D(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > ( nPts * nPts ) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int*	d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int*	d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int*	d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	double* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( (void** )&d_data,				nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_ranges,				4 * sizeof(double)));
	gpuErrorCheck( cudaMalloc( (void** )&d_indicesOfMutVars,	2 * sizeof(int)));
	gpuErrorCheck( cudaMalloc( (void** )&d_initialConditions,	amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck( cudaMalloc( (void** )&d_values,				amountOfValues * sizeof(double)));
					 		  		   
	gpuErrorCheck( cudaMalloc( (void** )&d_amountOfPeaks,		nPtsLimiter * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_intervals,			nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_dbscanResult,		nPtsLimiter * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_helpfulArray,		nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				4 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	2 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice) );

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = ( size_t )ceil( ( double )( nPts * nPts ) / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------


	/*mexPrintf("Bifurcation 2D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);*/


	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for ( int i = 0; i < amountOfIteration; ++i )
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if ( i == amountOfIteration - 1 )
			nPtsLimiter = ( nPts * nPts ) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		blockSize = blockSize > 512 ? 512 : blockSize;			// �� ��������� ����������� � 512 ������ � �����
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				h * preScaller);							// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> > 
			(	d_data, 
				amountOfPointsInBlock, 
				nPtsLimiter,
				d_amountOfPeaks, 
				d_intervals, 
				d_helpfulArray, 
				eps, 
				d_dbscanResult);

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
				mexPrintf("\nOutput file open error\n");
				exit(1);
			}


		//mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));

	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

	// ---------------------------
}



// ------------------------------------------------------------------------------
// --- �������, ��� ������� ��������� �������������� ��������� (DBSCAN) �� IC ---
// ------------------------------------------------------------------------------

__host__ void bifurcation2DIC(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps)								// ������� ��� ��������� DBSCAN 
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;											// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;											// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)		

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > ( nPts * nPts ) ? (nPts * nPts) : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )



	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_data;					// ��������� �� ������ � ������ GPU ��� �������� ���������� �������
	double* d_ranges;				// ��������� �� ������ � ���������� ��������� ����������
	int*	d_indicesOfMutVars;		// ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	// ��������� �� ������ � ���������� ���������
	double* d_values;				// ��������� �� ������ � �����������

	int*	d_amountOfPeaks;		// ��������� �� ������ � GPU � ���-��� ����� � ������ �������.
	double* d_intervals;			// ��������� �� ������ � GPU � ����������� ����������� �����
	int*	d_dbscanResult;			// ��������� �� ������ � GPU �������������� ������� (���������) � GPU
	double* d_helpfulArray;			// ��������� �� ������ � GPU �� ��������������� ������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( (void** )&d_data,				nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_ranges,				4 * sizeof(double)));
	gpuErrorCheck( cudaMalloc( (void** )&d_indicesOfMutVars,	2 * sizeof(int)));
	gpuErrorCheck( cudaMalloc( (void** )&d_initialConditions,	amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck( cudaMalloc( (void** )&d_values,				amountOfValues * sizeof(double)));
					 		  		   
	gpuErrorCheck( cudaMalloc( (void** )&d_amountOfPeaks,		nPtsLimiter * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_intervals,			nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_dbscanResult,		nPtsLimiter * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( (void** )&d_helpfulArray,		nPtsLimiter * amountOfPointsInBlock * sizeof( double ) ) );

	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				4 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	2 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice) );

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = ( size_t )ceil( ( double )( nPts * nPts ) / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("Bifurcation 2DIC\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	int stringCounter = 0; // ��������������� ���������� ��� ���������� ������ ������� � ����

	// --- ������� � ����� ������ ����� ����������� �������� ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for ( int i = 0; i < amountOfIteration; ++i )
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if ( i == amountOfIteration - 1 )
			nPtsLimiter = ( nPts * nPts ) - (nPtsLimiter * i);

		int blockSize;			// ���������� ��� �������� ������� �����
		int minGridSize;		// ���������� ��� �������� ������������ ������� �����
		int gridSize;			// ���������� ��� �������� �����

		// --- �������, ��� ���� ���� �� ����� ������������ ������ ��� 48�� ������ ---
		// --- ������ ������ � ����� ��������� (amountOfInitialConditions + amountOfValues) * sizeof(double) ���� ---
		// --- ���������� ������, ����� ������������ ���������� ������� � ����� �� ����� ���������� ---
		// --- ��������, ��� � ����� �� ����� ���� ������ 1024 ������� ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )

		// --------------------------------------------------
		// --- CUDA ������� ��� ������� ���������� ������ ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(	nPts,						// ����� ���������� ��������� - nPts
				nPtsLimiter,				// ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
				amountOfPointsInBlock,		// ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// ���������� ��� ����������� ����� ������
				amountOfPointsForSkip,		// ���������� ����� ��� �������� ( transientTime )
				2,							// ����������� ( ��������� ���������� )
				d_ranges,					// ������ � �����������
				h,							// ��� ��������������
				d_indicesOfMutVars,			// ������� ���������� ����������
				d_initialConditions,		// ��������� �������
				amountOfInitialConditions,	// ���������� ��������� �������
				d_values,					// ���������
				amountOfValues,				// ���������� ����������
				amountOfPointsInBlock,		// ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
				preScaller,					// ���������, ������� ��������� ����� � ����� ��������
				writableVar,				// ������ ���������, �� �������� ����� ������� ���������
				maxValue,					// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
				d_data,						// ������, ��� ����� �������� ���������� ������
				d_amountOfPeaks);			// ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������

		// --------------------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ���������� ����� ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// ������ � ������������ ������
				amountOfPointsInBlock,		// ���������� ����� � ����� ����������
				nPtsLimiter,				// ���������� ������, ������������� � ������� ��������
				d_amountOfPeaks,			// �������� ������, ���� ����� �������� ���������� ����� ��� ������ �������
				d_data,						// �������� ������, ���� ����� �������� �������� �����
				d_intervals,				// ���������� ��������
				h * preScaller);							// ��� ��������������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- ���������� ���������� ������� CUDA, ��� ���������� ����������� �������� ����� � ����� ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA ������� ��� ��������� DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> > 
			(	d_data, 					// ������ (����)
				amountOfPointsInBlock, 		// ���������� ����� � ����� �������
				nPtsLimiter,				// ���������� ������ (������) � data
				d_amountOfPeaks, 			// ������, ���������� ���������� ����� ��� ������� ����� � data
				d_intervals, 				// ���������� ���������
				d_helpfulArray, 			// ��������������� ������ 
				eps, 						// �������
				d_dbscanResult);			// �������������� ������

		// -----------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---
		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
				mexPrintf("\nOutput file open error\n");
				exit(1);
			}
		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

	// ---------------------------
}



__host__ void LLE1D(
	const double	tMax,								// ����� ������������� �������
	const double	NT,									// ����� ������������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const double	eps,								// ������� ��� LLE
	const double*	initialConditions,					// ������ � ���������� ���������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues)						// ���������� ����������
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� �� ����� ������������ NT ---
	size_t amountOfNT_points = NT / h;

	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / NT;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;																// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;																// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )

	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_ranges;				   // ��������� �� ������ � ���������� ��������� ����������
	int*	d_indicesOfMutVars;		   // ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	   // ��������� �� ������ � ���������� ���������
	double* d_values;				   // ��������� �� ������ � �����������

	double* d_lleResult;			   // ������ ��� �������� ��������� ����������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_ranges,				2 * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_indicesOfMutVars,	1 * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );
					 		  	 
	gpuErrorCheck( cudaMalloc( ( void** )&d_lleResult,			nPtsLimiter * sizeof(double)));
			
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice ) );

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("LLE 1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// ���������� ��� �������� ������� �����
		int minGridSize;	// ���������� ��� �������� ������������ ������� �����
		int gridSize;		// ���������� ��� �������� �����

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( 3 * amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );

		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )


		// ------------------------------------
		// --- CUDA ������� ��� ������� LLE ---
		// ------------------------------------

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > 
			(	nPts,								// ����� ����������
				nPtsLimiter, 						// ���������� � ������� �������
				NT, 								// ����� ������������
				tMax, 								// ����� �������������
				amountOfPointsInBlock,				// ���������� �����, ���������� ����� �������� � "data"
				i * originalNPtsLimiter, 			// ���������� ��� ����������� �����
				amountOfPointsForSkip,				// ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
				1, 									// �����������
				d_ranges, 							// ������, ���������� ��������� ������������� ���������
				h, 									// ��� ��������������
				eps, 								// �������
				d_indicesOfMutVars, 				// ������� ���������� ����������
				d_initialConditions,				// ��������� �������
				amountOfInitialConditions, 			// ���������� ��������� �������
				d_values, 							// ���������
				amountOfValues, 					// ���������� ����������
				tMax / NT, 							// ���������� �������� (����������� �� tMax)
				1, 									// ��������� ��� ��������� ��������
				writableVar,						// ������ ���������� � x[] �� �������� ������ ���������
				maxValue, 							// ������������� �������� ���������� ��� �������������
				d_lleResult);						// �������������� ������

		// ------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}
		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



__host__ void LLE1DIC(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues)
{
	// --- ���������� �����, ������� ����� ������������� ����� �������� �� ����� ������������ NT ---
	size_t amountOfNT_points = NT / h;

	// --- ���������� �����, ������� ����� ������������� ����� �������� � ����� ������� ���������� ---
	int amountOfPointsInBlock = tMax / NT;

	// --- ���������� �����, ������� ����� ��������� ��� ������������� ������� ---
	// --- (amountOfPointsForSkip ������ ��������������� ����� �� ����� ����������� � ��������) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// ���������� ��� �������� ���������� ������ ������ � GPU
	size_t totalMemory;																// ���������� ��� �������� ������ ������ ������ � GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// �������� ��������� � ����� ������ ������ GPU

	freeMemory *= 0.5;																// ������������ ������ (����� �������� ���� ����� ��������� GPU ������)

	// --- ������ ���������� ������, ������� �� ������ ��������������� ����������� � ���� ������ ������� ---
	// TODO ������� ������ ��������� ������
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// ���� �� ����� ��������� ������ ������, ��� ���������, �� ������ ������������ �� �������� (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// ���������� �������� �������� nPts ��� ���������� �������� ( getValueByIdx )

	// ----------------------------------------------------------
	// --- �������� ������ ��� �������� ��������� ����������  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- ��������� �� ������� ������ � GPU ---
	// -----------------------------------------

	double* d_ranges;				   // ��������� �� ������ � ���������� ��������� ����������
	int*	d_indicesOfMutVars;		   // ��������� �� ������ � �������� ���������� ���������� � ������� values
	double* d_initialConditions;	   // ��������� �� ������ � ���������� ���������
	double* d_values;				   // ��������� �� ������ � �����������

	double* d_lleResult;			   // ������ ��� �������� ��������� ����������

	// -----------------------------------------

	// -----------------------------
	// --- �������� ������ � GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_ranges,				2 * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_indicesOfMutVars,	1 * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );
					 		  	 
	gpuErrorCheck( cudaMalloc( ( void** )&d_lleResult,			nPtsLimiter * sizeof(double)));
			
	// -----------------------------

	// ---------------------------------------------------------
	// --- �������� ��������� ������� ��������� � ������ GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice ) );

	// ---------------------------------------------------------

	// --- ������ ���������� �������� ��� ��������� �������������� ��������� ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- �������� ��������� ���������� ����� ��� ������ ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------
	mexPrintf("LLE 1DIC\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- �������� ����, ������� ��������� amountOfIteration �������� ��� ������� �������� nPtsLimiter ������ ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- ���� �� �� ��������� ��������, ��������� ����������������� nPtsLimiter � ������� ��� ������ ---
		// --- ����������� �������������� ����� ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// ���������� ��� �������� ������� �����
		int minGridSize;	// ���������� ��� �������� ������������ ������� �����
		int gridSize;		// ���������� ��� �������� �����

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( 3 * amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );

		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// �� ��������� ����������� � 1024 ������ � �����

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// ������ ������� ����� ( ������� �������� �������� ceil() )


		// ------------------------------------
		// --- CUDA ������� ��� ������� LLE ---
		// ------------------------------------

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > 
			(	nPts,								// ����� ����������
				nPtsLimiter, 						// ���������� � ������� �������
				NT, 								// ����� ������������
				tMax, 								// ����� �������������
				amountOfPointsInBlock,				// ���������� �����, ���������� ����� �������� � "data"
				i * originalNPtsLimiter, 			// ���������� ��� ����������� �����
				amountOfPointsForSkip,				// ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
				1, 									// �����������
				d_ranges, 							// ������, ���������� ��������� ������������� ���������
				h, 									// ��� ��������������
				eps, 								// �������
				d_indicesOfMutVars, 				// ������� ���������� ����������
				d_initialConditions,				// ��������� �������
				amountOfInitialConditions, 			// ���������� ��������� �������
				d_values, 							// ���������
				amountOfValues, 					// ���������� ����������
				tMax / NT, 							// ���������� �������� (����������� �� tMax)
				1, 									// ��������� ��� ��������� ��������
				writableVar,						// ������ ���������� � x[] �� �������� ������ ���������
				maxValue, 							// ������������� �������� ���������� ��� �������������
				d_lleResult);						// �������������� ������

		// ------------------------------------

		// --- �������� �� CUDA ������ ---
		gpuGlobalErrorCheck();

		// --- ���� ���� ��� ������ �������� ���� ������ ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- ����������� �������� ����� � �� ���������� �� ������ GPU � ����������� ������ ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- �������� ����� � ��������� ������� ---
		outFileStream << std::setprecision(12);

		// --- ���������� ������ � ���� ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
			}
			else
			{
				mexPrintf("\nOutput file open error\n");
				exit(1);
			}
		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	// ---------------------------
	// --- ������������ ������ ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



__host__ void LLE2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = 200000; // Pizdec kostil' ot Boga

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)&d_lleResult, nPtsLimiter * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	mexPrintf("LLE2D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);
	int stringCounter = 0;

	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMax;
		int blockSizeMin;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LLEKernelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_lleResult[i];
				++stringCounter;
			}
			else
			{
				mexPrintf("\nOutput file open error\n");
				exit(1);
			}
		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}


__host__ void LLE2DIC(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.95;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	//nPtsLimiter = 22000; // Pizdec kostil' ot Boga

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)&d_lleResult, nPtsLimiter * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	mexPrintf("LLE2D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	int stringCounter = 0;
	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMax;
		int blockSizeMin;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LLEKernelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_lleResult[i];
				++stringCounter;
			}
			else
			{
				mexPrintf("\nOutput file open error\n");
				exit(1);
			}

		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}



__host__ void LS1D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 4;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter * amountOfInitialConditions];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)&d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	mexPrintf("LS1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double))* blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			1, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0);
				for (int j = 0; j < amountOfInitialConditions; ++j)
					outFileStream << ", " << h_lleResult[k * amountOfInitialConditions + j];
				outFileStream << '\n';
			}
			else
			{
				mexPrintf("\nOutput file open error\n");
				exit(1);
			}

		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}




__host__ void LS2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 4;
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts * nPts ? nPts * nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	double* h_lleResult = new double[nPtsLimiter * amountOfInitialConditions];

	double* d_ranges;
	int* d_indicesOfMutVars;
	double* d_initialConditions;
	double* d_values;

	double* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 4 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)&d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((double)nPts * (double)nPts) / (double)nPtsLimiter);

	std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	mexPrintf("LS2D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	int* stringCounter = new int[amountOfInitialConditions];

	for (int i = 0; i < amountOfInitialConditions; ++i)
		stringCounter[i] = 0;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		outFileStream.open(OUT_FILE_PATH + std::to_string(i + 1) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, 0, nPtsLimiter);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(double))* blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < amountOfInitialConditions; ++k)
		{
			outFileStream.open(OUT_FILE_PATH + std::to_string(k + 1) + ".csv", std::ios::app);
			for (size_t m = 0 + k; m < nPtsLimiter * amountOfInitialConditions; m = m + amountOfInitialConditions)
			{
				if (outFileStream.is_open())
				{
					if (stringCounter[k] != 0)
						outFileStream << ", ";
					if (stringCounter[k] == nPts)
					{
						outFileStream << "\n";
						stringCounter[k] = 0;
					}
					outFileStream << h_lleResult[m];
					stringCounter[k] = stringCounter[k] + 1;
				}
			}
			outFileStream.close();
		}
		mexPrintf("Progress: %f\%\n", (100.0f / (double)amountOfIteration) * (i + 1));
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] stringCounter;
	delete[] h_lleResult;
}
