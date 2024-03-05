// --- Заголовочный файл ---
#include "hostLibrary.cuh"
#include "mex.h"

// --- Путь для сохранения результирующих файлов ---
#define OUT_FILE_PATH "D:\\CUDABifurcation3.0\\MatlabScripts\\CUDA_OUT\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// --- Директива, объявление которой выводит в консоль отладочные сообщения ---


__host__ void distributedSystemSimulation(
	const double	tMax,							// Время моделирования системы
	const double	h,								// Шаг интегрирования
	const double	hSpecial,						// Шаг смещения между потоками
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				// Массив с начальными условиями
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							// Параметры
	const int		amountOfValues)					// Количество параметров	
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h;

	int amountOfThreads = hSpecial / h;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.8;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)	

	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_data = new double[amountOfPointsInBlock * sizeof(double)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_data,				amountOfPointsInBlock * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );
	
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof(double),			cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------


	mexPrintf("Distributed System Simulation\n");


		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil( ( 1024.0f * 8.0f ) / ( ( amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (amountOfThreads + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

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

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_data, d_data, amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
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
// --- Определение функции, для расчета одномерной бифуркационной диаграммы ---
// ----------------------------------------------------------------------------

__host__ void bifurcation1D(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const double	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				// Массив с начальными условиями
	const double*	ranges,							// Диаппазон изменения переменной
	const int*		indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------
	
	double* h_outPeaks		= new double	[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int*	h_amountOfPeaks = new int		[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int*	d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int*	d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
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
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof(double),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof(int),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof(double),			cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = ( size_t )ceil( ( double )nPts / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------


	mexPrintf("Bifurcation 1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);


	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, ( amountOfInitialConditions + amountOfValues ) * sizeof(double) * blockSize >> > 
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter );
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> > 
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks,		d_outPeaks,			nPtsLimiter * amountOfPointsInBlock * sizeof(double),	cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks,	d_amountOfPeaks,	nPtsLimiter * sizeof(int),								cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
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
	// --- Освобождение памяти ---
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
 * Функция, для расчета одномерной бифуркационной диаграммы по шагу.
 */
__host__ void bifurcation1DForH(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				// Массив с начальными условиями
	const double*	ranges,							// Диапазон изменения шага
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек в одном блоке ---
	int amountOfPointsInBlock = tMax / (ranges[0] < ranges[1] ? ranges[0] : ranges[1]) / preScaller;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	double* h_outPeaks = new double[nPtsLimiter * amountOfPointsInBlock * sizeof(double)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)&d_data, nPtsLimiter* amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 2 * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(double)));

	gpuErrorCheck(cudaMalloc((void**)&d_outPeaks, nPtsLimiter* amountOfPointsInBlock * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("Bifurcation 1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA_H << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				transientTime,				// Время пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				tMax,						// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA_H << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
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
	// --- Освобождение памяти ---
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
// --- Функция, для расчета одномерной бифуркационной диаграммы. (По начальным условиям) ---
// -----------------------------------------------------------------------------------------

__host__ void bifurcation1DIC(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const double	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				// Массив с начальными условиями
	const double*	ranges,							// Диаппазон изменения переменной
	const int*		indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------
	
	// Пояснение: пиков не может быть больше, чем (amountOfPointsInBlock / 2), т.к. после пика не может снова идти пик
	double* h_outPeaks		= new double	[ceil(nPtsLimiter * amountOfPointsInBlock * sizeof(double) / 2.0f)];
	int*	h_amountOfPeaks = new int		[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int*	d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	double* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int*	d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
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
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof(double),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof(int),							cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof(double),			cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = ( size_t )ceil( ( double )nPts / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("Bifurcation 1DIC\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512: blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, ( amountOfInitialConditions + amountOfValues ) * sizeof(double) * blockSize >> > 
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter );
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> > 
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks,		d_outPeaks,			nPtsLimiter * amountOfPointsInBlock * sizeof(double),	cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks,	d_amountOfPeaks,	nPtsLimiter * sizeof(int),								cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
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
	// --- Освобождение памяти ---
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
// --- Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) ---
// ------------------------------------------------------------------------

__host__ void bifurcation2D(
	const double	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const double	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,					// Массив с начальными условиями
	const double*	ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const double	eps)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > ( nPts * nPts ) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int*	d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	int*	d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	double* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int*	d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	double* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
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
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				4 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	2 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice) );

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = ( size_t )ceil( ( double )( nPts * nPts ) / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------


	/*mexPrintf("Bifurcation 2D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);*/


	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for ( int i = 0; i < amountOfIteration; ++i )
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if ( i == amountOfIteration - 1 )
			nPtsLimiter = ( nPts * nPts ) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		blockSize = blockSize > 512 ? 512 : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				h * preScaller);							// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
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

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
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
	// --- Освобождение памяти ---
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
// --- Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) по IC ---
// ------------------------------------------------------------------------------

__host__ void bifurcation2DIC(
	const double	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const double	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,					// Массив с начальными условиями
	const double*	ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const double	eps)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > ( nPts * nPts ) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	int* h_dbscanResult = new int[nPtsLimiter * sizeof(double)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	double* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int*	d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	// Указатель на массив с начальными условиями
	double* d_values;				// Указатель на массив с параметрами

	int*	d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	double* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int*	d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	double* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
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
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				4 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	2 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice) );

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = ( size_t )ceil( ( double )( nPts * nPts ) / ( double )nPtsLimiter );

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("Bifurcation 2DIC\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for ( int i = 0; i < amountOfIteration; ++i )
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if ( i == amountOfIteration - 1 )
			nPtsLimiter = ( nPts * nPts ) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(double) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(double)));
		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(double)* blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				h * preScaller);							// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> > 
			(	d_data, 					// Данные (пики)
				amountOfPointsInBlock, 		// Количество точек в одной системе
				nPtsLimiter,				// Количество блоков (систем) в data
				d_amountOfPeaks, 			// Массив, содержащий количество пиков для каждого блока в data
				d_intervals, 				// Межпиковые интервалы
				d_helpfulArray, 			// Вспомогательный массив 
				eps, 						// Эпселон
				d_dbscanResult);			// Результирующий массив

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---
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
	// --- Освобождение памяти ---
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
	const double	tMax,								// Время моделирования системы
	const double	NT,									// Время нормализации
	const int		nPts,								// Разрешение диаграммы
	const double	h,									// Шаг интегрирования
	const double	eps,								// Эпсилон для LLE
	const double*	initialConditions,					// Массив с начальными условиями
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const double*	ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,								// Параметры
	const int		amountOfValues)						// Количество параметров
{
	// --- Количество точек, которое будет смоделировано одной системой во время нормализации NT ---
	size_t amountOfNT_points = NT / h;

	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / NT;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;																// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;																// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_ranges;				   // Указатель на массив с диапазоном изменения переменной
	int*	d_indicesOfMutVars;		   // Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	   // Указатель на массив с начальными условиями
	double* d_values;				   // Указатель на массив с параметрами

	double* d_lleResult;			   // Память для хранения конечного результата

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_ranges,				2 * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_indicesOfMutVars,	1 * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );
					 		  	 
	gpuErrorCheck( cudaMalloc( ( void** )&d_lleResult,			nPtsLimiter * sizeof(double)));
			
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice ) );

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

	mexPrintf("LLE 1D\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// Переменная для хранения размера блока
		int minGridSize;	// Переменная для хранения минимального размера сетки
		int gridSize;		// Переменная для хранения сетки

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( 3 * amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );

		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )


		// ------------------------------------
		// --- CUDA функция для расчета LLE ---
		// ------------------------------------

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > 
			(	nPts,								// Общее разрешение
				nPtsLimiter, 						// Разрешение в текущем расчете
				NT, 								// Время нормализации
				tMax, 								// Время моделирования
				amountOfPointsInBlock,				// Количество точек, занимаемое одной системой в "data"
				i * originalNPtsLimiter, 			// Количество уже посчитанных точек
				amountOfPointsForSkip,				// Количество точек, которое будет промоделированно до основного расчета (transientTime)
				1, 									// Размерность
				d_ranges, 							// Массив, содержащий диапазоны перебираемого параметра
				h, 									// Шаг интегрирования
				eps, 								// Эпсилон
				d_indicesOfMutVars, 				// Индексы изменяемых параметров
				d_initialConditions,				// Начальные условия
				amountOfInitialConditions, 			// Количество начальных условий
				d_values, 							// Параметры
				amountOfValues, 					// Количество параметров
				tMax / NT, 							// Количество итерация (вычисляется от tMax)
				1, 									// Множитель для ускорения расчетов
				writableVar,						// Индекс переменной в x[] по которому строим диаграмму
				maxValue, 							// Макксимальное значение переменной при моделировании
				d_lleResult);						// Результирующий массив

		// ------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---

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
	// --- Освобождение памяти ---
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
	// --- Количество точек, которое будет смоделировано одной системой во время нормализации NT ---
	size_t amountOfNT_points = NT / h;

	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / NT;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;																// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;																// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(double) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	double* h_lleResult = new double[nPtsLimiter];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	double* d_ranges;				   // Указатель на массив с диапазоном изменения переменной
	int*	d_indicesOfMutVars;		   // Указатель на массив с индексом изменяемой переменной в массиве values
	double* d_initialConditions;	   // Указатель на массив с начальными условиями
	double* d_values;				   // Указатель на массив с параметрами

	double* d_lleResult;			   // Память для хранения конечного результата

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck( cudaMalloc( ( void** )&d_ranges,				2 * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_indicesOfMutVars,	1 * sizeof( int ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_initialConditions,	amountOfInitialConditions * sizeof( double ) ) );
	gpuErrorCheck( cudaMalloc( ( void** )&d_values,				amountOfValues * sizeof( double ) ) );
					 		  	 
	gpuErrorCheck( cudaMalloc( ( void** )&d_lleResult,			nPtsLimiter * sizeof(double)));
			
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck( cudaMemcpy( d_ranges,			ranges,				2 * sizeof( double ),							cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_indicesOfMutVars,	indicesOfMutVars,	1 * sizeof( int ),								cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_initialConditions, initialConditions,	amountOfInitialConditions * sizeof( double ),	cudaMemcpyKind::cudaMemcpyHostToDevice ) );
	gpuErrorCheck( cudaMemcpy( d_values,			values,				amountOfValues * sizeof( double ),				cudaMemcpyKind::cudaMemcpyHostToDevice ) );

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceilf((double)nPts / (double)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------
	mexPrintf("LLE 1DIC\n");
	mexPrintf("nPtsLimiter : %zu\n", nPtsLimiter);
	mexPrintf("Amount of iterations %zu: \n", amountOfIteration);

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// Переменная для хранения размера блока
		int minGridSize;	// Переменная для хранения минимального размера сетки
		int gridSize;		// Переменная для хранения сетки

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(double));
		//blockSizeMin = (3 + amountOfValues) * sizeof(double);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil( ( 1024.0f * 32.0f ) / ( ( 3 * amountOfInitialConditions + amountOfValues ) * sizeof( double ) ) );

		if (blockSize < 1)
		{
			mexPrintf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )


		// ------------------------------------
		// --- CUDA функция для расчета LLE ---
		// ------------------------------------

		LLEKernelICCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize >> > 
			(	nPts,								// Общее разрешение
				nPtsLimiter, 						// Разрешение в текущем расчете
				NT, 								// Время нормализации
				tMax, 								// Время моделирования
				amountOfPointsInBlock,				// Количество точек, занимаемое одной системой в "data"
				i * originalNPtsLimiter, 			// Количество уже посчитанных точек
				amountOfPointsForSkip,				// Количество точек, которое будет промоделированно до основного расчета (transientTime)
				1, 									// Размерность
				d_ranges, 							// Массив, содержащий диапазоны перебираемого параметра
				h, 									// Шаг интегрирования
				eps, 								// Эпсилон
				d_indicesOfMutVars, 				// Индексы изменяемых параметров
				d_initialConditions,				// Начальные условия
				amountOfInitialConditions, 			// Количество начальных условий
				d_values, 							// Параметры
				amountOfValues, 					// Количество параметров
				tMax / NT, 							// Количество итерация (вычисляется от tMax)
				1, 									// Множитель для ускорения расчетов
				writableVar,						// Индекс переменной в x[] по которому строим диаграмму
				maxValue, 							// Макксимальное значение переменной при моделировании
				d_lleResult);						// Результирующий массив

		// ------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(12);

		// --- Сохранение данных в файл ---

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
	// --- Освобождение памяти ---
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
