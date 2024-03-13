#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaMacros.cuh"

#include <fstream>
#include <stdio.h>



/**  
 * Вычисляет следующее значение дискретной модели
 * и записывает результат в x
 * 
 * \param x			- Начальные условия или значения переменных предыдущей точки
 * \param values	- Параметры
 * \param h			- Шаг интегрирования
 */
__device__ __host__ void calculateDiscreteModel(double* x, const double* values, const double h);



/**
 * Вычисляет траекторию для одной системы и записывает результат в "data" (если data != nullptr)
 * 
 * \param x						- Начальные условия для моделирования
 * \param values				- Параметры системы
 * \param h						- Шаг интегрирования
 * \param amountOfIterations	- Количество итераций
 * \param preScaller			- Множитель пропусков. Каждая 'preScaller' точка будет записана в результат
 * \param writableVar			- Какая из переменных в x[] будет записана в результат
 * \param maxValue				- Максимальное значение. Если будет x[writableVar] > maxValue, тогда функция вернет false
 * \param data					- Массив для записи данных
 * \param startDataIndex		- Индекс, с которого следует начинать запись в data
 * \param writeStep				- Шаг записи в массиве с данными (например, если шаг = 2, то запись будет в индексы: 0, 2, 4, ...)
 * \return						- Возаращет true, если ошибок не произошло
 */
__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, 
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller=0,
	const int writableVar = 0, const double maxValue = 0,
	double* data = nullptr, const int startDataIndex = 0, 
	const int writeStep = 1);



/**
 * Глобальная функция, которая распределенно вычисляет траекторию одной систем
 *
 * \param amountOfThreads			- 
 * \param h							- Шаг интегрирования
 * \param hSpecial					- 
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param data						- Массив, где будет хранится траектория систем
 * \return -
 */
__global__ void distributedCalculateDiscreteModelCUDA(
	const int		amountOfPointsForSkip,
	const int		amountOfThreads,
	const double	h,
	const double	hSpecial,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		writableVar = 0,
	double*			data = nullptr);



/**
 * Глобальная функция, которая вычисляет траекторию нескольких систем
 * 
 * \param nPts						- Общее разрешение диаграммы - nPts
 * \param nPtsLimiter				- Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
 * \param sizeOfBlock				- Количество точек в одной системе ( tMax / h / preScaller ) 
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек систем
 * \param amountOfPointsForSkip		- Количество точек для пропуска ( transientTime )
 * \param dimension					- Размерность ( диаграмма одномерная )
 * \param ranges					- Массив с диапазонами
 * \param h							- Шаг интегрирования
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param preScaller				- Множитель, который уменьшает время и объем расчетов
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param maxValue					- Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
 * \param data						- Массив, где будет хранится траектория систем
 * \param maxValueCheckerArray		- Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
 * \return -
 */
__global__ void calculateDiscreteModelCUDA(
	const int		nPts, 
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			data = nullptr,
	int*			maxValueCheckerArray = nullptr);

// --------------------------------------------------------------------------



/**
 * Глобальная функция, которая вычисляет траекторию нескольких систем по шагу
 *
 * \param nPts						- Общее разрешение диаграммы - nPts
 * \param nPtsLimiter				- Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
 * \param sizeOfBlock				- Количество точек в одной системе ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек систем
 * \param transientTime				- Время пропуска ( transientTime )
 * \param dimension					- Размерность ( диаграмма одномерная )
 * \param ranges					- Массив с диапазонами
 * \param h							- Шаг интегрирования
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param preScaller				- Множитель, который уменьшает время и объем расчетов
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param maxValue					- Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
 * \param data						- Массив, где будет хранится траектория систем
 * \param maxValueCheckerArray		- Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
 * \return -
 */
__global__ void calculateDiscreteModelCUDA_H(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const double	transientTime,
	const int		dimension,
	double*			ranges,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const double	amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double* data = nullptr,
	int* maxValueCheckerArray = nullptr);

// --------------------------------------------------------------------------



/**
 * Глобальная функция, которая вычисляет траекторию нескольких систем (по начальным условиям)
 *
 * \param nPts						- Общее разрешение диаграммы - nPts
 * \param nPtsLimiter				- Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
 * \param sizeOfBlock				- Количество точек в одной системе ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек систем
 * \param amountOfPointsForSkip		- Количество точек для пропуска ( transientTime )
 * \param dimension					- Размерность ( диаграмма одномерная )
 * \param ranges					- Массив с диапазонами
 * \param h							- Шаг интегрирования
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param preScaller				- Множитель, который уменьшает время и объем расчетов
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param maxValue					- Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
 * \param data						- Массив, где будет хранится траектория систем
 * \param maxValueCheckerArray		- Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
 * \return -
 */
__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			data = nullptr,
	int*			maxValueCheckerArray = nullptr);



/**
 * Функция, которая находит индекс в последовательности значений
 * Пример:
 * Последовательность:
 * 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5
 * 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5
 * 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 * 
 * getValueByIdx(7, 5, 1, 5, 0) = 3
 * getValueByIdx(7, 5, 1, 5, 1) = 2
 * getValueByIdx(7, 5, 1, 5, 2) = 1
 * 
 * \param idx			- Текущий idx в потоке
 * \param nPts			- Количество точек для разбиения диапазона (разрешение)
 * \param startRange	- Левая граница диапазона
 * \param finishRange	- Правая граница диапазона
 * \param valueNumber	- Номер требуемой переменной
 * \return Value		- Результат
 */
__device__ __host__ double getValueByIdx(const int idx, const int nPts, 
	const double startRange, const double finishRange, const int valueNumber);



/**
 * Функция, которая находит индекс в последовательности значений по логарифмической шкале
 *
 * \param idx			- Текущий idx в потоке
 * \param nPts			- Количество точек для разбиения диапазона (разрешение)
 * \param startRange	- Левая граница диапазона
 * \param finishRange	- Правая граница диапазона
 * \param valueNumber	- Номер требуемой переменной
 * \return Value		- Результат
 */
__device__ __host__ double getValueByIdxLog(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber);



/**
 * Находит пики в интервале [startDataIndex; startDataIndex + amountOfPoints] в "data" массиве
 * Результат записывается в outPeaks и timeOfPeaks ( если outPeaks != nullptr и timeOfPeaks != nullptr )
 * 
 * \param data				- Данные с точками
 * \param startDataIndex	- Индекс, откуда начинаем искать пики
 * \param amountOfPoints	- Количество точек для поиска пиков
 * \param outPeaks			- Выходной массив для записи в него найденных пиков
 * \param timeOfPeaks		- Выходной массив для записи в него индексов найденных пиков
 * \param h					- Шаг интегрирования (для вычисления межпикового интервала)
 * \return - Amount of found peaks
 */
__device__ __host__ int peakFinder(double* data, const int startDataIndex, const int amountOfPoints, 
	double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h=0);



/**
 * Нахождение пиков в "data" массиве в многопоточном режиме 
 * Результат записывается в "outPeaks", "timeOfPeaks" и "amountOfPeaks" ( если outPeaks != nullptr и timeOfPeaks != nullptr и amountOfPeaks != nullptr )
 * 
 * \param data				- Данные
 * \param sizeOfBlock		- Количество точек для одной системы ( tmax / h / preScaller )
 * \param amountOfBlocks	- Количество блоков ( систем ) в массиве "data"
 * \param amountOfPeaks		- Выходной массив, содержащий количество пиков для каждого блока ( системы )
 * \param outPeaks			- Выходной массив, содержащий fамплитуды найденных пиков
 * \param timeOfPeaks		- Выходной массив, содержащий межпиковые интервалы найденных пиков
 * \param h					- Шаг интегрирования ( для вычисления межпикового интервала )
 */
__global__ void peakFinderCUDA( double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h = 0 );



/**
 * Нахождение пиков в "data" массиве в многопоточном режиме
 * Результат записывается в "outPeaks", "timeOfPeaks" и "amountOfPeaks" ( если outPeaks != nullptr и timeOfPeaks != nullptr и amountOfPeaks != nullptr )
 *
 * \param data				- Данные
 * \param sizeOfBlock		- Количество точек для одной системы ( tmax / h / preScaller )
 * \param amountOfBlocks	- Количество блоков ( систем ) в массиве "data"
 * \param amountOfPeaks		- Выходной массив, содержащий количество пиков для каждого блока ( системы )
 * \param outPeaks			- Выходной массив, содержащий fамплитуды найденных пиков
 * \param timeOfPeaks		- Выходной массив, содержащий межпиковые интервалы найденных пиков
 * \param h					- Шаг интегрирования ( для вычисления межпикового интервала )
 */
__global__ void peakFinderCUDA_H(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h = 0);



/**
 * Вычисляет расстояние между двумя точками
 * 
 * \param x1 - x первой точки
 * \param y1 - y первой точки
 * \param x2 - x второй точки
 * \param y2 - y второй точки
 * \return - Дистанция
 */
__device__ __host__ double distance(double x1, double y1, double x2, double y2);



/**
 * Функция DBSCAN
 * 
 * \param data					- Данные (пики)
 * \param intervals				- Межпиковые интервалы
 * \param helpfulArray			- Вспомогательный массив
 * \param startDataIndex		- Индекс, с которого будет вестись запись в outData
 * \param amountOfPeaks			- Массив с количеством пиков в каждой системе
 * \param sizeOfHelpfulArray	- Размер вспомогательного массива
 * \param idx					- Текущий idx в потоке
 * \param eps					- Эпселон
 * \param outData				- Результирующий массив
 */
__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray,
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData);



/**
 * Глобальная функция DBSCAN
 * 
 * \param data				- Данные (пики)
 * \param sizeOfBlock		- Количество точек в одной системе
 * \param amountOfBlocks	- Количество блоков (систем) в data
 * \param amountOfPeaks		- Массив, содержащий количество пиков для каждого блока в data
 * \param intervals			- Межпиковые интервалы
 * \param helpfulArray		- Вспомогательный массив
 * \param eps				- Эпселон
 * \param outData			- Результирующий массив
 */
__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray, const double eps, int* outData);



/**
 * Ядро для LLE
 * 
 * \param nPts						- Общее разрешение
 * \param nPtsLimiter				- Разрешение в текущем расчете
 * \param NT						- Время нормализации
 * \param tMax						- Время моделирования
 * \param sizeOfBlock				- Количество точек, занимаемое одной системой в "data"
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек
 * \param amountOfPointsForSkip		- Количество точек, которое будет промоделированно до основного расчета (transientTime)
 * \param dimension					- Размерность
 * \param ranges					- Массив, содержащий диапазоны перебираемого параметра
 * \param h							- Шаг интегрирования
 * \param eps						- Эпсилон
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итерация (вычисляется от tMax)
 * \param preScaller				- Множитель для ускорения расчетов
 * \param writableVar				- Индекс переменной в x[] по которому строим диаграмму
 * \param maxValue					- Макксимальное значение переменной при моделировании
 * \param resultArray				- Результирующий массив
 * \return -
 */
__global__ void LLEKernelCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			resultArray = nullptr);



/**
 * Ядро для LLE (IC)
 *
 * \param nPts						- Общее разрешение
 * \param nPtsLimiter				- Разрешение в текущем расчете
 * \param NT						- Время нормализации
 * \param tMax						- Время моделирования
 * \param sizeOfBlock				- Количество точек, занимаемое одной системой в "data"
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек
 * \param amountOfPointsForSkip		- Количество точек, которое будет промоделированно до основного расчета (transientTime)
 * \param dimension					- Размерность
 * \param ranges					- Массив, содержащий диапазоны перебираемого параметра
 * \param h							- Шаг интегрирования
 * \param eps						- Эпсилон
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итерация (вычисляется от tMax)
 * \param preScaller				- Множитель для ускорения расчетов
 * \param writableVar				- Индекс переменной в x[] по которому строим диаграмму
 * \param maxValue					- Макксимальное значение переменной при моделировании
 * \param resultArray				- Результирующий массив
 * \return -
 */
__global__ void LLEKernelICCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const double	maxValue = 0,
	double*			resultArray = nullptr);



/**
 * Kernel for metric LS
 *
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param NT - Normalization time
 * \param tMax - Simulation time
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip	- Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Array with variable parameter ranges
 * \param h - Integration step
 * \param eps - Eps
 * \param indicesOfMutVars - Index of unknown variable
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Threshold signal level
 * \param resultArray - Result array
 * \return -
 */
__global__ void LSKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* resultArray = nullptr);
