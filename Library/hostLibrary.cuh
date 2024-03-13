#pragma once

// -----------------------
// --- Библиотеки CUDA ---
// -----------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// -----------------------

// --------------------------------------------
// --- KiShiVi библиотеки для работы с CUDA ---
// --------------------------------------------

#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"

// --------------------------------------------

// -----------------------------
// --- Встроенные библиотеки ---
// -----------------------------

#include <iomanip>
#include <string>

// -----------------------------



/**
 * Функция, для расчета одномерной бифуркационной диаграммы.
 */
__host__ void distributedSystemSimulation(
	const double	tMax,							// Время моделирования системы
	const double	h,								// Шаг интегрирования
	const double	hSpecial,						// Шаг смещения между потоками
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				// Массив с начальными условиями
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							// Параметры
	const int		amountOfValues);				// Количество параметров				



/**
 * Функция, для расчета одномерной бифуркационной диаграммы.
 */
__host__ void bifurcation1D(
	const double	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const double	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				// Массив с начальными условиями
	const double*	ranges,							// Диапазон изменения переменной
	const int*		indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller);					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)



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
	const int		preScaller);					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)



/**
 * Функция, для расчета одномерной бифуркационной диаграммы. (По начальным условиям)
 */
__host__ void bifurcation1DIC(
	const double	tMax,							  // Время моделирования системы
	const int		nPts,							  // Разрешение диаграммы
	const double	h,								  // Шаг интегрирования
	const int		amountOfInitialConditions,		  // Количество начальных условий ( уравнений в системе )
	const double*	initialConditions,				  // Массив с начальными условиями
	const double*	ranges,							  // Диапазон изменения начального условия
	const int*		indicesOfMutVars,				  // Индекс изменяемого начального условия
	const int		writableVar,					  // Индекс уравнения, по которому будем строить диаграмму
	const double	maxValue,						  // Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double	transientTime,					  // Время, которое будет промоделировано перед расчетом диаграммы
	const double*	values,							  // Параметры
	const int		amountOfValues,					  // Количество параметров
	const int		preScaller);					  // Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)



/**
 * Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN)
 */
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
	const double	eps);								// Эпсилон для алгоритма DBSCAN 



/**
 * Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) (for initial conditions)
 */
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
	const double	eps);								// Эпсилон для алгоритма DBSCAN 



/**
 * Построение 1D LLE диаграммы
 */
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
	const int		amountOfValues);					// Количество параметров



/**
 * Построение 1D LLE диаграммы (IC)
 */
__host__ void LLE1DIC(			
	const double tMax,									// Время моделирования системы
	const double NT,									// Время нормализации
	const int nPts,										// Разрешение диаграммы
	const double h,										// Шаг интегрирования
	const double eps,									// Эпсилон для LLE
	const double* initialConditions,					// Массив с начальными условиями
	const int amountOfInitialConditions,				// Количество начальных условий ( уравнений в системе )
	const double* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,						// Индексы изменяемых параметров
	const int writableVar,								// Индекс уравнения, по которому будем строить диаграмму
	const double maxValue,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const double transientTime,							// Время, которое будет промоделировано перед расчетом диаграммы
	const double* values,								// Параметры
	const int amountOfValues);							// Количество параметров



/**
 * Construction of a 2D LLE diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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
	const int amountOfValues);



/**
 * Construction of a 2D LLE diagram (for initial conditions)
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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
	const int amountOfValues);



/**
 * Construction of a 1D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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
	const int amountOfValues);




/**
 * Construction of a 2D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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
	const int amountOfValues);
