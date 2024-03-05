#include "cudaLibrary.cuh"

// ---------------------------------------------------------------------------------
// --- Вычисляет следующее значение дискретной модели и записывает результат в x ---
// ---------------------------------------------------------------------------------

__device__ __host__ void calculateDiscreteModel(double* x, const double* a, const double h)
{
	/**
	 * here we abstract from the concept of parameter names. 
	 * ALL parameters are numbered with indices. 
	 * In the current example, the parameters go like this:
	 * 
	 * values[0] - sym
	 * values[1] - A
	 * values[2] - B
	 * values[3] - C
	 */

	double h1 = h * a[0];
	double h2 = h * (1 - a[0]);
	x[0] = x[0] + h1 * (-x[1] - x[2]);
	x[1] = (x[1] + h1 * (x[0])) / (1 - a[1] * h1);
	x[2] = (x[2] + h1 * a[2]) / (1 - h1 * (x[0] - a[3]));
	x[2] = x[2] + h2 * (a[2] + x[2] * (x[0] - a[3]));
	x[1] = x[1] + h2 * (x[0] + a[1] * x[1]);
	x[0] = x[0] + h2 * (-x[1] - x[2]);

	//x[0] = x[0] + h * (-x[1] - x[2]);
	//x[1] = x[1] + h * (x[0] + a[0] * x[1]);
	//x[2] = x[2] + h * (a[1] + x[2] * (x[0] - a[2]));

	/*double k11, k21, k31, k12, k22, k32, k13, k23, k33, k14, k24, k34, k15, k25, k35, k16, k26, k36, k17, k27, k37, k18, k28, k38, k19, k29, k39;
	double k1A0, k2A0, k3A0, k1A1, k2A1, k3A1, k1A2, k2A2, k3A2, k1A3, k2A3, k3A3, x01, x02, x03;

	double B[2][13];
	B[0][0] = 0.04174749114153;
	B[0][1] = 0;
	B[0][2] = 0;
	B[0][3] = 0;
	B[0][4] = 0;
	B[0][5] = -0.05545232861124;
	B[0][6] = 0.2393128072012;
	B[0][7] = 0.7035106694034;
	B[0][8] = -0.7597596138145;
	B[0][9] = 0.6605630309223;
	B[0][10] = 0.1581874825101;
	B[0][11] = -0.2381095387529;
	B[0][12] = 0.25;

	B[1][0] = 0.02955321367635;
	B[1][1] = 0;
	B[1][2] = 0;
	B[1][3] = 0;
	B[1][4] = 0;
	B[1][5] = -0.8286062764878;
	B[1][6] = 0.3112409000511;
	B[1][7] = 2.4673451906;
	B[1][8] = -2.546941651842;
	B[1][9] = 1.443548583677;
	B[1][10] = 0.07941559588113;
	B[1][11] = 0.04444444444444;
	B[1][12] = 0;

	double M[13][12];
	M[0][0] = 0;
	M[0][1] = 0;
	M[0][2] = 0;
	M[0][3] = 0;
	M[0][4] = 0;
	M[0][5] = 0;
	M[0][6] = 0;
	M[0][7] = 0;
	M[0][8] = 0;
	M[0][9] = 0;
	M[0][10] = 0;
	M[0][11] = 0;

	M[1][0] = 0.05555555555556;
	M[1][1] = 0;
	M[1][2] = 0;
	M[1][3] = 0;
	M[1][4] = 0;
	M[1][5] = 0;
	M[1][6] = 0;
	M[1][7] = 0;
	M[1][8] = 0;
	M[1][9] = 0;
	M[1][10] = 0;
	M[1][11] = 0;

	M[2][0] = 0.02083333333333;
	M[2][1] = 0.0625;
	M[2][2] = 0;
	M[2][3] = 0;
	M[2][4] = 0;
	M[2][5] = 0;
	M[2][6] = 0;
	M[2][7] = 0;
	M[2][8] = 0;
	M[2][9] = 0;
	M[2][10] = 0;
	M[2][11] = 0;

	M[3][0] = 0.03125;
	M[3][1] = 0;
	M[3][2] = 0.09375;
	M[3][3] = 0;
	M[3][4] = 0;
	M[3][5] = 0;
	M[3][6] = 0;
	M[3][7] = 0;
	M[3][8] = 0;
	M[3][9] = 0;
	M[3][10] = 0;
	M[3][11] = 0;

	M[4][0] = 0.3125;
	M[4][1] = 0;
	M[4][2] = -1.171875;
	M[4][3] = 1.171875;
	M[4][4] = 0;
	M[4][5] = 0;
	M[4][6] = 0;
	M[4][7] = 0;
	M[4][8] = 0;
	M[4][9] = 0;
	M[4][10] = 0;
	M[4][11] = 0;

	M[5][0] = 0.0375;
	M[5][1] = 0;
	M[5][2] = 0;
	M[5][3] = 0.1875;
	M[5][4] = 0.15;
	M[5][5] = 0;
	M[5][6] = 0;
	M[5][7] = 0;
	M[5][8] = 0;
	M[5][9] = 0;
	M[5][10] = 0;
	M[5][11] = 0;

	M[6][0] = 0.04791013711111;
	M[6][1] = 0;
	M[6][2] = 0;
	M[6][3] = 0.1122487127778;
	M[6][4] = -0.02550567377778;
	M[6][5] = 0.01284682388889;
	M[6][6] = 0;
	M[6][7] = 0;
	M[6][8] = 0;
	M[6][9] = 0;
	M[6][10] = 0;
	M[6][11] = 0;

	M[7][0] = 0.01691798978729;
	M[7][1] = 0;
	M[7][2] = 0;
	M[7][3] = 0.387848278486;
	M[7][4] = 0.0359773698515;
	M[7][5] = 0.1969702142157;
	M[7][6] = -0.1727138523405;
	M[7][7] = 0;
	M[7][8] = 0;
	M[7][9] = 0;
	M[7][10] = 0;
	M[7][11] = 0;

	M[8][0] = 0.06909575335919;
	M[8][1] = 0;
	M[8][2] = 0;
	M[8][3] = -0.6342479767289;
	M[8][4] = -0.1611975752246;
	M[8][5] = 0.1386503094588;
	M[8][6] = 0.9409286140358;
	M[8][7] = 0.2116363264819;
	M[8][8] = 0;
	M[8][9] = 0;
	M[8][10] = 0;
	M[8][11] = 0;

	M[9][0] = 0.183556996839;
	M[9][1] = 0;
	M[9][2] = 0;
	M[9][3] = -2.468768084316;
	M[9][4] = -0.2912868878163;
	M[9][5] = -0.02647302023312;
	M[9][6] = 2.847838764193;
	M[9][7] = 0.2813873314699;
	M[9][8] = 0.1237448998633;
	M[9][9] = 0;
	M[9][10] = 0;
	M[9][11] = 0;

	M[10][0] = -1.215424817396;
	M[10][1] = 0;
	M[10][2] = 0;
	M[10][3] = 16.67260866595;
	M[10][4] = 0.9157418284168;
	M[10][5] = -6.056605804357;
	M[10][6] = -16.00357359416;
	M[10][7] = 14.8493030863;
	M[10][8] = -13.37157573529;
	M[10][9] = 5.13418264818;
	M[10][10] = 0;
	M[10][11] = 0;

	M[11][0] = 0.2588609164383;
	M[11][1] = 0;
	M[11][2] = 0;
	M[11][3] = -4.774485785489;
	M[11][4] = -0.435093013777;
	M[11][5] = -3.049483332072;
	M[11][6] = 5.577920039936;
	M[11][7] = 6.155831589861;
	M[11][8] = -5.062104586737;
	M[11][9] = 2.193926173181;
	M[11][10] = 0.1346279986593;
	M[11][11] = 0;

	M[12][0] = 0.8224275996265;
	M[12][1] = 0;
	M[12][2] = 0;
	M[12][3] = -11.65867325728;
	M[12][4] = -0.7576221166909;
	M[12][5] = 0.7139735881596;
	M[12][6] = 12.07577498689;
	M[12][7] = -2.12765911392;
	M[12][8] = 1.990166207049;
	M[12][9] = -0.234286471544;
	M[12][10] = 0.1758985777079;
	M[12][11] = 0;

	k11 = -x[1] - x[2];
	k21 = x[0] + a[0] * x[1];
	k31 = a[1] + x[2] * (x[0] - a[2]);


	x01 = x[0] + h * M[1][0] * k11;
	x02 = x[1] + h * M[1][0] * k21;
	x03 = x[2] + h * M[1][0] * k31;

	k12 = -x02 - x03;
	k22 = x01 + a[0] * x02;
	k32 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[2][0] * k11 + M[2][1] * k12);
	x02 = x[1] + h * (M[2][0] * k21 + M[2][1] * k22);
	x03 = x[2] + h * (M[2][0] * k31 + M[2][1] * k32);

	k13 = -x02 - x03;
	k23 = x01 + a[0] * x02;
	k33 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[3][0] * k11 + M[3][1] * k12 + M[3][2] * k13);
	x02 = x[1] + h * (M[3][0] * k21 + M[3][1] * k22 + M[3][2] * k23);
	x03 = x[2] + h * (M[3][0] * k31 + M[3][1] * k32 + M[3][2] * k33);

	k14 = -x02 - x03;
	k24 = x01 + a[0] * x02;
	k34 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[4][0] * k11 + M[4][1] * k12 + M[4][2] * k13 + M[4][3] * k14);
	x02 = x[1] + h * (M[4][0] * k21 + M[4][1] * k22 + M[4][2] * k23 + M[4][3] * k24);
	x03 = x[2] + h * (M[4][0] * k31 + M[4][1] * k32 + M[4][2] * k33 + M[4][3] * k34);

	k15 = -x02 - x03;
	k25 = x01 + a[0] * x02;
	k35 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[5][0] * k11 + M[5][1] * k12 + M[5][2] * k13 + M[5][3] * k14 + M[5][4] * k15);
	x02 = x[1] + h * (M[5][0] * k21 + M[5][1] * k22 + M[5][2] * k23 + M[5][3] * k24 + M[5][4] * k25);
	x03 = x[2] + h * (M[5][0] * k31 + M[5][1] * k32 + M[5][2] * k33 + M[5][3] * k34 + M[5][4] * k35);

	k16 = -x02 - x03;
	k26 = x01 + a[0] * x02;
	k36 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[6][0] * k11 + M[6][1] * k12 + M[6][2] * k13 + M[6][3] * k14 + M[6][4] * k15 + M[6][5] * k16);
	x02 = x[1] + h * (M[6][0] * k21 + M[6][1] * k22 + M[6][2] * k23 + M[6][3] * k24 + M[6][4] * k25 + M[6][5] * k26);
	x03 = x[2] + h * (M[6][0] * k31 + M[6][1] * k32 + M[6][2] * k33 + M[6][3] * k34 + M[6][4] * k35 + M[6][5] * k36);

	k17 = -x02 - x03;
	k27 = x01 + a[0] * x02;
	k37 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[7][0] * k11 + M[7][1] * k12 + M[7][2] * k13 + M[7][3] * k14 + M[7][4] * k15 + M[7][5] * k16 + M[7][6] * k17);
	x02 = x[1] + h * (M[7][0] * k21 + M[7][1] * k22 + M[7][2] * k23 + M[7][3] * k24 + M[7][4] * k25 + M[7][5] * k26 + M[7][6] * k27);
	x03 = x[2] + h * (M[7][0] * k31 + M[7][1] * k32 + M[7][2] * k33 + M[7][3] * k34 + M[7][4] * k35 + M[7][5] * k36 + M[7][6] * k37);

	k18 = -x02 - x03;
	k28 = x01 + a[0] * x02;
	k38 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[8][0] * k11 + M[8][1] * k12 + M[8][2] * k13 + M[8][3] * k14 + M[8][4] * k15 + M[8][5] * k16 + M[8][6] * k17 + M[8][7] * k18);
	x02 = x[1] + h * (M[8][0] * k21 + M[8][1] * k22 + M[8][2] * k23 + M[8][3] * k24 + M[8][4] * k25 + M[8][5] * k26 + M[8][6] * k27 + M[8][7] * k28);
	x03 = x[2] + h * (M[8][0] * k31 + M[8][1] * k32 + M[8][2] * k33 + M[8][3] * k34 + M[8][4] * k35 + M[8][5] * k36 + M[8][6] * k37 + M[8][7] * k38);

	k19 = -x02 - x03;
	k29 = x01 + a[0] * x02;
	k39 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[9][0] * k11 + M[9][1] * k12 + M[9][2] * k13 + M[9][3] * k14 + M[9][4] * k15 + M[9][5] * k16 + M[9][6] * k17 + M[9][7] * k18 + M[9][8] * k19);
	x02 = x[1] + h * (M[9][0] * k21 + M[9][1] * k22 + M[9][2] * k23 + M[9][3] * k24 + M[9][4] * k25 + M[9][5] * k26 + M[9][6] * k27 + M[9][7] * k28 + M[9][8] * k29);
	x03 = x[2] + h * (M[9][0] * k31 + M[9][1] * k32 + M[9][2] * k33 + M[9][3] * k34 + M[9][4] * k35 + M[9][5] * k36 + M[9][6] * k37 + M[9][7] * k38 + M[9][8] * k39);

	k1A0 = -x02 - x03;
	k2A0 = x01 + a[0] * x02;
	k3A0 = a[1] + x03 * (x01 - a[2]);

	x01 = x[0] + h * (M[10][0] * k11 + M[10][1] * k12 + M[10][2] * k13 + M[10][3] * k14 + M[10][4] * k15 + M[10][5] * k16 + M[10][6] * k17 + M[10][7] * k18 + M[10][8] * k19 + M[10][9] * k1A0);
	x02 = x[1] + h * (M[10][0] * k21 + M[10][1] * k22 + M[10][2] * k23 + M[10][3] * k24 + M[10][4] * k25 + M[10][5] * k26 + M[10][6] * k27 + M[10][7] * k28 + M[10][8] * k29 + M[10][9] * k2A0);
	x03 = x[2] + h * (M[10][0] * k31 + M[10][1] * k32 + M[10][2] * k33 + M[10][3] * k34 + M[10][4] * k35 + M[10][5] * k36 + M[10][6] * k37 + M[10][7] * k38 + M[10][8] * k39 + M[10][9] * k3A0);

	k1A1 = -x02 - x03;
	k2A1 = x01 + a[0] * x02;
	k3A1 = a[1] + x03 * (x01 - a[2]);


	x01 = x[0] + h * (M[11][0] * k11 + M[11][1] * k12 + M[11][2] * k13 + M[11][3] * k14 + M[11][4] * k15 + M[11][5] * k16 + M[11][6] * k17 + M[11][7] * k18 + M[11][8] * k19 + M[11][9] * k1A0 + M[11][10] * k1A1);
	x02 = x[1] + h * (M[11][0] * k21 + M[11][1] * k22 + M[11][2] * k23 + M[11][3] * k24 + M[11][4] * k25 + M[11][5] * k26 + M[11][6] * k27 + M[11][7] * k28 + M[11][8] * k29 + M[11][9] * k2A0 + M[11][10] * k2A1);
	x03 = x[2] + h * (M[11][0] * k31 + M[11][1] * k32 + M[11][2] * k33 + M[11][3] * k34 + M[11][4] * k35 + M[11][5] * k36 + M[11][6] * k37 + M[11][7] * k38 + M[11][8] * k39 + M[11][9] * k3A0 + M[11][10] * k3A1);

	k1A2 = -x02 - x03;
	k2A2 = x01 + a[0] * x02;
	k3A2 = a[1] + x03 * (x01 - a[2]);


	x01 = x[0] + h * (M[12][0] * k11 + M[12][1] * k12 + M[12][2] * k13 + M[12][3] * k14 + M[12][4] * k15 + M[12][5] * k16 + M[12][6] * k17 + M[12][7] * k18 + M[12][8] * k19 + M[12][9] * k1A0 + M[12][10] * k1A1 + M[12][11] * k1A2);
	x02 = x[1] + h * (M[12][0] * k21 + M[12][1] * k22 + M[12][2] * k23 + M[12][3] * k24 + M[12][4] * k25 + M[12][5] * k26 + M[12][6] * k27 + M[12][7] * k28 + M[12][8] * k29 + M[12][9] * k2A0 + M[12][10] * k2A1 + M[12][11] * k2A2);
	x03 = x[2] + h * (M[12][0] * k31 + M[12][1] * k32 + M[12][2] * k33 + M[12][3] * k34 + M[12][4] * k35 + M[12][5] * k36 + M[12][6] * k37 + M[12][7] * k38 + M[12][8] * k39 + M[12][9] * k3A0 + M[12][10] * k3A1 + M[12][11] * k3A2);

	k1A3 = -x02 - x03;
	k2A3 = x01 + a[0] * x02;
	k3A3 = a[1] + x03 * (x01 - a[2]);


	x[0] = x[0] + h * (B[0][0] * k11 + B[0][1] * k12 + B[0][2] * k13 + B[0][3] * k14 + B[0][4] * k15 + B[0][5] * k16 + B[0][6] * k17 + B[0][7] * k18 + B[0][8] * k19 + B[0][9] * k1A0 + B[0][10] * k1A1 + B[0][11] * k1A2 + B[0][12] * k1A3);
	x[1] = x[1] + h * (B[0][0] * k21 + B[0][1] * k22 + B[0][2] * k23 + B[0][3] * k24 + B[0][4] * k25 + B[0][5] * k26 + B[0][6] * k27 + B[0][7] * k28 + B[0][8] * k29 + B[0][9] * k2A0 + B[0][10] * k2A1 + B[0][11] * k2A2 + B[0][12] * k2A3);
	x[2] = x[2] + h * (B[0][0] * k31 + B[0][1] * k32 + B[0][2] * k33 + B[0][3] * k34 + B[0][4] * k35 + B[0][5] * k36 + B[0][6] * k37 + B[0][7] * k38 + B[0][8] * k39 + B[0][9] * k3A0 + B[0][10] * k3A1 + B[0][11] * k3A2 + B[0][12] * k3A3);*/

	//z[0] = x[0] + h * (B[1][0] * k11 + B[1][1] * k12 + B[1][2] * k13 + B[1][3] * k14 + B[1][4] * k15 + B[1][5] * k16 + B[1][6] * k17 + B[1][7] * k18 + B[1][8] * k19 + B[1][9] * k1A0 + B[1][10] * k1A1 + B[1][11] * k1A2 + B[1][12] * k1A3);
	//z[1] = x[1] + h * (B[1][0] * k21 + B[1][1] * k22 + B[1][2] * k23 + B[1][3] * k24 + B[1][4] * k25 + B[1][5] * k26 + B[1][6] * k27 + B[1][7] * k28 + B[1][8] * k29 + B[1][9] * k2A0 + B[1][10] * k2A1 + B[1][11] * k2A2 + B[1][12] * k2A3);
	//z[2] = x[2] + h * (B[1][0] * k31 + B[1][1] * k32 + B[1][2] * k33 + B[1][3] * k34 + B[1][4] * k35 + B[1][5] * k36 + B[1][6] * k37 + B[1][7] * k38 + B[1][8] * k39 + B[1][9] * k3A0 + B[1][10] * k3A1 + B[1][11] * k3A2 + B[1][12] * k3A3);

	//int i;
	//double h1;
	//double b[17];
	//b[0] = 0.1302024830889;
	//b[1] = 0.5611629817751;
	//b[2] = -0.3894749626448;
	//b[3] = 0.1588419065552;
	//b[4] = -0.3959038941332;
	//b[5] = 0.1845396409783;
	//b[6] = 0.2583743876863;
	//b[7] = 0.2950117236093;
	//b[8] = -0.60550853383;
	//b[9] = 0.2950117236093;
	//b[10] = 0.2583743876863;
	//b[11] = 0.1845396409783;
	//b[12] = -0.3959038941332;
	//b[13] = 0.1588419065552;
	//b[14] = -0.3894749626448;
	//b[15] = 0.5611629817751;
	//b[16] = 0.1302024830889;

	//for (i = 0; i < 17; ++i)
	//{
	//	h1 = h * 0.5 * b[i];
	//	x[0] = x[0] + h1 * (-x[1] - x[2]);
	//	x[1] = (x[1] + h1 * x[0]) / (1 - a[0] * h1);
	//	x[2] = (x[2] + h1 * a[1]) / (1 - h1 * (x[0] - a[2]));
	//	x[2] = x[2] + h1 * (a[1] + x[2] * (x[0] - a[2]));
	//	x[1] = x[1] + h1 * (x[0] + a[0] * x[1]);
	//	x[0] = x[0] + h1 * (-x[1] - x[2]);
	//}

}



// -----------------------------------------------------------------------------------------------------
// --- Вычисляет траекторию для одной системы и записывает результат в "data" (если data != nullptr) ---
// -----------------------------------------------------------------------------------------------------

__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, 
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const double maxValue, double* data, 
	const int startDataIndex, const int writeStep)
{
	double* xPrev = new double[amountOfX];
	// --- Глобальный цикл, который производит вычисления заданные amountOfIterations раз ---
	for ( int i = 0; i < amountOfIterations; ++i )
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		// --- Если все-таки передали массив для записи - записываем значение переменной ---
		if ( data != nullptr )
			data[startDataIndex + i * writeStep] = x[writableVar];

		// --- Моделируем систему preScaller раз ( то есть если preScaller > 1, то мы пропустим ( preScaller - 1 ) в смоделированной траектории ) ---
		for ( int j = 0; j < preScaller; ++j )
			calculateDiscreteModel(x, values, h);

		// --- Если isnan или isinf - возвращаем false, ибо это нежелательное поведение системы ---
		if ( isnan( x[writableVar] ) || isinf( x[writableVar] ) )
		{
			delete[] xPrev;
			return false;
		}

		// --- Если maxValue == 0, это значит пользователь не выставил ограничение, иначе требуется его проверить ---
		if ( maxValue != 0 )
			if ( fabsf( x[writableVar] ) > maxValue )
			{
				delete[] xPrev;
				return false;
			}
	}

	// --- Проверка на сваливание в точку ---
	double tempResult = 0;
	for (int j = 0; j < amountOfX; ++j)
	{
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
	}

	if (tempResult == 0)
	{
		delete[] xPrev;
		return false;
	}

	if (sqrt(tempResult) < 1e-12)
	{
		delete[] xPrev;
		return false;
	}

	delete[] xPrev;
	return true;
}



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
	const int		writableVar,
	double*			data)
{
	extern __shared__ double s[];
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfThreads)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0);

	loopCalculateDiscreteModel(localX, localValues, h, idx,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0, 0);

	loopCalculateDiscreteModel(localX, localValues, hSpecial, amountOfIterations,
		amountOfInitialConditions, 1, writableVar, 0, data, idx, amountOfThreads);

	return;
}



// --------------------------------------------------------------------------
// --- Глобальная функция, которая вычисляет траекторию нескольких систем ---
// --------------------------------------------------------------------------

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
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ double s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		1, amountOfInitialConditions, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Теперь уже по-взрослому моделируем систему --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;	

	return;
}



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
	const double	tMax,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			data,
	int*			maxValueCheckerArray)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ double s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	//// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	//for (int i = 0; i < dimension; ++i)
	//	localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
	//		nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	
	double h = pow((double)10, getValueByIdxLog(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0));

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, transientTime / h,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Теперь уже по-взрослому моделируем систему --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, tMax / h / preScaller,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;
	else
		maxValueCheckerArray[idx] = tMax / h / preScaller;

	return;
}



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
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ double s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Теперь уже по-взрослому моделируем систему --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;	

	return;
}


// --- Функция, которая находит индекс в последовательности значений ---
__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return startRange + ( ( ( int )( ( int )idx / powf( ( double )nPts, ( double )valueNumber) ) % nPts )
		* ( ( double )( finishRange - startRange ) / ( double )( nPts - 1 ) ) );
}



// --- Функция, которая находит индекс в последовательности значений ---
__device__ __host__ double getValueByIdxLog(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return log10(startRange) + (((int)((int)idx / powf((double)nPts, (double)valueNumber)) % nPts)
		* ((double)(log10(finishRange) - log10(startRange)) / (double)(nPts - 1)));
}



// ---------------------------------------------------------------------------------------------------
// --- Находит пики в интервале [startDataIndex; startDataIndex + amountOfPoints] в "data" массиве ---
// ---------------------------------------------------------------------------------------------------

__device__ __host__ int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- Переменная для хранения найденных пиков ---
	int amountOfPeaks = 0;

	// --- Начинаем просматривать заданных интервал на наличие пиков ---
	for ( int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i )
	{
		// --- Если текущая точка больше предыдущей и больше ИЛИ РАВНА следующей, то... ( не факт, что это пик ( например: 2 3 3 4 ) ) ---
		if ( data[i] > data[i - 1] && data[i] >= data[i + 1] )
		{
			// --- От найденной точки начинаем идти вперед, пока не наткнемся на точку строго больше или меньше ---
			for ( int j = i; j < startDataIndex + amountOfPoints - 1; ++j )
			{
				// --- Если наткнулись на точку строго больше, значит это был не пик ---
				if ( data[j] < data[j + 1] )
				{
					i = j + 1;	// --- Обновляем внешний счетчик, чтобы дважды не проходить один и тот же интервал
					break;		// --- Возвращаемся к внешнему циклу
				}
				// --- Если о чудо, мы нашли точку меньше, чем текущая, значит мы нашли пик ---
				if ( data[j] > data[j + 1] )
				{
					// --- Если массик outPeaks не пуст, то делаем запись ---
					if ( outPeaks != nullptr )
						outPeaks[startDataIndex + amountOfPeaks] = data[j];
					// --- Если массик timeOfPeaks не пуст, то делаем запись ---
					if ( timeOfPeaks != nullptr )
						timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (double)j + (double)i ) / (double)2 );	// Выбираем индекс посередине между j и i
					++amountOfPeaks;
					i = j + 1; // Потому что следующая точка точно не может быть пиком ( два пика не могут идти подряд )
					break;
				}
			}
		}
	}
	// --- Вычисляем межпиковые интервалы ---
	if ( amountOfPeaks > 1 ) {
		// --- Пробегаемся по всем найденным пикам и их индексам ---
		for ( size_t i = 0; i < amountOfPeaks - 1; i++ )
		{
			// --- Смещаем все пики на один индекс влево, а первый пик удаляем ---
			if ( outPeaks != nullptr )
				outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
			// --- Вычисляем межпиковый интервал. Это разница индекса следующего прика и предыдущего, умноженная на шаг ---
			if ( timeOfPeaks != nullptr )
				timeOfPeaks[startDataIndex + i] = ( double )( ( timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i] ) * h );
		}
		// --- Так как один пик удалили - вычитаем единицу из результата ---
		amountOfPeaks = amountOfPeaks - 1;
	}
	else {
		amountOfPeaks = 0;
	}


	return amountOfPeaks;
}



// ----------------------------------------------------------------
// --- Нахождение пиков в "data" массиве в многопоточном режиме ---
// ----------------------------------------------------------------

__global__ void peakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks, 
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx >= amountOfBlocks )		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if ( amountOfPeaks[idx] == -1 )
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	
	amountOfPeaks[idx] = peakFinder( data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h );
	return;
}



__global__ void peakFinderCUDA_H(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, amountOfPeaks[idx], outPeaks, timeOfPeaks, h);
	return;
}



__global__ void peakFinderCUDAForCalculationOfPeriodicityByOstrovsky(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, bool* flags, double ostrovskyThreshold)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		flags[idx * 5 + 3] = true;
		return;
	}

	double lastPoint = data[idx * sizeOfBlock + sizeOfBlock - 1];

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks);

	//FIRST CONDITION
	flags[idx * 5 + 0] = true;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] - outPeaks[i - 1] > 0)
		{
			flags[idx * 5 + 0] = false;
			break;
		}
	}

	//SECOND & THIRD CONDITION
	bool flagOne = false;
	bool flagZero = false;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] > ostrovskyThreshold)
			flagOne = true;
		else
			flagZero = true;
		if (flagOne && flagZero)
			break;
	}

	if (flagOne && flagZero)
		flags[idx * 5 + 1] = true;
	else
		flags[idx * 5 + 1] = false;

	if (flagOne && !flagZero)
		flags[idx * 5 + 2] = false;
	else
		flags[idx * 5 + 2] = true;

	//FOUR CONDITION
	if (amountOfPeaks[idx] == 0 || amountOfPeaks[idx] == 1)
		flags[idx * 5 + 3] = true;
	else
		flags[idx * 5 + 3] = false;

	//FIVE CONDITION
	if (lastPoint > ostrovskyThreshold)
		flags[idx * 5 + 4] = true;
	else
		flags[idx * 5 + 4] = false;
	return;
}



__device__ __host__ int kde(double* data, const int startDataIndex, const int amountOfPoints,
	int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH)
{
	if (amountOfPoints == 0)
		return 0;
	if (amountOfPoints == 1 || amountOfPoints == 2)
		return 1;
	if (amountOfPoints > maxAmountOfPeaks)
		return maxAmountOfPeaks;

	double k1 = kdeSampling * amountOfPoints;
	double k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1);
	double delt = 0;
	double prevPrevData2 = 0;
	double prevData2 = 0;
	double data2 = 0;
	bool strangePeak = false;
	int resultKde = 0;

	for (int w = 0; w < k1 - 1; ++w)
	{
		delt = w * k2 + kdeSamplesInterval1;
		prevPrevData2 = prevData2;
		prevData2 = data2;
		data2 = 0;
		for (int m = 0; m < amountOfPoints; ++m)
		{
			double tempData = (data[startDataIndex + m] - delt) / kdeSmoothH;
			data2 += expf(-((tempData * tempData) / 2));
		}

		if (w < 2)
			continue;
		if (strangePeak)
		{
			if (prevData2 == data2)
				continue;
			else if (prevData2 < data2)
			{
				strangePeak = false;
				continue;
			}
			else if (prevData2 > data2)
			{
				strangePeak = false;
				++resultKde;
				continue;
			}
		}
		else if (prevData2 > prevPrevData2 && prevData2 > data2)
		{
			++resultKde;
			continue;
		}
		else if (prevData2 > prevPrevData2 && prevData2 == data2)
		{
			strangePeak = true;
			continue;
		}
	}
	if (prevData2 < data2)
	{
		++resultKde;
	}
	return resultKde;
}



__global__ void kdeCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, int* kdeResult, int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		kdeResult[idx] = 0;
		return;
	}
	kdeResult[idx] = kde(data, idx * sizeOfBlock, amountOfPeaks[idx], maxAmountOfPeaks,
		kdeSampling, kdeSamplesInterval1, kdeSamplesInterval2, kdeSmoothH);
}


// ------------------------------------------------
// --- Вычисляет расстояние между двумя точками ---
// ------------------------------------------------

__device__ __host__ double distance(double x1, double y1, double x2, double y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;
	double dx = x2 - x1;
	double dy = y2 - y1;

	return hypotf(dx, dy);
}



// ----------------------
// --- Функция DBSCAN ---
// ----------------------

__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData)
{
	// ------------------------------------------------------------
	// --- Если пиков 0 или 1 - даже не обрабатываем эти случаи ---
	// ------------------------------------------------------------

	if (amountOfPeaks <= 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	// ------------------------------------------------------------

	int cluster = 0;
	int NumNeibor = 0;

	for (int i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	for (int i = 0; i < amountOfPeaks; i++)
		if (NumNeibor >= 1)
		{
			i = helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1];
			helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1] = 0;
			NumNeibor = NumNeibor - 1;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
			}
		}
		else if (helpfulArray[startDataIndex + i] == 0) {
			NumNeibor = 0;
			++cluster;
			helpfulArray[startDataIndex + i] = cluster;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
			}
		}

	return cluster - 1;
}



// ---------------------------------
// --- Глобальная функция DBSCAN ---
// ---------------------------------

__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const double eps, int* outData)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == -1)
	{
		outData[idx] = 0;
		return;
	}

	// --- Применяем алгоритм dbscan к каждой системе
	outData[idx] = dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, idx, eps, outData);
}



// --------------------
// --- Ядро для LLE ---
// --------------------
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
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			resultArray)
{
	extern __shared__ double s[];
	double* x = s + threadIdx.x * amountOfInitialConditions;
	double* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	//printf("%f %f %f %f\n", localValues[0], localValues[1], localValues[2], localValues[3]);

	double zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		double tempData = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l)
			tempData += (x[l] - y[l]) * (x[l] - y[l]);
		tempData = sqrt(tempData) / eps;

		result += log(tempData);
		
		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (double)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



// -------------------------
// --- Ядро для LLE (IC) ---
// -------------------------
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
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			resultArray)
{
	extern __shared__ double s[];
	double* x = s + threadIdx.x * amountOfInitialConditions;
	double* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	//printf("%f %f %f %f\n", localValues[0], localValues[1], localValues[2], localValues[3]);

	double zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		// z[i] = sinf(0.2171828 * (i + 1) + idx + (0.2171828 + i * idx)) * 0.5;
		z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		double tempData = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l)
			tempData += (x[l] - y[l]) * (x[l] - y[l]);
		tempData = sqrt(tempData) / eps;

		result += log(tempData);

		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (double)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



//find projection operation (ab)
__device__ __host__ void projectionOperator(double* a, double* b, double* minuend, int amountOfValues)
{
	double numerator = 0;
	double denominator = 0;
	for (int i = 0; i < amountOfValues; ++i)
	{
		numerator += a[i] * b[i];
		denominator += b[i] * b[i];
	}

	double fraction = denominator == 0 ? 0 : numerator / denominator;

	for (int i = 0; i < amountOfValues; ++i)
		minuend[i] -= fraction * b[i];
}



__device__ __host__ void gramSchmidtProcess(double* a, double* b, int amountOfVectorsAndValuesInVector, double* denominators=nullptr/*They are is equale for our task*/)
{
	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[j + i * amountOfVectorsAndValuesInVector] = a[j + i * amountOfVectorsAndValuesInVector];

		for (int j = 0; j < i; ++j)
			projectionOperator(a + i * amountOfVectorsAndValuesInVector,
				b + j * amountOfVectorsAndValuesInVector,
				b + i * amountOfVectorsAndValuesInVector,
				amountOfVectorsAndValuesInVector);
	}

	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		double denominator = 0;
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			denominator += b[i * amountOfVectorsAndValuesInVector + j] * b[i * amountOfVectorsAndValuesInVector + j];
		denominator = sqrt(denominator);
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[i * amountOfVectorsAndValuesInVector + j] = denominator == 0 ? 0 : b[i * amountOfVectorsAndValuesInVector + j] / denominator;

		if (denominators != nullptr)
			denominators[i] = denominator;
	}
}



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
	const int preScaller,
	const int writableVar,
	const double maxValue,
	double* resultArray)
{
	extern __shared__ double s[];

	unsigned long long buferForMem = 0;
	double* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	double* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	double* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	double* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	double* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	double* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		double zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			z[j * amountOfInitialConditions + i] = sinf(0.2171828 * (i + 1) * (j + 1) + idx + (0.2171828 + i * j * idx)) * 0.5;//0.5 * (sinf(idx * ((1 + i + j) * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating


	gramSchmidtProcess(z, y, amountOfInitialConditions);


	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m ) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error;*/ }

		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel(y + j * amountOfInitialConditions, localValues, h, amountOfNTPoints,
				amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
			if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error; */ }
		}

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		// 
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (double)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
}