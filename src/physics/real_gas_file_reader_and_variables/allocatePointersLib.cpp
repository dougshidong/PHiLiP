#include <iostream>
#include "allocatePointersLib.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

using namespace std;
//===============================================
int *ivector(int n)
{
	int *v = new int[n];
	return v;
}
//===============================================
double *dvector(int n)
{
	double *v = new double[n];
	return v;
}
//===============================================
string *svector(int n)
{
	string *v = new string[n];
	return v;
}
//===============================================
int **imatrix(int nRow, int nCol)
{
	int **A = new int*[nRow];
	for(int i=0; i<nRow; i++)
	{
		A[i] = new int[nCol];		
	}
	return A;
}
//===============================================
double **dmatrix(int nRow, int nCol)
{
	double **A = new double*[nRow];
	for(int i=0; i<nRow; i++)
	{
		A[i] = new double[nCol];		
	}
	return A;
}
//===============================================
int ***iblock(int nRow, int nCol, int nMat)
{
	int ***A = new int**[nRow];
	for(int i=0; i<nRow; i++)
	{
		A[i] = new int*[nCol];
		for(int j=0; j<nCol; j++)
		{
			A[i][j] = new int[nMat];
		}
	}
	return A;
}
//===============================================
double ***dblock(int nRow, int nCol, int nMat)
{
	double ***A = new double**[nRow];
	for(int i=0; i<nRow; i++)
	{
		A[i] = new double*[nCol];
		for(int j=0; j<nCol; j++)
		{
			A[i][j] = new double[nMat];
		}
	}
	return A;
}