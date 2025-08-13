#ifndef ALLOCATEPOINTERSLIB_H
#define ALLOCATEPOINTERSLIB_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

int *ivector(int);
double *dvector(int);
std::string *svector(int);

int **imatrix(int, int);
double **dmatrix(int, int);

int ***iblock(int, int, int);
double ***dblock(int, int, int);

#endif