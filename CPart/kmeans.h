//
// Created by User on 4/23/2022.
//

#ifndef CPART_KMEANS_H
#define CPART_KMEANS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define TRUE 1
#define FALSE 0


double **createMatrix(unsigned int rows, unsigned int cols);
double getDistance(const double * point1, const double * point2, unsigned int dimNum);
void copyArrayIntoArray(double ** arrayToChange, double ** arrayToCopy, unsigned int rows, unsigned int cols);
void freeMatrixMemory(double ** matrixToFree, unsigned int rows);

static void get_new_centroids(unsigned int iterations, unsigned int rows, unsigned int cols, unsigned int k, double epsilon, double **dataPoints,double **centroids);

#endif //CPART_KMEANS_H
