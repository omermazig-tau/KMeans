//
// Created by roydd on 4/15/2022.
//

#ifndef CPART_SPKMEANS_H
#define CPART_SPKMEANS_H

#define MAX_NUM_ITER 100
#define EPSILON (pow(10, -5))

#include "kmeans.h"


typedef struct Pair {
    unsigned int index;
    double value;
} Pair;
double ** getWeightAdjacency(double ** x, unsigned int n, unsigned int d);
double ** getDiagonalDegreeMat(double ** weights, unsigned int n);
double ** getNormalizedGraphLaplacian(double ** weights, double ** diagDegreeMat, unsigned int n);
double ** jacobiAlgorithm(double ** mat, unsigned int n);
double ** createCopyMat(double ** mat, unsigned int rows, unsigned int cols);
double ** subtractSquaredMatrices(double ** mat1, double ** mat2, unsigned int n);
double ** multiSquaredMatrices(double ** mat1, double ** mat2, unsigned int n);
double ** getPowMinusHalfDiagMat(double ** mat, unsigned int n);
double ** getIdentityMat(unsigned n);
unsigned int * getIndexesValOffDiagSquaredMat(double ** mat, unsigned int n);
double ** createMatrixP(double ** mat, unsigned int n);
double getSumSquaredOffDiagElement(double ** mat, unsigned int n);
double ** transformSquaredMatrix(double ** mat, unsigned int n);
double * getDiagSquaredMatrix(double ** mat, unsigned int n);
double ** addVectorFirstLineMatrix(double ** mat, const double * vector, unsigned int rowsMat, unsigned int cols);
unsigned int determineK(double * eigenValues, unsigned int n);
double ** getKFirstEigenvectors(double * eigenValues, double ** eigenVectors, unsigned int n, unsigned int k);
double ** calcTMat(double ** uMat, unsigned int rows, unsigned int cols);
unsigned int * getShapeMatrixFile(FILE * f);
double ** createMatFromFile(FILE * f, const unsigned int * shape);
void printMat(double ** mat, unsigned int rows, unsigned int cols);
unsigned int isDiagonal(double ** mat, unsigned int n);

unsigned int checkMatSymmetric(double ** mat, unsigned int rows, unsigned int cols);
int getSign(double num);
int cmp(const void *a, const void *b);
unsigned int * getSortedIndex(const double * arr, unsigned len);
void printArrNoMinusZeros(double * arr, unsigned int len);
#endif //CPART_SPKMEANS_H
