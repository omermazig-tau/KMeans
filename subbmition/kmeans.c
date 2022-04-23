#include "kmeans.h"

double ** createZeroMatrix(unsigned int rows, unsigned int cols) {
    unsigned int i;
    double ** mat;

    mat = (double **)malloc(sizeof(double *) * rows);
    if(!mat) {
        printf(NOT_INPUT_ERR);
        exit(1);
    }
    for (i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if(!mat[i]) {
            printf(NOT_INPUT_ERR);
            exit(1);
        }
    }
    return mat;
}

void freeMatrixMemory(double ** matrixToFree, unsigned int rows){
    unsigned int i;
    for (i = 0; i < rows; i++) {
        free(matrixToFree[i]);
    }
    free(matrixToFree);
}

double getDistance(const double * point1, const double * point2, unsigned int dimNum) {
    unsigned int i;
    double sum;
    double distance;
    sum = 0;
    for(i = 0; i < dimNum; i++) {
        sum += (point1[i] - point2[i])*(point1[i] - point2[i]);
    }
    distance = sqrt(sum);
    return distance;
}

void copyArrayIntoArray(double **arrayToChange, double **arrayToCopy, unsigned int rows, unsigned int cols) {
    unsigned int i;
    unsigned int j;
    for (i=0;i<rows;i++) {
        for (j = 0; j < cols; j++) {
            arrayToChange[i][j] = arrayToCopy[i][j];
        }
    }
}

void get_new_centroids(unsigned int iterations, unsigned int rows, unsigned int cols, unsigned int k, double epsilon, double **dataPoints,
                            double **centroids) {
    unsigned int i;
    unsigned int j;
    unsigned int m;
    double minDistance;
    double tempDistance;
    unsigned int closestCentroid;

    unsigned int epsilonCondition = TRUE;
    unsigned int currentIteration = 0;
    unsigned int *centroidsLengths = calloc(k, sizeof(int));
    double **newCentroids = createZeroMatrix(k, cols);

    while (epsilonCondition == TRUE && currentIteration < iterations) {
        epsilonCondition = FALSE;

        for (i = 0; i < k; i++) {
            centroidsLengths[i] = 0;
            for (j = 0; j < cols; j++) {
                newCentroids[i][j] = 0.0;
            }
        }

        for (i = 0; i < rows; i++) {
            minDistance = getDistance(dataPoints[i], centroids[0], cols);
            closestCentroid = 0;
            for (j = 1; j < k; j++) {
                tempDistance = getDistance(dataPoints[i], centroids[j], cols);
                if (tempDistance < minDistance) {
                    minDistance = tempDistance;
                    closestCentroid = j;
                }
            }
            centroidsLengths[closestCentroid]++;
            for (m = 0; m < cols; m++) {
                newCentroids[closestCentroid][m] += dataPoints[i][m];
            }
        }
        for (i = 0; i < k; i++) {
            for (j = 0; j < cols; j++) {
                newCentroids[i][j] /= (double) centroidsLengths[i];
            }
            if (getDistance(centroids[i], newCentroids[i], cols) >= epsilon) {
                epsilonCondition = TRUE;
            }
        }
        copyArrayIntoArray(centroids, newCentroids, k, cols);
        currentIteration++;
    }
    free(centroidsLengths);
    freeMatrixMemory(newCentroids, k);
}