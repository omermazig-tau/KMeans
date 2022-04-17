//
// Created by roydd on 4/13/2022.
//
#include "spkmeans.h"

//Core methods
int main(int argc, char ** argv) {
    char *inputFile, *goal;
    unsigned int * shape;
    double **x, **mat1, **mat2, **mat3;
    FILE *f;
    char goalOptions[4][10] = {"wam", "ddg", "lnorm", "jacobi"};

    if (argc != 3) {
        printf(INPUT_ERR);
        return 1;
    }
    inputFile = argv[2];
    f = fopen(inputFile, "r");
    if (!f) {
        printf(NOT_INPUT_ERR);
        return 1;
    }
    shape = getShapeMatrixFile(f);
    x = createMatFromFile(f, shape);
    fclose(f);
    goal = argv[1];
    spk(x, shape[0], shape[1], 2);


    if (strcmp(goal, goalOptions[0]) == 0) {
        mat1 = getWeightAdjacency(x, shape[0], shape[1]);
        printMat(mat1, shape[0], shape[0]);
        freeMat(mat1, shape[0]);
        return 0;
    }
    if (strcmp(goal, goalOptions[1]) == 0) {
        mat1 = getWeightAdjacency(x, shape[0], shape[1]);
        mat2 = getDiagonalDegreeMat(mat1, shape[0]);
        printMat(mat2, shape[0], shape[0]);
        freeMat(mat1, shape[0]);
        freeMat(mat2, shape[0]);
        return 0;
    }
    if (strcmp(goal, goalOptions[2]) == 0) {
        mat1 = getWeightAdjacency(x, shape[0], shape[1]);
        mat2 = getDiagonalDegreeMat(mat1, shape[0]);
        mat3 = getNormalizedGraphLaplacian(mat1, mat2, shape[0]);
        printMat(mat3, shape[0], shape[0]);
        freeMat(mat1, shape[0]);
        freeMat(mat2, shape[0]);
        freeMat(mat3, shape[0]);
        return 0;
    }
    if (strcmp(goal, goalOptions[3]) == 0) {
        mat1 = jacobiAlgorithm(x, shape[0]);
        printMat(mat1, shape[0] + 1, shape[0]);
        freeMat(mat1, shape[0]);
        return 0;
    }
    printf(INPUT_ERR);
    return 1;
}

double ** spk (double ** x, unsigned int rows, unsigned int cols, unsigned int k) {
    double **mat1, **mat2, **mat3, **mat4, **mat5, **tMat;

    mat1 = getWeightAdjacency(x, rows, cols);
    mat2 = getDiagonalDegreeMat(mat1, rows);
    mat3 = getNormalizedGraphLaplacian(mat1, mat2, rows);
    mat4 = jacobiAlgorithm(mat3, rows);

    if (k == 0) {
        mat5 = mat4;
        k = determineK(mat4[0], rows);
        mat4 = getKFirstEigenvectors(mat4, rows, k);
        freeMat(mat5, rows);
    }
    freeMat(mat1, rows);
    freeMat(mat2, rows);
    freeMat(mat3, rows);
    freeMat(mat4, rows);

    tMat =  calcTMat(mat4+1, rows, k);
    printf("T\n");
    printMat(tMat, rows, k);
    printf("\n");
    return tMat;
}


double ** getWeightAdjacency(double ** x, unsigned int n, unsigned int d) {
    unsigned int i, j;
    double ** weights;

    weights = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                weights[i][j] = exp(-getDistanceVectors(x[i], x[j], d) / 2);
            }
        }
    }
    return weights;
}

double ** getDiagonalDegreeMat(double ** weights, unsigned int n) {
    double ** diagMat, sum;
    unsigned int i, z;

    sum = 0;
    diagMat = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
        for (z = 0; z < n; z++) {
            sum += weights[i][z];
        }
        diagMat[i][i] = sum;
        sum = 0;
    }
    return diagMat;
}


double ** getNormalizedGraphLaplacian(double ** weights, double ** diagDegreeMat, unsigned int n) {
    double **mat1, **mat2, **mat3, **powMinusHalfD, **NormalizedGraphLaplacian;

    mat1 = getIdentityMat(n);
    powMinusHalfD = getPowMinusHalfDiagMat(diagDegreeMat, n);
    mat3 = multiSquaredMatrices(powMinusHalfD, weights, n);
    mat2 = multiSquaredMatrices(mat3, powMinusHalfD, n);
    NormalizedGraphLaplacian = subtractSquaredMatrices(mat1, mat2, n);
    freeMat(mat1, n);
    freeMat(mat2, n);
    freeMat(mat3, n);
    return NormalizedGraphLaplacian;
}

double ** jacobiAlgorithm(double ** mat, unsigned int n) {
    double ** vMat, **pMat, **newA, **oldA, *eigenValues, **eigenVectors, **oldVMat, **returnedMat, **matForMulti, **transP;
    unsigned int iter;

    iter = 0;
    oldA = createCopyMat(mat, n, n);
    pMat = createMatrixP(oldA, n);
    vMat = createCopyMat(pMat, n, n);
    transP = transformSquaredMatrix(pMat, n);
    matForMulti = multiSquaredMatrices(transP, oldA, n);
    newA = multiSquaredMatrices(matForMulti, pMat, n);
    freeMat(transP, n);
    freeMat(matForMulti, n);

    while (!(isConvergenceDiag(newA, oldA, n)) && iter < MAX_NUM_ITER && !(isDiagonal(newA, n))) {
        iter++;
        freeMat(oldA, n);
        oldA = newA;
        freeMat(pMat, n);
        pMat = createMatrixP(oldA, n);

        oldVMat = vMat;
        vMat = multiSquaredMatrices(vMat, pMat, n);
        freeMat(oldVMat, n);

        transP = transformSquaredMatrix(pMat, n);
        matForMulti = multiSquaredMatrices(transP, oldA, n);
        newA = multiSquaredMatrices(matForMulti, pMat, n);
        freeMat(transP, n);
        freeMat(matForMulti, n);
    }

    freeMat(oldA, n);
    eigenValues = getDiagSquaredMatrix(newA, n);
    eigenVectors = vMat;
    returnedMat = addVectorFirstLineMatrix(eigenVectors, eigenValues, n, n);

    free(eigenValues);
    freeMat(eigenVectors, n);
    return returnedMat;
}



// Helpful methods

double ** createZeroMatrix (unsigned int rows, unsigned int cols) {
    unsigned int i;
    double ** mat;

    mat = (double **)malloc(sizeof(double *) * rows);
    assert(mat);
    for (i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        assert(mat[i]);
    }
    return mat;
}

double ** createCopyMat(double ** mat, unsigned int rows, unsigned int cols) {
    unsigned int i, j;
    double ** copyMat;

    copyMat = createZeroMatrix(rows, cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            copyMat[i][j] = mat[i][j];
        }
    }
    return copyMat;
}


double getDistanceVectors(double * arr1, double * arr2, unsigned int length) {
    double sum;
    unsigned int i;

    sum = 0;
    for (i = 0; i < length; i++) {
        sum += pow((arr1[i] - arr2[i]), 2);
    }
    return sqrt(sum);
}

double ** subtractSquaredMatrices(double ** mat1, double ** mat2, unsigned int n) {
    double ** mat;
    unsigned int i, j;

    mat = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return mat;
}

double ** multiSquaredMatrices(double ** mat1, double ** mat2, unsigned int n) {
    double ** mat;
    unsigned int i, j, k;

    mat = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                mat[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return mat;
}


double ** getPowMinusHalfDiagMat(double ** mat, unsigned int n) {
    double **minusSquaredDiagMat;
    unsigned int i;

   minusSquaredDiagMat = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
       minusSquaredDiagMat[i][i] = 1 / sqrt(mat[i][i]);
    }
    return minusSquaredDiagMat;
}

double ** getIdentityMat(unsigned n) {
    double **iMat;
    unsigned int i;

   iMat = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
       iMat[i][i] = 1.0;
    }
    return iMat;
}


unsigned int * getIndexesValOffDiagSquaredMat(double ** mat, unsigned int n) {
    double max;
    unsigned int i, j;
    unsigned int * indexMax;

    max = 0;
    indexMax = (unsigned int *)malloc(2 * sizeof(unsigned int));
    assert(indexMax);
    indexMax[0] = 0;
    indexMax[1] = 1;

    if (n == 1) {
        indexMax[1] = 0;
        return indexMax;
    }
    if (n == 2) {
        max = fabs(mat[0][1]);
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && fabs(mat[i][j]) > max) {
                max = fabs(mat[i][j]);
                indexMax[0] = i;
                indexMax[1] = j;
            }
        }
    }
    return indexMax;
}

double ** createMatrixP(double ** mat, unsigned int n) {
    double ** pMat, s, t, c, theta;
    unsigned int * indexMax, i, j;

    indexMax = getIndexesValOffDiagSquaredMat(mat, n);
    i = indexMax[0];
    j = indexMax[1];
    free(indexMax);

    pMat = getIdentityMat(n);
    theta = (mat[j][j] - mat[i][i]) / (2 * mat[i][j]);
    t = SIGN(theta) / (fabs(theta) + sqrt(pow(theta, 2) + 1));
    c = 1 / sqrt(pow(t, 2) + 1);
    s = t * c;
    pMat[i][j] = s;
    pMat[i][i] = c;
    pMat[j][i] = -s;
    pMat[j][j] = c;
    return pMat;
}

double getSumSquaredOffDiagElement(double ** mat, unsigned int n) {
    double sum;
    unsigned int i, j;

    sum = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                sum += pow(mat[i][j], 2);
            }
        }
    }
    return sum;
}

unsigned int isConvergenceDiag(double ** matNew, double ** matOld, unsigned int n) {
    double sumOld, sumNew;

    sumOld = getSumSquaredOffDiagElement(matOld, n);
    sumNew = getSumSquaredOffDiagElement(matNew, n);
    if (fabs(sumOld - sumNew) <= EPSILON) {
        return TRUE;
    }
    return FALSE;
}

double ** transformSquaredMatrix(double ** mat, unsigned int n) {
    double ** transMat;
    unsigned int i, j;

    transMat = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            transMat[i][j] = mat[j][i];
        }
    }
    return transMat;
}

double * getDiagSquaredMatrix(double ** mat, unsigned int n) {
    unsigned int i;
    double * diag;

    diag = (double *)malloc(n * sizeof(double));
    for (i = 0; i < n; i++) {
        diag[i] = mat[i][i];
    }
    return diag;
}

double ** addVectorFirstLineMatrix(double ** mat, const double * vector, unsigned int rowsMat, unsigned int cols) {
    double ** newMat;
    unsigned int i, j;

    newMat = createZeroMatrix(rowsMat + 1, cols);
    for (j = 0; j < cols; j++) {
        newMat[0][j] = vector[j];
    }
    for (i = 0; i < rowsMat; i++) {
        for (j = 0; j < cols; j++) {
            newMat[i + 1][j] = mat[i][j];
        }
    }
    return newMat;
}



unsigned int determineK(double * eigenValues, unsigned int n) {
    unsigned int limit, i, maxI;
    double max;

    max = fabs(eigenValues[0] - eigenValues[1]);
    maxI = 0;
    limit = n / 2;

    for (i = 1; i <= limit; i++) {
        if (fabs(eigenValues[i] - eigenValues[i + 1]) > max) {
            max = fabs(eigenValues[i] - eigenValues[i + 1]);
            maxI = i;
        }
    }
    return maxI + 1;
}


double ** getKFirstEigenvectors(double ** eigenVectors, unsigned int n, unsigned int k) {
    return createCopyMat(eigenVectors, n, k);
}


double ** calcTMat(double ** uMat, unsigned int rows, unsigned int cols) {
    double ** tMat, sum;
    unsigned int i, j, k;

    sum = 0;
    tMat = createZeroMatrix(rows, cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            for (k = 0; k < cols; k++) {
                sum += pow(uMat[i][k], 2);
            }
            tMat[i][j] = uMat[i][j] / sqrt(sum);
            sum = 0;
        }
    }
    return tMat;
}

unsigned int * getShapeMatrixFile(FILE * f) {
    unsigned int *shape, doneCol;
    char c;

    shape = (unsigned int *)malloc(2 * sizeof(unsigned int));
    assert(shape);
    doneCol = 0;
    c = '0';
    shape[0] = 1;
    shape[1] = 1;

    while (c != EOF) {
        c = (char)fgetc(f);
        if (doneCol == 0 && c == ',') {
            shape[1]++;
        }
        if (c == '\n') {
            shape[0] ++;
            doneCol = 1;
        }
    }
    rewind(f);
    return shape;
}


double ** createMatFromFile(FILE * f, const unsigned int * shape) {
    unsigned int rows, cols, i, j;
    double **mat;

    rows = shape[0];
    cols = shape[1];
    mat = createZeroMatrix(rows, cols);
    for(i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++) {
            fscanf(f, "%lf%*c", &mat[i][j]);
        }
    }
    return mat;
}

void printMat(double ** mat, unsigned int rows, unsigned int cols) {
    unsigned int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", mat[i][j]);
            if (j + 1 != cols) {
                printf(",");
            }
        }
        printf("\n");
    }
}


unsigned int isDiagonal(double ** mat, unsigned int n) {
    unsigned int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && mat[i][j] != 0) {
                return FALSE;
            }
        }
    }
    return TRUE;
}

void freeMat(double ** mat, unsigned int rows){
    unsigned int i;

    for (i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}




