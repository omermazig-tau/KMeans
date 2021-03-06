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


    //wam
    if (strcmp(goal, goalOptions[0]) == 0) {
        mat1 = getWeightAdjacency(x, shape[0], shape[1]);
        printMat(mat1, shape[0], shape[0]);
        freeMatrixMemory(mat1, shape[0]);
        freeMatrixMemory(x, shape[0]);
        free(shape);
        return 0;
    }
    //ddg
    else if (strcmp(goal, goalOptions[1]) == 0) {
        mat1 = getWeightAdjacency(x, shape[0], shape[1]);
        mat2 = getDiagonalDegreeMat(mat1, shape[0]);
        printMat(mat2, shape[0], shape[0]);
        freeMatrixMemory(mat1, shape[0]);
        freeMatrixMemory(mat2, shape[0]);
        freeMatrixMemory(x, shape[0]);
        free(shape);
        return 0;
    }
    //lnorm
    else if (strcmp(goal, goalOptions[2]) == 0) {
        mat1 = getWeightAdjacency(x, shape[0], shape[1]);
        mat2 = getDiagonalDegreeMat(mat1, shape[0]);
        mat3 = getNormalizedGraphLaplacian(mat1, mat2, shape[0]);
        printMat(mat3, shape[0], shape[0]);
        freeMatrixMemory(mat1, shape[0]);
        freeMatrixMemory(mat2, shape[0]);
        freeMatrixMemory(mat3, shape[0]);
        freeMatrixMemory(x, shape[0]);
        free(shape);
        return 0;
    }
    //jacobi
    else if (strcmp(goal, goalOptions[3]) == 0) {
        if (!checkMatSymmetric(x, shape[0], shape[1])) { //Not symmetric of squared
            printf(INPUT_ERR);
            return 1;
        }
        mat1 = jacobiAlgorithm(x, shape[0]);
        printArrNoMinusZeros(mat1[0], shape[0]);
        printMat(mat1 + 1, shape[0], shape[0]);
        freeMatrixMemory(mat1, shape[0] + 1);
        freeMatrixMemory(x, shape[0]);
        free(shape);
        return 0;
    }
    else {
        printf(INPUT_ERR);
        return 1;
    }
}

double ** getWeightAdjacency(double ** x, unsigned int n, unsigned int d) {
    unsigned int i, j;
    double ** weights;

    weights = createZeroMatrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                weights[i][j] = exp(-getDistance(x[i], x[j], d) / 2);
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
    freeMatrixMemory(mat1, n);
    freeMatrixMemory(mat2, n);
    freeMatrixMemory(mat3, n);
    freeMatrixMemory(powMinusHalfD, n);
    return NormalizedGraphLaplacian;
}

double ** jacobiAlgorithm(double ** mat, unsigned int n) {
    double ** vMat, **pMat, **newA, **oldA, *eigenValues, **eigenVectors, **returnedMat, **matForMulti, **transP;
    double sumOld, sumNew;
    unsigned int iter;

    if(isDiagonal(mat, n)) {
        eigenValues = getDiagSquaredMatrix(mat, n);
        eigenVectors = getIdentityMat(n);
        returnedMat = addVectorFirstLineMatrix(eigenVectors, eigenValues, n, n);

        free(eigenValues);
        freeMatrixMemory(eigenVectors, n);
        return returnedMat;
    }

    newA = createCopyMat(mat, n, n);
    vMat = getIdentityMat(n);
    sumNew = getSumSquaredOffDiagElement(newA, n);
    iter = 0;

    do {
        sumOld = sumNew;
        oldA = newA;
        pMat = createMatrixP(oldA, n);
        matForMulti = multiSquaredMatrices(vMat, pMat, n);
        freeMatrixMemory(vMat, n);
        vMat = matForMulti;
        transP = transformSquaredMatrix(pMat, n);
        matForMulti = multiSquaredMatrices(transP, oldA, n);
        newA = multiSquaredMatrices(matForMulti, pMat, n);

        freeMatrixMemory(matForMulti, n);
        freeMatrixMemory(oldA, n);
        freeMatrixMemory(transP, n);
        freeMatrixMemory(pMat, n);

        sumNew = getSumSquaredOffDiagElement(newA, n);
        iter++;
    }
    while ((sumOld - sumNew > EPSILON) && iter < MAX_NUM_ITER);

    eigenValues = getDiagSquaredMatrix(newA, n);
    freeMatrixMemory(newA, n);
    eigenVectors = vMat;
    returnedMat = addVectorFirstLineMatrix(eigenVectors, eigenValues, n, n);

    free(eigenValues);
    freeMatrixMemory(eigenVectors, n);
    return returnedMat;
}



// Helpful methods

double ** createCopyMat(double ** mat, unsigned int rows, unsigned int cols) {
    double ** copyMat;

    copyMat = createZeroMatrix(rows, cols);
    copyArrayIntoArray(copyMat, mat, rows, cols);
    return copyMat;
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
    if(!indexMax) {
        printf(NOT_INPUT_ERR);
        exit(1);
    }
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
            if (i < j && fabs(mat[i][j]) > max) {
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
    t = getSign(theta) / (fabs(theta) + sqrt(pow(theta, 2) + 1));
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
    unsigned int i, maxI;
    double max, possibleMax;

    unsigned int *indices = getSortedIndex(eigenValues, n);

    max = eigenValues[indices[1]] - eigenValues[indices[0]];
    maxI = 0;

    for (i = 1; i <= (n / 2); i++) {
        possibleMax = eigenValues[indices[i+1]] - eigenValues[indices[i]];
        if (possibleMax > max) {
            max = possibleMax;
            maxI = i;
        }
    }
    return maxI + 1;
}


double ** getKFirstEigenvectors(double * eigenValues, double ** eigenVectors, unsigned int n, unsigned int k) {
    unsigned int * indices, i, j;
    double **firstKEigenVectors;

    indices = getSortedIndex(eigenValues, n);
    firstKEigenVectors = createZeroMatrix(n, k);
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            firstKEigenVectors[i][j] = eigenVectors[i][indices[j]];
        }
    }
    free(indices);
    return firstKEigenVectors;
}


double ** calcTMat(double ** uMat, unsigned int rows, unsigned int cols) {
    double ** tMat, sum, divider;
    unsigned int i, j;

    sum = 0;
    tMat = createZeroMatrix(rows, cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            sum += pow(uMat[i][j], 2);
        }
        divider = sqrt(sum);
        for (j = 0; j < cols; j++) {
            if (divider == 0)
                tMat[i][j] = 0.0;
            else
                tMat[i][j] = uMat[i][j] / divider;
        }
        sum = 0;
    }
    return tMat;
}


unsigned int * getShapeMatrixFile(FILE * f) {
    unsigned int *shape, doneCol;
    char c;

    shape = (unsigned int *)malloc(2 * sizeof(unsigned int));
    if(!shape) {
        printf(NOT_INPUT_ERR);
        exit(1);
    }
    doneCol = 0;
    c = '0';
    shape[0] = 0;
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


void printArrNoMinusZeros(double * arr, unsigned int len) {
    unsigned int i;

    for (i = 0; i < len; i++) {
        if (arr[i] > -0.0001 && arr[i] < 0)  {
            printf("%.4f", 0.0);
        }
        else {
            printf("%.4f", arr[i]);
        }
        if (i + 1 != len) {
            printf(",");
        }
    }
    printf("\n");
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

unsigned int checkMatSymmetric(double ** mat, unsigned int rows, unsigned int cols) {
    unsigned int i, j;

    if(rows != cols)
        return FALSE;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (mat[i][j] != mat[j][i]) {
                return FALSE;
            }
        }
    }
    return TRUE;
}


int getSign(double num) {
    if(num >= 0) {
        return 1;
    }
    return -1;
}


int cmp(const void *a, const void *b)
{
    struct Pair *a1 = (struct Pair *)a;
    struct Pair *a2 = (struct Pair *)b;
    if ((*a1).value > (*a2).value)
        return 1;
    else if ((*a1).value < (*a2).value)
        return -1;
    else
        return 0;
}


unsigned int * getSortedIndex(const double * arr, unsigned len) {
    Pair *pairArr;
    unsigned int * sortedIndices;
    unsigned int i;

    pairArr = malloc(sizeof(*pairArr) * len);
    sortedIndices = malloc(sizeof(unsigned int) * len);

    if(!pairArr || !sortedIndices) {
        printf(NOT_INPUT_ERR);
        exit(1);
    }
    for (i = 0; i < len ; i++) {
        pairArr[i].value = arr[i];
        pairArr[i].index = i;
    }
    qsort(pairArr, len, sizeof(pairArr[0]), cmp);
    for (i = 0; i < len; i++) {
        sortedIndices[i] = pairArr[i].index;
    }
    free(pairArr);
    return sortedIndices;
}


