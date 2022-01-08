#define TRUE 1
#define FALSE 0
#define PY_SSIZE_T_CLEAN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>

int isNumber(char str[]);
double **createMatrix(unsigned int rows, unsigned int cols);
double getDistance(double * point1, double * point2, unsigned int dimNum);
void copyArrayIntoArray(double ** arrayToChange, double ** arrayToCopy, unsigned int rows, unsigned int cols);
void freeMatrixMemory(double ** matrixToFree, unsigned int rows);
double **initialize_centroids(int rows, int cols, int k, FILE * f, double **dataPoints);
static void get_new_centroids(unsigned int iterations, unsigned int rows, unsigned int cols, unsigned int k, double epsilon, double **dataPoints,double **centroids);

double ** createMatrix(unsigned int rows, unsigned int cols) {
    unsigned int i;
    double **array = (double **) malloc(rows * sizeof(double *));
    for(i = 0; i < rows; i++) {
        array[i] = (double *) malloc(cols * sizeof(*(array[i])));
    }
    return array;
}

void freeMatrixMemory(double ** matrixToFree, unsigned int rows){
    unsigned int i;
    for (i = 0; i < rows; i++) {
        free(matrixToFree[i]);
    }
    free(matrixToFree);
}

double getDistance(double * point1, double * point2, unsigned int dimNum) {
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

int isNumber(char str[]) {
    unsigned int i;
    for(i = 0; i < strlen(str); i++) {
        if(str[i] < '0' || str[i] > '9' ) {
            return FALSE;
        }
    }
    return TRUE;
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

double ** initialize_centroids(int rows, int cols, int k, FILE *f, double **dataPoints) {
    int i;
    int j;
    double **centroids = createMatrix(k, cols);
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            fscanf(f, "%lf%*c", &dataPoints[i][j]);
            if(i < k) {
                centroids[i][j] = dataPoints[i][j];
            }
        }
    }
    return centroids;
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
    double **newCentroids = createMatrix(k, cols);

    while (epsilonCondition == TRUE && currentIteration < iterations) {
        epsilonCondition = FALSE;
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
        for (i = 0; i < k; i++) {
            centroidsLengths[i] = 0;
            for (j = 0; j < cols; j++) {
                newCentroids[i][j] = 0.0;
            }
        }
    }
    free(centroidsLengths);
    freeMatrixMemory(newCentroids, k);
}

int main(int argc, char *argv[]) {
    unsigned int k;
    char *input_file;
    char *output_file;
    double **dataPoints;
    double **centroids;
    unsigned int i;
    unsigned int j;
    FILE *f;
    double epsilon = 0.001;
    char *strK = NULL;
    char *strIter = NULL;
    unsigned int iterations = 200;
    unsigned int cols = 0;
    unsigned int rows = 1;
    char c = '0';


    if (argc < 4 || argc > 5) {
        printf("Invalid Input!");
        return 1;
    }
    strK = argv[1];
    strIter = argv[2];
    if (argc == 5) {
        if(isNumber(strIter) == FALSE) {
            printf("Invalid Input!");
            return 1;
        }
        iterations = atoi(argv[2]);
        input_file = argv[3];
        output_file = argv[4];
    } else {
        input_file = argv[2];
        output_file = argv[3];
    }
    if(isNumber(strK) == FALSE) {
        printf("Invalid Input!");
        return 1;
    }
    k = atoi(argv[1]);

    f = fopen(input_file, "r");
    if(f) {
        while(c != '\n') {
            fscanf(f, "%*f%c", &c);
            cols++;
        }
        while(fscanf(f, "%*s\n") != EOF) {
            rows++;
        }
        rewind(f);

        if (rows < k) {
            printf("An Error Has Occurred");
            return 1;
        }

        dataPoints = createMatrix(rows, cols);
        centroids = initialize_centroids(rows, cols, k, f, dataPoints);

        fclose(f);
    }
    else {
        printf("An Error Has Occurred");
        return 1;
    }

    if(k > 0) {
        get_new_centroids(iterations, rows, cols, k, epsilon, dataPoints, centroids);
    }
    f = fopen(output_file, "w");
    if(f) {
        for (i = 0; i < k; i++) {
            for (j = 0; j < cols - 1; j++) {
                fprintf(f, "%.4f%c", centroids[i][j], ',');
            }
            fprintf(f, "%.4f", centroids[i][cols - 1]);
            fprintf(f, "%c", '\n');
        }
        fclose(f);
    }
    freeMatrixMemory(dataPoints, rows);
    freeMatrixMemory(centroids, k);
    return 0;
}

static PyObject* fit(PyObject *self, PyObject *args)
{
    unsigned int iterations;
    unsigned int rows;
    unsigned int cols;
    unsigned int k;
    double epsilon;
    PyObject* flattenCentroids;
    PyObject* flattenDataPoints;

    unsigned int i;
    unsigned int j;

    if (!PyArg_ParseTuple(args, "iiiidOO", &iterations, &rows, &cols, &k, &epsilon, &flattenCentroids, &flattenDataPoints))
        return NULL;

    double **centroids = createMatrix(k, cols);
    double **dataPoints = createMatrix(rows, cols);

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            dataPoints[i][j] = PyFloat_AsDouble(PyTuple_GetItem(flattenDataPoints, j + i*cols));
        }
    }

    for(i = 0; i < k; i++) {
        for(j = 0; j < cols; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyTuple_GetItem(flattenCentroids, j + i*cols));
        }
    }

    get_new_centroids(iterations, rows, cols, k, epsilon, dataPoints, centroids);

    PyObject* newFlattenCentroids = PyTuple_New(k*cols);
    if(newFlattenCentroids == NULL)
        return NULL;

    for(i = 0; i < k; i++) {
        for(j = 0; j < cols; j++) {
            PyTuple_SetItem(newFlattenCentroids, j + i*cols, PyFloat_FromDouble(centroids[i][j]));
        }
    }

    freeMatrixMemory(centroids, k);
    freeMatrixMemory(dataPoints, rows);

    return newFlattenCentroids;
}

static PyMethodDef _methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, PyDoc_STR("C Api to calculate centroids")},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}