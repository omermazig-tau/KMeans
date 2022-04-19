#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.c"


PyObject* getFlattenMatrixFromMatrix(double ** matrix, unsigned int rows, unsigned int cols) {
    unsigned int i;
    unsigned int j;

    PyObject* newFlattenMatrix = PyTuple_New(rows*cols);
    if(newFlattenMatrix == NULL)
        return NULL;

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            PyTuple_SetItem(newFlattenMatrix, j + i*cols, PyFloat_FromDouble(matrix[i][j]));
        }
    }

    return newFlattenMatrix;
}

double ** getMatrixFromFlattenMatrix(PyObject* flattenMatrix, unsigned int rows, unsigned int cols) {
    unsigned int i;
    unsigned int j;
    double **matrix = createMatrix(rows, cols);

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            matrix[i][j] = PyFloat_AsDouble(PyTuple_GetItem(flattenMatrix, j + i*cols));
        }
    }

    return matrix;
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

    double **centroids = getMatrixFromFlattenMatrix(flattenCentroids, k, cols);
    double **dataPoints = getMatrixFromFlattenMatrix(flattenDataPoints, rows, cols);

    get_new_centroids(iterations, rows, cols, k, epsilon, dataPoints, centroids);

    PyObject* newFlattenCentroids = getFlattenMatrixFromMatrix(centroids, k, cols);

    freeMatrixMemory(centroids, k);
    freeMatrixMemory(dataPoints, rows);

    return newFlattenCentroids;
}

static PyMethodDef _methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, PyDoc_STR("Don't know yet")},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeans_api",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_spkmeans_api(void)
{
    return PyModule_Create(&_moduledef);
}