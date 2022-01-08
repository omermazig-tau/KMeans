#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.c"


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