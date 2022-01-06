#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.c"


static PyObject* get_new_centroids_api(PyObject *self, PyObject *args)
{
    unsigned int iterations;
    unsigned int rows;
    unsigned int cols;
    unsigned int k;
    double epsilon;
    
    unsigned int i;
    unsigned int j;

    PyObject* params_list;

    if (!PyArg_ParseTuple(args, "O", &params_list))
        return NULL;

    iterations = (unsigned int) PyLong_AsLong(PyList_GetItem(params_list, 0));
    rows = (unsigned int) PyLong_AsLong(PyList_GetItem(params_list, 1));
    cols = (unsigned int) PyLong_AsLong(PyList_GetItem(params_list, 2));
    k = (unsigned int) PyLong_AsLong(PyList_GetItem(params_list, 3));
    epsilon = PyFloat_AsDouble(PyList_GetItem(params_list, 4));
    PyObject* flattenCentroids = PyList_GetItem(params_list, 5);
    PyObject* flattenDataPoints = PyList_GetItem(params_list, 6);

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

    PyObject* newFlattenCentroids = PyList_New(k*cols);
    for(i = 0; i < k; i++) {
        for(j = 0; j < cols; j++) {
            PyList_SetItem(newFlattenCentroids, j + i*cols, Py_BuildValue("d", centroids[i][j]));
        }
    }

    freeMatrixMemory(centroids, k);
    freeMatrixMemory(dataPoints, rows);

    return newFlattenCentroids;
}

static PyMethodDef _methods[] = {
    {"get_new_centroids_api", (PyCFunction)get_new_centroids_api, METH_VARARGS, PyDoc_STR("C Api to calculate centroids")},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "kmeans_c_api",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_kmeans_c_api(void)
{
    return PyModule_Create(&_moduledef);
}