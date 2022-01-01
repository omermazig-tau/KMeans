#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.c"


static PyObject* get_new_centroids_api(PyObject *self, PyObject *args)
{
    long iterations;
    int rows;
    int cols;
    int k;
    double epsilon;
    
    int i;
    int j;

    PyObject* params_list;

    if (!PyArg_ParseTuple(args, "O", &params_list))
        return NULL;

    iterations = PyLong_AsLong(PyList_GetItem(params_list, 0));
    rows = PyLong_AsLong(PyList_GetItem(params_list, 1));
    cols = PyLong_AsLong(PyList_GetItem(params_list, 2));
    k = PyLong_AsLong(PyList_GetItem(params_list, 3));
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

    PyObject* my_list = PyList_New(0);
    for(i = 0; i < k; i++) {
        PyObject* temp_list = PyTuple_New(cols);
        for(j = 0; j < cols; j++) {
            PyTuple_SetItem(temp_list, j, Py_BuildValue("d", centroids[i][j]));
        }
        PyList_Append(my_list, temp_list);
    }

    freeMatrixMemory(centroids, k);
    freeMatrixMemory(dataPoints, rows);

    return my_list;
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