#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.c"


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

static PyObject* get_weight_adjacency(PyObject *self, PyObject *args)
{
    unsigned int rows;
    unsigned int cols;
    PyObject* flattenMatrix;

    if (!PyArg_ParseTuple(args, "iiO", &rows, &cols, &flattenMatrix))
        return NULL;

    double **matrix = getMatrixFromFlattenMatrix(flattenMatrix, rows, cols);

    matrix = getWeightAdjacency(matrix, rows, cols);

    PyObject* newFlattenCentroids = getFlattenMatrixFromMatrix(matrix, rows, cols);

    freeMatrixMemory(matrix, rows);

    return newFlattenCentroids;
}

static PyMethodDef _methods[] = {
    {"get_weight_adjacency", (PyCFunction)get_weight_adjacency, METH_VARARGS, PyDoc_STR("Getting the weight adjacency matrix from a matrix")},
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