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

static PyObject* get_weight_adjacency_matrix(PyObject *self, PyObject *args)
{
    unsigned int rows;
    unsigned int cols;
    PyObject* flattenMatrix;

    if (!PyArg_ParseTuple(args, "iiO", &rows, &cols, &flattenMatrix))
        return NULL;

    double **matrix = getMatrixFromFlattenMatrix(flattenMatrix, rows, cols);

    matrix = getWeightAdjacency(matrix, rows, cols);

    PyObject* newFlattenMatrix = getFlattenMatrixFromMatrix(matrix, rows, cols);

    freeMatrixMemory(matrix, rows);

    return newFlattenMatrix;
}

static PyObject* get_diagonal_degree_matrix(PyObject *self, PyObject *args)
{
    unsigned int rows;
    unsigned int cols;
    PyObject* flattenMatrix;

    if (!PyArg_ParseTuple(args, "iiO", &rows, &cols, &flattenMatrix))
        return NULL;

    double **matrix = getMatrixFromFlattenMatrix(flattenMatrix, rows, cols);

    matrix = getWeightAdjacency(matrix, rows, cols);

    matrix = getDiagonalDegreeMat(matrix, rows);

    PyObject* newFlattenMatrix = getFlattenMatrixFromMatrix(matrix, rows, cols);

    freeMatrixMemory(matrix, rows);

    return newFlattenMatrix;
}

static PyObject* get_normalized_graph_laplacian(PyObject *self, PyObject *args)
{
    unsigned int rows;
    unsigned int cols;
    PyObject* flattenMatrix;

    if (!PyArg_ParseTuple(args, "iiO", &rows, &cols, &flattenMatrix))
        return NULL;

    double **matrix = getMatrixFromFlattenMatrix(flattenMatrix, rows, cols);
    double **matrix2;

    matrix = getWeightAdjacency(matrix, rows, cols);

    matrix2 = getDiagonalDegreeMat(matrix, rows);

    matrix = getNormalizedGraphLaplacian(matrix, matrix2, rows);

    PyObject* newFlattenMatrix = getFlattenMatrixFromMatrix(matrix, rows, cols);

    freeMatrixMemory(matrix, rows);
    freeMatrixMemory(matrix2, rows);

    return newFlattenMatrix;
}

static PyObject* get_jacobi_matrix(PyObject *self, PyObject *args)
{
    unsigned int n;
    PyObject* flattenMatrix;

    if (!PyArg_ParseTuple(args, "iO", &n, &flattenMatrix))
        return NULL;

    double **matrix = getMatrixFromFlattenMatrix(flattenMatrix, n, n);

    matrix = jacobiAlgorithm(matrix, n);

    PyObject* newFlattenMatrix = getFlattenMatrixFromMatrix(matrix, n, n);

    freeMatrixMemory(matrix, n);

    return newFlattenMatrix;
}

static PyMethodDef _methods[] = {
    {"get_weight_adjacency_matrix", (PyCFunction)get_weight_adjacency_matrix, METH_VARARGS, PyDoc_STR("Getting the Weight Adjacency matrix from a matrix")},
    {"get_diagonal_degree_matrix", (PyCFunction)get_diagonal_degree_matrix, METH_VARARGS, PyDoc_STR("Getting the Diagonal Degree matrix from a matrix")},
    {"get_normalized_graph_laplacian", (PyCFunction)get_normalized_graph_laplacian, METH_VARARGS, PyDoc_STR("Getting the Normalized graph laplacian matrix from a matrix")},
    {"get_jacobi_matrix", (PyCFunction)get_jacobi_matrix, METH_VARARGS, PyDoc_STR("Getting the matrix after jacobi algorithm")},
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