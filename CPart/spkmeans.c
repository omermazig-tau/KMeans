//
// Created by roydd on 4/13/2022.
//
#include <math.h>
#include <stdio.h>
#include <malloc.h>

//Core methods
double ** get_weight_adjacency(double **, unsigned int);
double ** get_diagonal_degree_mat(double **, unsigned int);

// Helpful methods
double ** create_zero_matrix(unsigned int, unsigned int);
double get_distance_vectors(double *, double *, unsigned int);
double ** get_identity_mat(unsigned int);
double ** get_pow_minus_half_diag_mat(double **, unsigned int);
double ** multi_squared_matrices(double **, double **, unsigned int);
double ** subtract_squared_matrices(double **, double **, unsigned int);


//Core methods

double ** get_weight_adjacency(double ** x, unsigned int n) {
    unsigned int i, j;
    double ** weights;

    weights = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                weights[i][j] = exp(-get_distance_vectors(x[i], x[j], n) / 2);
            }
        }
    }
    return weights;
}

double ** get_diagonal_degree_mat(double ** weights, unsigned int n) {
    double ** diag_mat, sum;
    unsigned int i, z;

    sum = 0;
    diag_mat = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (z = 0; z < n; z++) {
            sum += weights[i][z];
        }
        diag_mat[i][i] = sum;
    }
    return diag_mat;
}


double ** get_normalized_graph_laplacian(double ** weights, double ** diag_degree_mat, unsigned int n) {
    double ** normalized_graph_laplacian, **mat1, **mat2, **pow_minus_half_d;

    mat1 = get_identity_mat(n);
    pow_minus_half_d = get_pow_minus_half_diag_mat(diag_degree_mat, n);
    mat2 = multi_squared_matrices(multi_squared_matrices(pow_minus_half_d, weights, n), pow_minus_half_d, n);
    return subtract_squared_matrices(mat1, mat2, n);
}



// Helpful methods

double ** create_zero_matrix (unsigned int rows, unsigned int cols) {
    unsigned int i, j;
    double ** mat;

    mat = malloc(sizeof(int *) * rows);
    for (i = 0; i < rows; i++) {
        mat[i] = calloc(cols, sizeof(int));
    }
    return mat;
}

double get_distance_vectors(double * arr1, double * arr2, unsigned int length) {
    double sum;
    unsigned int i;

    sum = 0;
    for (i = 0; i < length; i++) {
        sum += pow((arr1[i] - arr2[i]), 2);
    }
    return sqrt(sum);
}

double ** subtract_squared_matrices(double ** mat1, double ** mat2, unsigned int n) {
    double ** mat;
    unsigned int i, j;

    mat = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return mat;
}

double ** multi_squared_matrices(double ** mat1, double ** mat2, unsigned int n) {
    double ** mat;
    unsigned int i, j, k;

    mat = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                mat[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return mat;
}


double ** get_pow_minus_half_diag_mat(double ** mat, unsigned int n) {
    double ** minus_squared_diag_mat;
    unsigned int i, j;

    minus_squared_diag_mat = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        minus_squared_diag_mat[i][i] = 1 / sqrt(mat[i][i]);
    }
    return minus_squared_diag_mat;
}

double ** get_identity_mat(unsigned n) {
    double ** i_mat;
    unsigned int i;

    i_mat = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        i_mat[i][i] = 1.0;
    }
    return i_mat;
}



