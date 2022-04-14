//
// Created by roydd on 4/13/2022.
//
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>

#define SIGN(x) (x >= 0 ? 1 : -1)
#define EPSILON 1.0 * pow(10, -5)
#define MAX_NUM_ITER 100

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
double ** create_matrix_p(double **, unsigned int);
double ** create_copy_mat(double **, unsigned int, unsigned int);
unsigned int is_convergence_diag(double **, double **, unsigned int);
double * get_diag_squared_matrix(double **, unsigned int);
double ** add_vector_as_first_line_matrix(double **, double *, unsigned int, unsigned int);
double ** transform_squared_matrix(double **, unsigned int);


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
        sum = 0;
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

double ** jacobi_algorithm(double ** mat, unsigned int n) {
    double ** v_mat, **p_mat, **new_A, **old_A, *eigen_values, **eigen_vectors;
    unsigned int iter;

    iter = 0;
    old_A = mat;
    p_mat = create_matrix_p(old_A, n);
    v_mat = create_copy_mat(p_mat, n, n);
    new_A = multi_squared_matrices(multi_squared_matrices(transform_squared_matrix(p_mat, n), old_A, n), p_mat, n);
    while (!(is_convergence_diag(new_A, old_A, n)) && iter < MAX_NUM_ITER) {
        iter++;
        old_A = new_A;
        p_mat = create_matrix_p(old_A, n);
        v_mat = multi_squared_matrices(v_mat, p_mat, n);
        new_A = multi_squared_matrices(multi_squared_matrices(transform_squared_matrix(p_mat, n), old_A, n), p_mat, n);
    }
    eigen_values = get_diag_squared_matrix(new_A, n);
    eigen_vectors = v_mat;
    return add_vector_as_first_line_matrix(eigen_vectors, eigen_values, n, n);
}



// Helpful methods

double ** create_zero_matrix (unsigned int rows, unsigned int cols) {
    unsigned int i, j;
    double ** mat;

    mat = malloc(sizeof(int *) * rows);
    assert(mat);
    for (i = 0; i < rows; i++) {
        mat[i] = calloc(cols, sizeof(int));
        assert(mat[i]);
    }
    return mat;
}

double ** create_copy_mat(double ** mat, unsigned int rows, unsigned int cols) {
    unsigned int i, j;
    double ** copy_mat;

    copy_mat = create_zero_matrix(rows, cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            copy_mat[i][j] = mat[i][j];
        }
    }
    return copy_mat;
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


unsigned int * get_indexes_val_off_diag_squared_mat(double ** mat, unsigned int n) {
    double max;
    unsigned int i, j;
    unsigned int * index_max;

    max = 0;
    index_max = malloc(2 * sizeof(unsigned int));
    assert(index_max);
    index_max[0] = 0;
    index_max[1] = 1;

    if (n == 1) {
        index_max[1] = 0;
        return index_max;
    }
    if (n == 2) {
        max = fabs(mat[0][1]);
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && fabs(mat[i][j]) > max) {
                max = fabs(mat[i][j]);
                index_max[0] = i;
                index_max[1] = j;
            }
        }
    }
    return index_max;
}

double ** create_matrix_p(double ** mat, unsigned int n) {
    double ** p_mat, s, t, c, theta;
    unsigned int * index_max, i, j;

    index_max = get_indexes_val_off_diag_squared_mat(mat, n);
    i = index_max[0];
    j = index_max[1];
    free(index_max);

    p_mat = get_identity_mat(n);
    theta = (mat[j][j] - mat[i][i]) / (2 * mat[i][j]);
    t = SIGN(theta) / (fabs(theta) + sqrt(pow(theta, 2) + 1));
    c = 1 / sqrt(pow(t, 2) + 1);
    s = t * c;
    p_mat[i][j] = s;
    p_mat[i][i] = c;
    p_mat[j][i] = -s;
    p_mat[j][j] = c;
    return p_mat;
}

double get_sum_squared_off_diag_element(double ** mat, unsigned int n) {
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

unsigned int is_convergence_diag(double ** mat_new, double ** mat_old, unsigned int n) {
    double sum_old, sum_new;

    sum_old = get_sum_squared_off_diag_element(mat_old, n);
    sum_new = get_sum_squared_off_diag_element(mat_new, n);
    if (sum_old - sum_new <= EPSILON) {
        return 1;
    }
    return 0;
}

double ** transform_squared_matrix(double ** mat, unsigned int n) {
    double ** trans_mat;
    unsigned int i, j;

    trans_mat = create_zero_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            trans_mat[i][j] = mat[j][i];
        }
    }
    return trans_mat;
}

double * get_diag_squared_matrix(double ** mat, unsigned int n) {
    unsigned int i;
    double * diag = malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        diag[i] = mat[i][i];
    }
    return diag;
}

double ** add_vector_as_first_line_matrix(double ** mat, double * vector, unsigned int rows_mat, unsigned int cols) {
    double ** new_mat;
    unsigned int i;

    new_mat = create_zero_matrix(rows_mat + 1, cols + 1);
    new_mat[0] = vector;
    for (i = 1; i < rows_mat + 1; i++) {
        new_mat[i] = mat[i - 1];
    }
    return new_mat;
}



unsigned int determine_k(double * eigen_values, unsigned int n) {
    unsigned int limit, i, max_i;
    double max;

    max = fabs(eigen_values[0] - eigen_values[1]);
    max_i = 0;
    limit = n / 2;

    for (i = 1; i <= limit; i++) {
        if (fabs(eigen_values[i] - eigen_values[i + 1]) > max) {
            max = fabs(eigen_values[i] - eigen_values[i + 1]);
            max_i = i;
        }
    }
    return max_i + 1;
}


double ** get_k_first_eigenvectors(double ** eigen_vectors, unsigned int n, unsigned int k) {
    return create_copy_mat(eigen_vectors, n, k);
}


double ** calc_t_mat(double ** u_mat, unsigned int rows, unsigned int cols) {
    double ** t_mat, sum;
    unsigned int i, j, k;

    sum = 0;
    t_mat = create_zero_matrix(rows, cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            for (k = 0; k < cols; k++) {
                sum += pow(u_mat[i][k], 2);
            }
            t_mat[i][j] = u_mat[i][j] / sqrt(sum);
            sum = 0;
        }
    }
    return t_mat;
}






