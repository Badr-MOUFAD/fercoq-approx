# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True algorithms.pyx

from atoms cimport *
# bonus: same imports as in atoms


cdef void one_step_coordinate_descent(int ii, DOUBLE[:] x,
        DOUBLE[:] y, DOUBLE[:] Sy, DOUBLE[:] prox_y,
        DOUBLE[:] rhx, DOUBLE[:] rf, DOUBLE[:] rhy, DOUBLE[:] rhy_ii,
        DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff, DOUBLE[:] x_ii,
        DOUBLE[:] grad,
        UINT32_t[:] blocks, UINT32_t[:] blocks_f, UINT32_t[:] blocks_h,
        UINT32_t[:] Af_indptr, UINT32_t[:] Af_indices, DOUBLE[:] Af_data,
        DOUBLE[:] cf, DOUBLE[:] bf,
        DOUBLE[:] Dg_data, DOUBLE[:] cg, DOUBLE[:] bg,
        UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices, DOUBLE[:] Ah_data,
        UINT32_t[:] inv_blocks_f,
        UINT32_t[:] inv_blocks_h, UINT32_t[:] Ah_nnz_perrow,
        UINT32_t[:] Ah_col_indices, UINT32_t[:,:] dual_vars_to_update,
        DOUBLE[:] ch, DOUBLE[:] bh,
        unsigned char** f, unsigned char** g, unsigned char** h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] primal_step_size, DOUBLE[:] dual_step_size,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil

cdef void one_step_accelerated_coordinate_descent(int ii, DOUBLE[:] x,
        DOUBLE[:] xe, DOUBLE[:] xc, DOUBLE[:] y_center, DOUBLE[:] prox_y,
        DOUBLE[:] rhxe, DOUBLE[:] rhxc, DOUBLE[:] rfe, DOUBLE[:] rfc,
        DOUBLE[:] rhy, DOUBLE* theta, DOUBLE theta0, DOUBLE* c_theta,
        DOUBLE* beta,
        DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff, DOUBLE[:] xe_ii,
        DOUBLE[:] xc_ii, DOUBLE[:] grad,
        UINT32_t[:] blocks, UINT32_t[:] blocks_f, UINT32_t[:] blocks_h,
        UINT32_t[:] Af_indptr, UINT32_t[:] Af_indices, DOUBLE[:] Af_data,
        DOUBLE[:] cf, DOUBLE[:] bf,
        DOUBLE[:] Dg_data, DOUBLE[:] cg, DOUBLE[:] bg,
        UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices, DOUBLE[:] Ah_data,
        UINT32_t[:] inv_blocks_f,
        UINT32_t[:] inv_blocks_h, UINT32_t[:] Ah_nnz_perrow,
        UINT32_t[:] Ah_col_indices, UINT32_t[:,:] dual_vars_to_update,
        DOUBLE[:] ch, DOUBLE[:] bh,
        unsigned char** f, unsigned char** g, unsigned char** h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] Lf, DOUBLE[:] norm2_columns_Ah, DOUBLE* change_in_x) nogil