# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True algorithms.pyx

from .atoms cimport *
# bonus: same imports as in atoms


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef void dual_prox_operator(DOUBLE[:] buff, DOUBLE[:] buff_y,
                             DOUBLE[:] y, DOUBLE[:] prox_y,
                             DOUBLE[:] rhx, int len_pb_h, atom* h,
                             UINT32_t[:] blocks_h, DOUBLE[:] dual_step_size,
                             DOUBLE[:] ch) nogil

cdef void one_step_coordinate_descent(DOUBLE[:] x,
        DOUBLE[:] y, DOUBLE[:] Sy, DOUBLE[:] prox_y,
        DOUBLE[:] rhx, DOUBLE[:] rf, DOUBLE[:] rhy, DOUBLE[:] rhy_ii,
	DOUBLE[:] rQ,
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
        UINT32_t[:] Q_indptr, UINT32_t[:] Q_indices, DOUBLE[:] Q_data,
        atom* f, atom* g, atom* h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] primal_step_size, DOUBLE[:] dual_step_size,
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active,
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
	UINT32_t per_pass,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil

cdef void one_step_accelerated_coordinate_descent(DOUBLE[:] x,
        DOUBLE[:] xe, DOUBLE[:] xc, DOUBLE[:] y_center, DOUBLE[:] prox_y,
        DOUBLE[:] rhxe, DOUBLE[:] rhxc, DOUBLE[:] rfe, DOUBLE[:] rfc,
        DOUBLE[:] rhy, DOUBLE[:] rQe, DOUBLE[:] rQc,
	DOUBLE* theta, DOUBLE theta0, DOUBLE* c_theta,
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
        UINT32_t[:] Q_indptr, UINT32_t[:] Q_indices, DOUBLE[:] Q_data,
        atom* f, atom* g, atom* h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] Lf, DOUBLE[:] norm2_columns_Ah,
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active,
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
	UINT32_t per_pass,
        DOUBLE* change_in_x) nogil
