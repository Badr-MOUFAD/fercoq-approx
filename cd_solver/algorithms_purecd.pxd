# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True algorithms_purecd.pyx

from .atoms cimport *
# bonus: same imports as in atoms


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef void one_step_pure_cd(DOUBLE[:] x, DOUBLE[:] x_av,
        DOUBLE[:] y, DOUBLE [:] y_av, DOUBLE[:] prox_y, DOUBLE[:] prox_y_cpy,
	DOUBLE[:] rhx,
        DOUBLE[:] rhx_jj, DOUBLE[:] rf, DOUBLE[:] rQ,
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
        DOUBLE [:] theta_pure_cd,
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active, 
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
        UINT32_t per_pass, DOUBLE[:] averages,
        DOUBLE* change_in_x, DOUBLE* change_in_y) 


cdef void one_step_s_pdhg(DOUBLE[:] x,
        DOUBLE[:] y, DOUBLE[:] rhx,
        DOUBLE[:] rhx_jj, DOUBLE[:] rf, DOUBLE[:] rQ,
        DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff, DOUBLE[:] x_ii,
        DOUBLE[:] grad,
        UINT32_t[:] blocks, UINT32_t[:] blocks_f, UINT32_t[:] blocks_h,
        UINT32_t[:] Af_indptr, UINT32_t[:] Af_indices, DOUBLE[:] Af_data,
        DOUBLE[:] cf, DOUBLE[:] bf,
        DOUBLE[:] Dg_data, DOUBLE[:] cg, DOUBLE[:] bg,
        UINT32_t[:] Ah_indptr, UINT32_t[:] Ah_indices, DOUBLE[:] Ah_data,
        UINT32_t[:] inv_blocks_f, UINT32_t[:] inv_blocks_h, 
        DOUBLE[:] ch, DOUBLE[:] bh,
        UINT32_t[:] Q_indptr, UINT32_t[:] Q_indices, DOUBLE[:] Q_data,
        atom* f, atom* g, atom* h,
        int f_present, int g_present, int h_present,
        DOUBLE[:] primal_step_size, DOUBLE[:] dual_step_size,
        int sampling_law, UINT32_t* rand_r_state,
        UINT32_t[:] active_set, UINT32_t n_active, 
        UINT32_t[:] focus_set, UINT32_t n_focus, UINT32_t n,
        UINT32_t nh, UINT32_t per_pass,
        DOUBLE* change_in_x, DOUBLE* change_in_y) nogil

