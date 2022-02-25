# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True screening.pyx

from .atoms cimport *
# bonus: same imports as in atoms
from .helpers cimport compute_smoothed_gap


cdef DOUBLE polar_matrix_norm(atom func, UINT32_t* Af_indptr,
                                  UINT32_t nb_coord, UINT32_t[:] Af_indices,
                                  DOUBLE[:] Af_data, DOUBLE Qii,
				  UINT32_t kink_number)

cdef DOUBLE dual_scaling(DOUBLE[:] z, DOUBLE[:] AfTz, DOUBLE[:] w,
                              UINT32_t n_active,
                              UINT32_t[:] active_set, pb, atom* g,
                              DOUBLE[:] buff_x)


cdef UINT32_t do_gap_safe_screening(UINT32_t[:] active_set,
                              UINT32_t n_active_prev, pb,
                              atom* f, atom* g, atom* h, DOUBLE[:] Lf, 
                              DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                              DOUBLE[:] rQ, DOUBLE[:] prox_y,
                              DOUBLE[:] z, DOUBLE[:] AfTz,
                              DOUBLE[:] xe, DOUBLE[:] xc, DOUBLE[:] rfe,
                              DOUBLE[:] rfc, DOUBLE[:] rQe, DOUBLE[:] rQc,
                              DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                              g_norms_Af, norms_Af, DOUBLE max_Lf, int accelerated)

cdef UINT32_t update_focus_set(UINT32_t[:] focus_set, UINT32_t n_active,
                         UINT32_t[:] active_set,
                         atom* g, pb, DOUBLE[:] x,
                         DOUBLE[:] buff_x, DOUBLE[:] buff)
