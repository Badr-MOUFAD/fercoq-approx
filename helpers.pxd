# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True helpers.pyx

from atoms cimport *
# bonus: same imports as in atoms

cdef void compute_primal_value(pb, unsigned char** f, unsigned char** g, unsigned char** h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* val, DOUBLE* infeas)

cdef DOUBLE compute_smoothed_gap(pb, unsigned char** f, unsigned char** g, unsigned char** h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx, DOUBLE[:] Sy,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* beta, DOUBLE* gamma)