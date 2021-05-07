# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>
# cython --cplus -X boundscheck=False -X cdivision=True helpers.pyx

from .atoms cimport *
# bonus: same imports as in atoms
import numpy as np
import sys

cdef void compute_primal_value(pb, atom* f, atom* g, atom* h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
			     DOUBLE[:] rQ,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* val, DOUBLE* infeas)

cdef DOUBLE compute_smoothed_gap(pb, atom* f, atom* g, atom* h,
                             DOUBLE[:] x, DOUBLE[:] rf, DOUBLE[:] rhx,
			     DOUBLE[:] rQ, DOUBLE[:] Sy,
			     DOUBLE[:] z, DOUBLE[:] AfTz, DOUBLE[:] w,
			     DOUBLE[:] x_center, DOUBLE[:] y_center,
                             DOUBLE[:] buff_x, DOUBLE[:] buff_y, DOUBLE[:] buff,
                             DOUBLE* beta, DOUBLE* gamma, compute_z=*,
			     compute_gamma=*)