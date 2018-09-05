all: cython
	python setup.py build_ext --inplace

debug:
	cython --cplus atoms.pyx
	cython --cplus cd_solver.pyx
	python setup.py build_ext --inplace

cython: atoms.cpp cd_solver.cpp

cd_solver.cpp: atoms.cpp cd_solver.pyx
	cython --cplus -X boundscheck=False -X cdivision=True cd_solver.pyx
#	cython --cplus cd_solver.pyx
atoms.cpp: atoms.pyx atoms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True atoms.pyx
