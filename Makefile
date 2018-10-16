all: cython
	python setup.py build_ext --inplace

debug:
	cython --cplus atoms.pyx
	cython --cplus helpers.pyx
	cython --cplus algorithms.pyx
	cython --cplus cd_solver.pyx
	python setup.py build_ext --inplace

cython: atoms.cpp helpers.cpp algorithms.cpp cd_solver.cpp

atoms.cpp: atoms.pyx atoms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True atoms.pyx

helpers.cpp: atoms.cpp helpers.pyx helpers.pxd
	cython --cplus -X boundscheck=False -X cdivision=True helpers.pyx

algorithms.cpp: atoms.cpp algorithms.pyx algorithms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True algorithms.pyx

cd_solver.cpp: atoms.cpp helpers.cpp algorithms.cpp cd_solver.pyx
	cython --cplus -X boundscheck=False -X cdivision=True cd_solver.pyx

