all: cython
	python3 setup.py build_ext --inplace

python2: cython
	python2 setup.py build_ext --inplace

debug:
	cython --cplus atoms.pyx
	cython --cplus helpers.pyx
	cython --cplus algorithms.pyx
	cython --cplus cd_solver.pyx
	python setup.py build_ext --inplace
	python3 setup.py build_ext --inplace

cython: atoms.cpp helpers.cpp algorithms.cpp cd_solver.cpp

atoms.cpp: atoms.pyx atoms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False atoms.pyx

helpers.cpp: atoms.cpp helpers.pyx helpers.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False helpers.pyx

algorithms.cpp: atoms.cpp algorithms.pyx algorithms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False algorithms.pyx

cd_solver.cpp: atoms.cpp helpers.cpp algorithms.cpp cd_solver.pyx
	cython --cplus -X boundscheck=False -X cdivision=True cd_solver.pyx

