all:
	python3 setup.py build_ext --inplace

python2:
	python2 setup.py build_ext --inplace

debug:
	cython --cplus atoms.pyx
	cython --cplus helpers.pyx
	cython --cplus algorithms.pyx
	cython --cplus algorithms_stripd.pyx
	cython --cplus screening.pyx
	cython --cplus cd_solver.pyx
	python setup.py build_ext --inplace
	python3 setup.py build_ext --inplace

cython: atoms.cpp helpers.cpp algorithms.cpp algorithms_stripd.cpp cd_solver.cpp

atoms.cpp: atoms.pyx atoms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False atoms.pyx

helpers.cpp: atoms.cpp helpers.pyx helpers.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False helpers.pyx

algorithms.cpp: atoms.cpp algorithms.pyx algorithms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False algorithms.pyx

algorithms_stripd.cpp: atoms.cpp algorithms.cpp algorithms_stripd.pyx algorithms_stripd.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False algorithms_stripd.pyx

screening.cpp: helpers.cpp screening.pyx screening.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False screening.pyx

cd_solver.cpp: atoms.cpp helpers.cpp algorithms.cpp algorithms_stripd.cpp screening.cpp cd_solver.pyx
	cython --cplus -X boundscheck=False -X cdivision=True cd_solver.pyx
