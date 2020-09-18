all:
	python3 setup.py build_ext --inplace

python2:
	python2 setup.py build_ext --inplace

debug:
	cython --cplus atoms.pyx
	cython --cplus helpers.pyx
	cython --cplus algorithms.pyx
	cython --cplus algorithms_purecd.pyx
	cython --cplus screening.pyx
	cython --cplus cd_solver_.pyx
	python setup.py build_ext --inplace
	python3 setup.py build_ext --inplace

cython: atoms.cpp helpers.cpp algorithms.cpp algorithms_purecd.cpp cd_solver_.cpp

atoms.cpp: atoms.pyx atoms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False atoms.pyx

helpers.cpp: atoms.cpp helpers.pyx helpers.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False helpers.pyx

algorithms.cpp: atoms.cpp algorithms.pyx algorithms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False algorithms.pyx

algorithms_purecd.cpp: atoms.cpp algorithms.cpp algorithms_purecd.pyx algorithms_purecd.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False algorithms_purecd.pyx

screening.cpp: helpers.cpp screening.pyx screening.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False screening.pyx

cd_solver.cpp: atoms.cpp helpers.cpp algorithms.cpp algorithms_purecd.cpp screening.cpp cd_solver_.pyx
	cython --cplus -X boundscheck=False -X cdivision=True cd_solver_.pyx
