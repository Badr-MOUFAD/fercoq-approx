all:
	make cython
	python3 setup.py build_ext --inplace

debug:
	cython --cplus -X language_level=3 atoms.pyx
	cython --cplus -X language_level=3 helpers.pyx
	cython --cplus -X language_level=3 algorithms.pyx
	cython --cplus -X language_level=3 algorithms_purecd.pyx
	cython --cplus -X language_level=3 screening.pyx
	cython --cplus -X language_level=3 cd_solver_.pyx
	python3 setup.py build_ext --inplace

cython: atoms.cpp helpers.cpp algorithms.cpp algorithms_purecd.cpp cd_solver_.cpp

atoms.cpp: atoms.pyx atoms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False -X language_level=3 atoms.pyx

helpers.cpp: atoms.cpp helpers.pyx helpers.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False -X language_level=3 helpers.pyx

algorithms.cpp: atoms.cpp algorithms.pyx algorithms.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False -X language_level=3 algorithms.pyx

algorithms_purecd.cpp: atoms.cpp algorithms.cpp algorithms_purecd.pyx algorithms_purecd.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False -X language_level=3 algorithms_purecd.pyx

screening.cpp: helpers.cpp screening.pyx screening.pxd
	cython --cplus -X boundscheck=False -X cdivision=True -X wraparound=False -X language_level=3 screening.pyx

cd_solver_.cpp: atoms.cpp helpers.cpp algorithms.cpp algorithms_purecd.cpp screening.cpp cd_solver_.pyx
	cython --cplus -X boundscheck=False -X cdivision=True -X language_level=3 cd_solver_.pyx


clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f
	find . -name "*.cpp" | xargs rm -f
	find . -name "*.c" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags