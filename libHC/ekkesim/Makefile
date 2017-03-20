# compile the simulation as a shared library.
# produces a position independent shared object,
# linked to the GNU scientific lib & its BLAS implementation & lib math
all:
	gcc -c -fpic -O3 -Wall -std=c99 src/simulation.c
	gcc -shared -fpic -o ekkesim.so simulation.o -lgsl -lgslcblas -lm
	rm simulation.o


# you may not have the GSL installed in /usr/lib (e.g. as a non-root user).
# in that case, use "make local PREFIX={the prefix path used to install the GSL}"
local:
	gcc -c -fpic -O3 -Wall -std=c99 -I $(PREFIX)/include src/simulation.c
	gcc -shared -fpic -L $(PREFIX)/lib -o ekkesim.so simulation.o -lgsl -lgslcblas -lm
	rm simulation.o



# compile the rng test script. produces a position independent shared object,
# linked to GNU scientific lib & its BLAS implementation & lib math
rngtest:
	gcc -c -fpic -O3 -Wall -std=c99 src/test_random_GSL.c
	gcc -shared -fpic -o test_random_GSL.so test_random_GSL.o -lgsl -lgslcblas -lm
	rm test_random_GSL.o


izhikevich:
	gcc -c -fpic -O3 -Wall -std=c99 src/simulation_izhikevich.c
	gcc -shared -fpic -o simulation_izhikevich.so simulation_izhikevich.o -lgsl -lgslcblas -lm
	rm simulation_izhikevich.o
