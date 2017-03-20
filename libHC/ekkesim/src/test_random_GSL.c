// get some values from a GNU scienctific library random number generator.
// CK 2014
#include <stdio.h>
#include <gsl/gsl_rng.h>


void rantest(long seed,int N,double* out){
    gsl_rng* rng = gsl_rng_alloc (gsl_rng_taus); // initialize a tausworthe rng
    gsl_rng_set(rng, seed); // seed it
    double number = -1;

    for(int i=0; i<N; i++){
        number = gsl_rng_uniform(rng); // sample a number
        printf("%ld  %f\n",seed,number);
        out[i] = number;
    }
    puts("---");

    gsl_rng_free(rng);
}
