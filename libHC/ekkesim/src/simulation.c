/*
Simulation of a network of FitzHugh-Nagumo neurons, coupled by excitatory
chemical synapses, receiving decorrelated, random input spiketrains with
both excitatory and inhibitory components.

The simulation was written by Ekkehard Ullner in 2013.

This is a modified version of E.U.'s original, stand-alone program,
for use as a library (e.g. via the ctypes ffi in Python).
Changelog:
- Various simulation parameters are exposed (to allow arbitrary network topologies,
input patterns, neuron parameters etc.)
- results are written to a provided buffer instead of text files
- the non-free and problematic[0] "Numerical Recipes" RAN1 random number generator
was replaced with a Mersenne Twister rng from the GNU scientific library.
- cosmetics and a minimum of documentation (i.e. the comments)

C Korndoerfer 2014

[0 "Random Numbers in Scientific Computing: An Introduction", Katzgraber, 2010
http://arxiv.org/pdf/1005.4117.pdf]



The MIT License (MIT)

Copyright (c) 2013,2014 Ekkehard Ullner, Clemens Korndörfer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICU
LAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <assert.h>


void sagwas(){
    puts("simulation version Oct 12 2016");
}


/* function simulate
 * ------------------
 * Runs a single network simulation with the given parameters and writes the resulting
 * voltage traces into a given buffer.
 *
 * Parameters:
 *
 * N: Number of rows i in the simulated grid of neurons. (regardless of network topology,
 * the network is a 'grid' where each node is adressed by an (i,j) coordinate.)
 * M: number of columns j in the grid
 *
 * limit: max. simulation time
 * rec_resolution: to store only every ~'th simulation step.
 *
 * KT: maximum number of coupled neighbors of any node (degree of the network)
 * connection: function pointer, where
 *             connection(i,j,l, '0' | '1') must return the first (second) coordinate of the l'th afferent to cell ij
 *             connection(i,j,l, 's') must return the strength of that connection
 *
 * outputbuffer: array of shape (M,N,limit) in which simulation results will be written in row first order
 *
 * inputc: array of shape (M,N) in row-first order. modulates the rate of input pulses fed into
 *         each cell, values within [0,1]. aka "the stimulus".
 * laminex: firing rate "lambda_in,ex", the base rate of excitatory input spikes modulated by inputc
 * lamini: firing rate "lambda_in,in", the base rate of inhibitory input spikes modulated by inputc
 *
 * double fhn_a: parameter a of the FHN neuron
 * double fhn_eps: parameter epsilon of the FHN neuron
 *
 * double con_upstr_exc: synapse conductance "g^{up,ex}" of the excitatory random external inputs.
 * double con_upstr_inh: synapse conductance "g^{up,in}" of the inhibitory random external inputs.
 *
 * seed: a random seed > 0
 *
 * delta_t : integration step width
 *
 * Return: None (simulation results are passed back as a side effect, by filling the provided output buffer)
*/

void simulate(int M,
              int N,
              int limit,
              int rec_resolution,
              int KT,
              double (*connection)(int,int,int,char),
              double* outputbuffer,
              double* debug_outbuf,
              double* inputc_exc,
              double* inputc_inh,
              double laminex,
              double lamini,
              double fhn_a,
              double fhn_eps,
              double con_upstr_exc,
              double con_upstr_inh,
              double taus,
              double alp,
              double bet,
              double tauni,
              double ani,
              double bni,
              double tauna,
              double ana,
              double bna,
              int seed,
              int verbose,
              double delta_t){



    // say hello & show the current input pattern.
    if (verbose){
        printf(" simulation with seed %d\n",seed);
        printf("laminex: %f\nlamini: %f\nfhn_a: %f\nfhn_eps: %f\ncon_upstr_exc: %f\ncon_upstr_inh: %f\ntaus: %f\nalp: %f\
               \nbet: %f\ntauni: %f\nani: %f\nbni: %f\ntauna: %f\nana: %f\nbna: %f\n", laminex, lamini, fhn_a, fhn_eps, con_upstr_exc,
               con_upstr_inh, taus, alp, bet, tauni, ani, bni, tauna, ana, bna);
        int i1,j1;
        int print_exc, print_inh;
        for (j1 = 0; j1<M; j1++ ){
            for (i1 = 0; i1<N; i1++ ){
                print_exc = inputc_exc[i1 + N*j1]  >  0;
                print_inh = inputc_inh[i1 + N*j1]  >  0;
                if (print_exc && print_inh) printf(" ±");
                else if (print_exc) printf(" +");
                else if (print_inh) printf(" -");
                else printf(" .");
            }
            printf("\n");
        }
    }

    assert(seed>0);
    gsl_rng* rng  =  gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);

    // Initialisations & constant parameters:
    // neuron model:
    double xold[N][M],xh[N][M],xnew[N][M]; // activation variable (previous value, intermediate value, new value)
    double yold[N][M],yh[N][M],ynew[N][M]; // recovery variable (previous value, intermediate value, new value)
    double Isyn[N][M], Ihsyn[N][M]; // lateral input current (Ihsyn: intermediate value)

    // lateral synapses:
    double rold[N][M],rh[N][M],rnew[N][M];  // fraction of open receptors (previous value, intermediate value, new value).

        // more precisely: r[i][j] is the fraction of open receptors at synapses anywhere in the network
        // that are the target of neuron (i,j). Since all synapses have the same rise and decay constants
        // alpha and beta, the effect of a spike at neuron i,j is the same at all of its target synapses.
        // We can therefore write the fractions of open receptors of all these target synapses as a
        // single variable r[i][j], associated with the source neuron i,j.

    double Tl[N][M]; // spike arrival times "T_j" (indexing follows the same logic as r[][])
    double Hs[N][M]; // transmitter concentrations "[T]_j" (...here, too)
    double Vsyn = 0.0; // synaptic reversal potential

    // double taus = 0.01; // duration of transmitter presence after spike
    double Tm = 1.0; // maximum transmitter concentration

    // double alp = 8.0; // rising time constant of fraction of open receptors
    // double bet = 8.0; // decay time constant of fraction of open receptors


    // activatory external input synapses:
    double rnoa[N][M],rnha[N][M],rnna[N][M]; // fraction of open receptors
    double Tnla[N][M],Hna[N][M]; // spike arrival times and transmitter concentrations
    double Vna  =  0.0; // synaptic reversal potential

    // double tauna  =  0.01; // duration of transmitter presence after spike
    double Tna  =  1.0; // maximum transmitter concentration

    // double ana  =  8.0; // rising time constant of fraction of open receptors
    // double bna  =  8.0; // decay time constant of fraction of open receptors


    // inhibitory external input synapses:
    double rnoi[N][M],rnhi[N][M],rnni[N][M]; // fraction of open receptors (previous value, intermediate value, new value)
    double Tnli[N][M],Hni[N][M]; // spike arrival times and transmitter concentrations
    double Vni  =  - 2.1; // synaptic reversal potential

    // double tauni  =  0.01; // duration of transmitter presence after spike
    double Tni = 1.0; // maximum transmitter concentration

    // double ani = 8.0; // rising time constant of fraction of open receptors
    // double bni = 8.0; // decay time constant of fraction of open receptors


    // misc:
    double xth = 1.0; // spike detection threshold
    long step; // simulation step count
    // double delta_t = 0.001; // integration step width
    // double delta_t = 0.01; // integration step width
    double current_time = 0.0; // simulation time, in some irrelevant real - numbered unit

    long i,j,l; // various loop variables over network nodes. don't ask me why those were chosen as long ints.

    int p1[N][M][KT],p2[N][M][KT]; // lookup tables holding the first (p1) and second (p2) coordinate of the KT'th afferent to neuron N,M
    float conductance_net[N][M][KT]; // lookup table: synapse conductance of the KT'th afferent to neuron N,M
    double in_exc[N][M]; // excitatory stimulus strength (input noise rate modulation) for neuron N,M
    double in_inh[N][M]; // inhibitory ...


    // remaining initialisations:
    for (j = 0;j<M;j++ ){
        for (i = 0;i<N;i++ ){
            in_exc[i][j] =  inputc_exc[i + N * j];
            in_inh[i][j] =  inputc_inh[i + N * j];

            xold[i][j] = -1.05 + 0.2 * (gsl_rng_uniform(rng) - 0.5);
            yold[i][j] = -0.66 + 0.2 * (gsl_rng_uniform(rng) - 0.5);
            rold[i][j] = 0.0;
            rnoa[i][j] = 0.0;
            rnoi[i][j] = 0.0;

            Tl[i][j] = -2 * taus;
            Hs[i][j] = 0.0;
            Isyn[i][j] = 0.0;
            Ihsyn[i][j] = 0.0;
            Tnla[i][j] = -2 * tauna;
            Tnli[i][j] = -2 * tauni;
            Hna[i][j] = 0.0;
            Hni[i][j] = 0.0;
            // network connectivity lookup tables:
            for (l = 0;l<KT;l++ ){
                p1[i][j][l]  =  (int)connection(j,i,l,'0');
                p2[i][j][l]  =  (int)connection(j,i,l,'1');
                conductance_net[i][j][l]  =  connection(j,i,l,'s');
    }}}


    // begin stepwise integration.
    for(step = 1;step<= limit;step++ ){
        if (verbose && (step % (limit/100)  ==  0)){
            printf("\r...%d%%",(int)(100 * step/(double)limit));
            fflush(stdout);
        }

        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                // set transmitter presence if there was a spike recently,
                // ...for synapses within the network..
                if (Tl[i][j] + taus>current_time) Hs[i][j] = 1.0 * Tm;
                else Hs[i][j] = 0.0;

                // ..for activatory external input synapses..
                if (Tnla[i][j] + tauna>current_time) Hna[i][j] = 1.0 * Tna;
                else  Hna[i][j] = 0.0;

                // ..and for inhibitory external input synapses.
                if (Tnli[i][j] + tauni>current_time) Hni[i][j] = 1.0 * Tni;
                else  Hni[i][j] = 0.0;

                // collect lateral synaptic currents (for first integration step)
                Isyn[i][j] = 0.0;
                for (l = 0;l<KT;l++ ){
                    if(p1[i][j][l]<N && p2[i][j][l]<M)
                        Isyn[i][j]  +=   conductance_net[i][j][l]  *  rold[p1[i][j][l]][p2[i][j][l]] * (xold[i][j] - Vsyn);
                        // Here, we add input currents to neuron (i,j) that depend mostly on
                        // r[ p1[i][j][l] ][ p2[i][j][l] ], that is, on the fraction of open
                        // receptors of those synapses targeted by neighbours projecting
                        // to neuron i,j. These synapses have a high fraction of open receptors
                        // if their source neuron recently fired a spike. Thus, input currents
                        // flow to neuron (i,j) if its neighbours have recently fired a spike.
                }

        }}


        // first integration step
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                //FHN neuron: eps * (dx/dt)  =  x -  x^3 / 3 -  y
                xh[i][j]  =  xold[i][j]  +  ( (1./fhn_eps) * (xold[i][j] -  xold[i][j] * xold[i][j] * xold[i][j]/3. -  yold[i][j]) )  *  delta_t;
                xh[i][j] +=  - (1./fhn_eps) * Isyn[i][j] * delta_t;
                //FHN neuron: (dy/dt)  =  x + a
                yh[i][j] = yold[i][j] + (xold[i][j] + fhn_a) * delta_t;

                // fraction of open receptors.. dr/dt  =  alpha [T] (1 - r) -  beta r
                // ..of synapses within the network:
                rh[i][j] = rold[i][j] + (alp * Hs[i][j] * (1 - rold[i][j]) - bet * rold[i][j]) * delta_t;

                // ..of synapses receiving external noise, activatory:
                rnha[i][j] = rnoa[i][j] + (ana * Hna[i][j] * (1 - rnoa[i][j]) - bna * rnoa[i][j]) * delta_t;
                // ..of synapses receiving external noise, inhibitory:
                rnhi[i][j] = rnoi[i][j] + (ani * Hni[i][j] * (1 - rnoi[i][j]) - bni * rnoi[i][j]) * delta_t;

                // directly add input currents from the external noise synapses to the neuron's activation variable:
                xh[i][j] -=  (1./fhn_eps) * con_upstr_exc * rnoa[i][j] * (xold[i][j] - Vna) * delta_t;
                xh[i][j] -=  (1./fhn_eps) * con_upstr_inh * rnoi[i][j] * (xold[i][j] - Vni) * delta_t;
        }}

        // collect updated lateral synaptic currents
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                Ihsyn[i][j] = 0.0;
                for (l = 0;l<KT;l++ ){
                    if(p1[i][j][l]<N && p2[i][j][l]<M)
                        Ihsyn[i][j] +=  conductance_net[i][j][l] * rh[p1[i][j][l]][p2[i][j][l]] * (xh[i][j] - Vsyn);
                }
        }}

        // second integration step
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                //FHN neuron: eps * (dx/dt)  =  x -  x^3 / 3 -  y
                xnew[i][j] = xold[i][j] + 0.5 * ((1./fhn_eps) * (xold[i][j] - xold[i][j] * xold[i][j] * xold[i][j]/3. - yold[i][j]  +  xh[i][j] - xh[i][j] * xh[i][j] * xh[i][j]/3. - yh[i][j])) * delta_t;
                xnew[i][j] +=  - (1./fhn_eps) * 0.5 * (Isyn[i][j]  +  Ihsyn[i][j]) * delta_t;
                //FHN neuron: (dy/dt)  =  x + a
                ynew[i][j] = yold[i][j] + 0.5 * (xold[i][j] + fhn_a  +  xh[i][j] + fhn_a) * delta_t;

                // fraction of open receptors.. dr/dt  =  alpha [T] (1 - r) -  beta r
                // ..of synapses within the network:
                rnew[i][j] = rold[i][j] + 0.5 * (alp * Hs[i][j] * (1 - rold[i][j]) - bet * rold[i][j]  +  alp * Hs[i][j] * (1 - rh[i][j]) - bet * rh[i][j]) * delta_t;
                // ..of synapses receiving external noise, activatory:
                rnna[i][j] = rnoa[i][j] + 0.5 * (ana * Hna[i][j] * (1 - rnoa[i][j]) - bna * rnoa[i][j]  +  ana * Hna[i][j] * (1 - rnha[i][j]) - bna * rnha[i][j]) * delta_t;
                // ..of synapses receiving external noise, inhibitory:
                rnni[i][j] = rnoi[i][j] + 0.5 * (ani * Hni[i][j] * (1 - rnoi[i][j]) - bni * rnoi[i][j]  +  ani * Hni[i][j] * (1 - rnhi[i][j]) - bni * rnhi[i][j]) * delta_t;

                // directly add input currents from the external noise synapses to the neuron's activation variable:
                xnew[i][j] -=  0.5 * (1./fhn_eps) * con_upstr_exc * (rnoa[i][j] * (xold[i][j] - Vna)  +  rnha[i][j] * (xh[i][j] - Vna)) * delta_t;
                xnew[i][j] -=  0.5 * (1./fhn_eps) * con_upstr_inh * (rnoi[i][j] * (xold[i][j] - Vni)  +  rnhi[i][j] * (xh[i][j] - Vni)) * delta_t;
        }}

        // identify spike times & prepare next step
        current_time = delta_t * step;
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                // note spike times within the network
                if(xold[i][j]<xth && xnew[i][j] >= xth){
                    Tl[i][j] = current_time;
                }

                // sample random spike times for activatory external input
                if(gsl_rng_uniform(rng) <=  (laminex * (double)in_exc[i][j]) * delta_t){
                    Tnla[i][j] = current_time;
                }
                // sample random spike times for inhibitory external input
                if(gsl_rng_uniform(rng)<= (lamini * (double)in_inh[i][j]) * delta_t){
                   Tnli[i][j] = current_time;
                }

                // swap
                xold[i][j] = xnew[i][j];
                yold[i][j] = ynew[i][j];
                rold[i][j] = rnew[i][j];
                rnoa[i][j] = rnna[i][j];
                rnoi[i][j] = rnni[i][j];
        }}


        // write the current network state into the provided output buffer.
        // This is a 3D array of shape (M,N,limit) in row first order,
        // so it has strides proportional to (N * limit, limit, 1).
        // ..and we write only every rec_resolution'th step.
        int limit_rec  =  limit/rec_resolution;
        if (step % rec_resolution  ==  0) {
            int linearind  =  0;
            for(int oi  =  0; oi < M; oi++ ) {
                for(int oj  =  0; oj < N; oj++ ) {
                        linearind  =  oi * limit_rec * N  +  oj * limit_rec  +  (step/rec_resolution) - 1;
                        outputbuffer[linearind]  =  xnew[oj][oi];
                        debug_outbuf[linearind]  =  ynew[oj][oi];
                }
            }
        }
    } // end of integration loop
    if (verbose){
        printf("\033[2K");
        fflush(stdout);
    }

    gsl_rng_free(rng);
}



// helper function to play with the random number generator.
void rantest(long seed,int N,double *  out){
    gsl_rng *  rng  =  gsl_rng_alloc(gsl_rng_mt19937); // initialize a mersenne twister rng
    gsl_rng_set(rng, seed); // seed it
    double number  =  - 1;

    for(int i = 0; i<N; i++ ){
        number  =  gsl_rng_uniform(rng); // sample a number
        printf("%ld  %f\n",seed,number);
        out[i]  =  number;
    }
    puts(" - - - ");

    gsl_rng_free(rng);
}
