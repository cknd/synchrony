/*
Simulation of a network of Izhikevich neurons, coupled by chemical synapses,
receiving uncorrelated, random input spiketrains.

The simulation was written by Ekkehard Ullner in 2013.

This is a modified version of E.U.'s original, stand-alone program,
for use e.g. via the ctypes ffi in Python.

Changelog:
- suport for networks with inhibitory synapses
- Neuron type switched from FitzHugh-Nagumo to Izhikevich
- Various simulation parameters exposed to allow arbitrary network topologies,
input patterns, neuron parameters etc.
- results are written to a provided buffer instead of text files
- the non-free, problematic[0] "Numerical Recipes" RAN1 random number generator
was replaced with a Mersenne Twister rng from the GNU scientific library.
    [0 "Random Numbers in Scientific Computing: An Introduction", Katzgraber, 2010
    http://arxiv.org/pdf/1005.4117.pdf]
- cosmetics and a minimum of documentation (i.e. the comments)

CK 2014-2016

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
 *    the network is a 'grid' where each node is adressed by an (i,j) coordinate.)
 * M: number of columns j in the grid
 *
 * limit: number of simulation steps to run
 *
 * KT: maximum number of coupled neighbors of any node (degree of the network)
 * connection: function pointer, where
 *             connection(i,j,l, '0' | '1') must return the first (second) coordinate of the l'th afferent to cell ij
 *             connection(i,j,l, 's') must return the strength of that connection
 *
 * recording_voltage: array of shape (M,N,limit) in which simulation results will be written in row first order
 * recording_recov: same as recording_voltage, for the neuron's recovery variable
 * recording_spikes: same as recording_voltage, for binary log of spike events
 *
 * inputc_exc: array of shape (M,N) in row-first order. modulates the rate of input pulses fed into
 *              each cell, values within [0,1]. aka "the stimulus".
 * inputc_inh: same for inhibitory input pulses
 * laminex: lambda_in_ex, the rate of excitatory input spikes to input-receiving nodes
 * lamini: lambda_in_in,  the rate of inhibitory input spikes to input-receiving nodes
 * izhi_a: parameter a of the izhikevich neuron
 * izhi_b: parameter b of the izhikevich neuron
 * izhi_reset_c: reset constant of the neuron
 * izhi_recovery_d: recovery reset constant
 * con_upstr_exc: synapse conductivity of the excitatory random inputs
 * con_upstr_inh: synapse conductivity of the inhibitory random inputs
 * # lateral synapses:
 * taus: duration of transmitter presence after spike
 * alp: rising time constant of fraction of open receptors
 * bet: decay time constant of fraction of open receptors
 * # inhibitory synapses:
 * tauni: duration of transmitter presence after spike
 * ani: rising time constant of fraction of open receptors
 * bni: decay time constant of fraction of open receptors
 * # external excitatory synapses:
 * tauna: duration of transmitter presence after spike
 * ana: rising time constant of fraction of open receptors
 * bna: decay time constant of fraction of open receptors
 * activation_noise: scale of white noise added to each neuron's activation variable
 * seed: random seed > 0
 * verbose: print a lot or not
 * delta_t: integration step width
*/

void simulate(int M,
              int N,
              int limit,
              int KT,
              double (*connection)(int,int,int,char),
              double* recording_voltage,
              double* recording_recov,
              double* recording_spikes,
              double* inputc_exc,
              double* inputc_inh,
              double laminex,
              double lamini,
              double izhi_a,
              double izhi_b,
              double izhi_reset_c,
              double izhi_recovery_d,
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
              double activation_noise,
              int seed,
              int verbose,
              double delta_t){



    // say hello & show the current parametrisation & input pattern.
    if (verbose){
        printf("starting simulation (Izhikevich neuron)\n");
        printf("seed: %d\nM: %d\nN: %d\nlimit: %d\nKT: %d\nlaminex: %f\nlamini: %f\nizhi_a: %f\nizhi_b: %f\nizhi_reset_c: %f\nizhi_recovery_d: %f\ncon_upstr_exc: %f\ncon_upstr_inh: %f\ntaus: %f\nalp: %f\nbet: %f\ntauni: %f\nani: %f\nbni: %f\ntauna: %f\nana: %f\nbna: %f\nactivation_noise: %f\ndelta_t: %f\n",
                seed, M, N, limit, KT, laminex, lamini, izhi_a, izhi_b, izhi_reset_c, izhi_recovery_d, con_upstr_exc, con_upstr_inh, taus, alp, bet, tauni, ani, bni, tauna, ana, bna, activation_noise, delta_t);

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
    double xold[N][M],dx_h[N][M],xh[N][M],xnew[N][M]; // activation variable (previous value, first step, intermediate value, new value)
    double yold[N][M],dy_h[N][M],yh[N][M],ynew[N][M]; // recovery variable (previous value, first step, intermediate value, new value)
    double Isyn[N][M]; // collection of lateral input currents at each neuron in the current time step

    double spikes[N][M]; // recording of spike events in the current time step

    // lateral synapses:
    double rold[N][M],rh[N][M],rnew[N][M];  // fraction of open receptors (previous value, intermediate value, new value).

        // more precisely: r[i][j] is the fraction of open receptors at synapses anywhere in the network
        // that are the target of neuron (i,j). Since all synapses have the same rise and decay constants
        // alpha and beta, the effect of a spike at neuron i,j is the same at all of its target synapses.
        // We can therefore write the fractions of open receptors of all these target synapses as a
        // single variable r[i][j], associated with the source neuron i,j.

    double Tl[N][M]; // spike arrival times "T_j" (indexing follows the same logic as r[][])
    double Hs[N][M]; // transmitter concentrations "[T]_j" (...here, too)
    double Vna  = 0.0; // synaptic reversal potential (excitatory)
    double Vni  = -80; // synaptic reversal potential (inhibitory)

    // double taus = 0.01; // duration of transmitter presence after spike
    double Tm = 1.0; // maximum transmitter concentration

    // double alp = 8.0; // rising time constant of fraction of open receptors
    // double bet = 8.0; // decay time constant of fraction of open receptors


    // activatory external input synapses:
    double rnoa[N][M],rnha[N][M],rnna[N][M]; // fraction of open receptors
    double Tnla[N][M],Hna[N][M]; // spike arrival times and transmitter concentrations

    // double tauna  =  0.01; // duration of transmitter presence after spike
    double Tna  =  1.0; // maximum transmitter concentration

    // double ana  =  8.0; // rising time constant of fraction of open receptors
    // double bna  =  8.0; // decay time constant of fraction of open receptors


    // inhibitory external input synapses:
    double rnoi[N][M],rnhi[N][M],rnni[N][M]; // fraction of open receptors (previous value, intermediate value, new value)
    double Tnli[N][M],Hni[N][M]; // spike arrival times and transmitter concentrations

    // double tauni  =  0.01; // duration of transmitter presence after spike
    double Tni = 1.0; // maximum transmitter concentration

    // double ani = 8.0; // rising time constant of fraction of open receptors
    // double bni = 8.0; // decay time constant of fraction of open receptors


    // misc:
    double izhi_reset_threshold = 30; // spike detection threshold
    long step; // simulation step count
    // double delta_t = 0.001; // integration step width
    // double delta_t = 0.01; // integration step width
    double current_time = 0.0; // simulation time

    int i,j,l; // various loop variables over network nodes.

    int p1[N][M][KT],p2[N][M][KT]; // lookup tables holding the first (p1) and second (p2) coordinate of the KT'th afferent to neuron N,M
    float conductance_net[N][M][KT]; // lookup table: synapse conductance of the KT'th afferent to neuron N,M

    double in_exc[N][M]; // excitatory stimulus strength (input noise rate modulation) for neuron N,M
    double in_inh[N][M]; // inhibitory ...


    // remaining initialisations:
    for (j = 0;j<M;j++ ){
        for (i = 0;i<N;i++ ){
            in_exc[i][j] =  inputc_exc[i + N * j];
            in_inh[i][j] =  inputc_inh[i + N * j];

            xold[i][j] = 30*(gsl_rng_uniform(rng)-0.5);
            yold[i][j] = 30*(gsl_rng_uniform(rng)-0.5);
            rold[i][j] = 0.0;
            rnoa[i][j] = 0.0;
            rnoi[i][j] = 0.0;

            Tl[i][j] = -2 * taus;
            Hs[i][j] = 0.0;
            Isyn[i][j] = 0.0;
            Tnla[i][j] = -2 * tauna;
            Tnli[i][j] = -2 * tauni;
            Hna[i][j] = 0.0;
            Hni[i][j] = 0.0;
            // network connectivity lookup tables:
            for (l = 0;l<KT;l++ ){
                p1[i][j][l]  =  (int)connection(j,i,l,'0');
                p2[i][j][l]  =  (int)connection(j,i,l,'1');
                conductance_net[i][j][l] = connection(j,i,l,'s');
    }}}

    // // some more extreme verbosity for verification
    // printf("\n connectivity structure: \n");
    // for (l = 0; l<KT; l++){
    //     printf("\n\n%d. neighbour of each unit:\n\n", l);
    //     for (j = 0;j<M;j++ ){
    //         for (i = 0;i<N;i++ ){
    //             printf("(%d,%d)   ", p1[i][j][l], p2[i][j][l]);
    //         }
    //         printf("\n");
    //     }
    // }
    // for (l = 0; l<KT; l++){
    //     printf("\n\nconductance from %d. neighbour of each unit:\n\n", l);
    //     for (j = 0;j<M;j++ ){
    //         for (i = 0;i<N;i++ ){
    //             printf("%05.2f  ", conductance_net[i][j][l]);
    //         }
    //         printf("\n");
    //     }
    // }


    // begin stepwise integration by the implicit midpoint method
    for(step = 1; step<= limit;step++ ){
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

        }}


        // first integration step
        double x;
        double y;
        double I;
        double g;
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                x = xold[i][j];
                y = yold[i][j];

                // collect lateral synaptic currents
                Isyn[i][j] = 0.0;
                for (l = 0;l<KT;l++ ){
                    if(p1[i][j][l]>=0 && p2[i][j][l]>=0){
                        // input currents to neuron (i,j) depend proportionally on
                        // r[ p1[i][j][l] ][ p2[i][j][l] ], that is, on the fraction of open
                        // receptors of all the synapses targeted by neighbour l projecting
                        // to neuron i,j. These synapses have a high fraction of open receptors
                        // if their source neuron recently fired a spike. Thus, input currents
                        // flow to neuron (i,j) if its neighbours have recently fired a spike.

                        // we use signed conductance values to encode excitatory vs inhibitory synapses.
                        // of course, conductance is in fact positive in both these synapses; what differs
                        // is the reversal potential.
                        g = conductance_net[i][j][l];
                        if (g > 0.0)
                            Isyn[i][j] -= g * rold[p1[i][j][l]][p2[i][j][l]] * (xold[i][j] - Vna);
                        else if (g < 0.0)
                            Isyn[i][j] -= -g * rold[p1[i][j][l]][p2[i][j][l]] * (xold[i][j] - Vni);

                    }
                }

                I = Isyn[i][j];

                // collect external synaptic currents
                I -= con_upstr_exc * rnoa[i][j] * (xold[i][j] - Vna);
                I -= con_upstr_inh * rnoi[i][j] * (xold[i][j] - Vni);

                // Izhikevich neuron:
                // dv/dt = 0.04v^2 + 5v + 140 - u + I
                // du/dt = a(bv - u)
                dx_h[i][j] = 0.04*x*x + 5*x + 140 - y + I;
                dy_h[i][j] = izhi_a * (izhi_b*x - y);

                // step:
                xh[i][j] = xold[i][j] + (dx_h[i][j])*delta_t;
                yh[i][j] = yold[i][j] + (dy_h[i][j])*delta_t;


                // fraction of open receptors.. dr/dt  =  alpha [T] (1 - r) -  beta r
                // ..of synapses within the network:
                rh[i][j] = rold[i][j] + (alp * Hs[i][j] * (1 - rold[i][j]) - bet * rold[i][j]) * delta_t;

                // ..of synapses receiving external noise, activatory:
                rnha[i][j] = rnoa[i][j] + (ana * Hna[i][j] * (1 - rnoa[i][j]) - bna * rnoa[i][j]) * delta_t;
                // ..of synapses receiving external noise, inhibitory:
                rnhi[i][j] = rnoi[i][j] + (ani * Hni[i][j] * (1 - rnoi[i][j]) - bni * rnoi[i][j]) * delta_t;


        }}


        // second integration step
        double dx;
        double dy;
        double Ihsyn;
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                x = xh[i][j];
                y = yh[i][j];

                Ihsyn = 0.0;
                for (l = 0;l<KT;l++ ){
                    if(p1[i][j][l]>=0 && p2[i][j][l]>=0){
                        g = conductance_net[i][j][l];
                        if (g > 0)
                            Ihsyn -= g * rh[p1[i][j][l]][p2[i][j][l]] * (xh[i][j] - Vna);
                        else if (g < 0)
                            Ihsyn -= -g * rh[p1[i][j][l]][p2[i][j][l]] * (xh[i][j] - Vni);
                    }
                }

                I =  0.5 * (Isyn[i][j] + Ihsyn);

                // input currents from external synapses
                I -= 0.5 * con_upstr_exc * (rnoa[i][j] * (xold[i][j] - Vna)  +  rnha[i][j] * (xh[i][j] - Vna));
                I -= 0.5 * con_upstr_inh * (rnoi[i][j] * (xold[i][j] - Vni)  +  rnhi[i][j] * (xh[i][j] - Vni));

                // Izhikevich neuron:
                // dv/dt = 0.04v^2 + 5v + 140 - u + I
                // du/dt = a(bv -u)
                dx = 0.04*x*x + 5*x + 140 - y + I;
                dy = izhi_a * (izhi_b*x - y);

                xnew[i][j] = xold[i][j] + 0.5*(dx_h[i][j] + dx)*delta_t;
                ynew[i][j] = yold[i][j] + 0.5*(dy_h[i][j] + dy)*delta_t;

                // fraction of open receptors.. dr/dt  =  alpha [T] (1 - r) -  beta r
                // ..of synapses within the network:
                rnew[i][j] = rold[i][j] + 0.5 * (alp * Hs[i][j] * (1 - rold[i][j]) - bet * rold[i][j]  +  alp * Hs[i][j] * (1 - rh[i][j]) - bet * rh[i][j]) * delta_t;
                // ..of synapses receiving external noise, activatory:
                rnna[i][j] = rnoa[i][j] + 0.5 * (ana * Hna[i][j] * (1 - rnoa[i][j]) - bna * rnoa[i][j]  +  ana * Hna[i][j] * (1 - rnha[i][j]) - bna * rnha[i][j]) * delta_t;
                // ..of synapses receiving external noise, inhibitory:
                rnni[i][j] = rnoi[i][j] + 0.5 * (ani * Hni[i][j] * (1 - rnoi[i][j]) - bni * rnoi[i][j]  +  ani * Hni[i][j] * (1 - rnhi[i][j]) - bni * rnhi[i][j]) * delta_t;

        }}

        // identify spike times & prepare next step
        current_time = delta_t * step;
        for (j = 0;j<M;j++ ){
            for (i = 0;i<N;i++ ){
                // note spike times within the network
                if(xnew[i][j] >= izhi_reset_threshold){
                    xnew[i][j] = izhi_reset_c;
                    ynew[i][j] += izhi_recovery_d;
                    spikes[i][j] = 1;
                    Tl[i][j] = current_time;
                }
                else spikes[i][j] = 0;

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

                if (activation_noise > 0)
                    xold[i][j] += (gsl_rng_uniform(rng)-0.5)*activation_noise;
        }}


        // write the current network state into the provided output buffers.
        // each is a 3D array of shape (M,N,limit) in row first order,
        // so it has strides proportional to (N * limit, limit, 1).
        int linearind  =  0;
        for(int oi  =  0; oi < M; oi++ ) {
            for(int oj  =  0; oj < N; oj++ ) {
                    linearind  =  oi * limit * N  +  oj * limit  + step - 1;
                    recording_voltage[linearind]  =  xnew[oj][oi];
                    recording_recov[linearind]  = ynew[oj][oi];
                    //recording_recov[linearind] = (double) Tnla[oj][oi] == current_time; // <-uncomment to show input spikes instead
                    recording_spikes[linearind]  =  spikes[oj][oi];
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
