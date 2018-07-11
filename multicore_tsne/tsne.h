/*
 *  tsne.h
 *  Header file for t-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 *
 *  Fork by Artsiom Sanakoyeu, 2018. enorone@gmail.com
 */


#ifndef TSNE_H
#define TSNE_H

#include <string>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

template <class treeT, double (*dist_fn)(const DataPoint&, const DataPoint&) >
class TSNE
{
public:
    /*  Run TNSE algorithm.

        Arguments:
            X - double matrix of size [N, D]
            N - number of points
            D - input dimensionality
            Y - array of size [N, no_dims], to fill with the resultant embedding
            no_dims - target dimensionality
            should_normalize_input - make X zero mean and divide each element by the overall maximum value.
    */
    void run(double* X, int N, int D, double* Y,
               int no_dims = 2, double perplexity = 30, double theta = .5,
               int num_threads = 1, int max_iter = 1000, int random_state = 0,
               bool init_from_Y = false, double* lr_mult = NULL, int verbose = 0,
               double early_exaggeration = 12, double learning_rate = 200,
               double *final_error = NULL, bool should_normalize_input = true);
    void symmetrizeMatrix(int** offset_P, int** nns_P, double** val_P, int N);

private:
    // Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
    double computeGradient(int* inp_offset_P, int* inp_nns_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, bool eval_error);
    double evaluateError(int* offset_P, int* nns_P, double* val_P, double* Y, int N, int no_dims, double theta);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, int** _offset_P, int** _nns_P, double** _val_P, double perplexity, int K, int verbose);
    double randn();
};

#endif

