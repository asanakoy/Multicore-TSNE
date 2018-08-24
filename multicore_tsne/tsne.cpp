/*
 *  tsne.cpp
 *  Implementation of Barnes-Hut-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 *
 *  Fork by Artsiom Sanakoyeu, 2018. enorone@gmail.com
 */

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

// #include "quadtree.h"
#include "splittree.h"
#include "vptree.h"
#include "tsne.h"


#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif

 // WARNING: No early exaggeration will be made if lr_mult != NULL
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::run(double* X, int N, int D, double* Y,
               int no_dims, double perplexity, double theta ,
               int num_threads, int max_iter, int random_state,
               bool init_from_Y, double* lr_mult, int verbose,
               double early_exaggeration, double learning_rate,
               double *final_error, double *final_pairs_error, bool should_normalize_input,
               int disjoint_set_size,
               double contrib_cost_pairs, const std::vector<std::pair<int, int> >& pairs) {
    if (contrib_cost_pairs < 0) {
        throw std::invalid_argument("contrib_cost_pairs multiplier cannot be < 0!");
    }
    if (early_exaggeration <= 0) {
        throw std::invalid_argument("early_exaggeration multiplier cannot be <= 0!");
    }
    if (disjoint_set_size < 0) {
        throw std::invalid_argument("disjoint_set_size must be >= 0");
    }

    if (N - 1 < 3 * perplexity) {
        perplexity = (N - 1) / 3;
        if (verbose)
            fprintf(stderr, "Perplexity too large for the number of data points! Adjusting ...\n");
    }

#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS(num_threads));
#if _OPENMP >= 200805
    omp_set_schedule(omp_sched_guided, 0);
#endif
#endif

    /* 
        ======================
            Step 1
        ======================
    */

    if (verbose)
        fprintf(stderr, "Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);

    // Set learning parameters
    float total_time = .0;
    time_t start, end;
    int stop_lying_iter = 250, mom_switch_iter = 250;
    double momentum = .5, final_momentum = .8;
    double eta = learning_rate;
    bool force_compute_pairs_error = true && pairs.size() > 0;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    std::vector<double> dY_pairs;
    if (contrib_cost_pairs > 0 || force_compute_pairs_error) {
        dY_pairs.assign(N * no_dims, 0.0);
    }
    double* uY    = (double*) calloc(N * no_dims , sizeof(double));
    // The learning rate η is initially set to 100 and it is updated after every
    // iteration by means of the adaptive learning rate scheme described by
    // Jacobs (1988)." The Jacobs paper is: R.A. Jacobs
    // The idea is to have parameter-dependent learning rates.
    // gains contain the parameter-dependent learning rate corrections.
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if (dY == NULL || uY == NULL || gains == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int i = 0; i < N * no_dims; ++i) {
        gains[i] = 1.0;
    }

    // Normalize input data (to prevent numerical problems)
    if (verbose)
        fprintf(stderr, "Computing input similarities...\n");

    start = time(0);
    if (dist_fn == precomputed_distance && should_normalize_input) {
        if (verbose)
            fprintf(stderr, "Ignoring should_normalize_input flag for metric='precomputed'...\n");
        should_normalize_input = false;
    }
    if (should_normalize_input) {
        assert(dist_fn != precomputed_distance);
        zeroMean(X, N, D);
        double max_X = .0;
        for (int i = 0; i < N * D; ++i) {
            if (X[i] > max_X) max_X = X[i];
        }
        for (int i = 0; i < N * D; ++i) {
            X[i] /= max_X;
        }
    }
    // Compute input similarities
    int* offset_P; int* nns_P; double* val_P;
    // Compute asymmetric pairwise input similarities
    int number_nns_to_consider = (int) (3 * perplexity);  // TODO: make as parameter
    computeGaussianPerplexity(X, N, D,
                              &offset_P, &nns_P, &val_P,
                              perplexity,
                              number_nns_to_consider,
                              verbose, disjoint_set_size);

    // Symmetrize input similarities
    symmetrizeMatrix(&offset_P, &nns_P, &val_P, N);
    // Renormalize probabilities to have sum = 1.
    double sum_P = .0;
    for (int i = 0; i < offset_P[N]; ++i) {
        sum_P += val_P[i];
    }
    for (int i = 0; i < offset_P[N]; ++i) {
        val_P[i] /= sum_P;
    }

    end = time(0);
    if (verbose)
        fprintf(stderr, "Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float)(end - start) , (double) offset_P[N] / ((double) N * (double) N));

    /* 
        ======================
            Step 2
        ======================
    */


    // Lie about the P-values
    for (int i = 0; i < offset_P[N]; ++i) {
        val_P[i] *= early_exaggeration;
    }

    bool lr_mult_was_passed = (lr_mult != NULL); // has at least one frozen point
    if (!lr_mult_was_passed) {
        lr_mult = new double[N];
        std::fill_n(lr_mult, N, 1.0);
    }


    // Initialize solution (randomly), unless Y is already initialized
    if (init_from_Y) {
        if (lr_mult_was_passed) {
            // Immediately stop lying if nothing is frozen.
            // Passed Y is close to the true solution.
            fprintf(stderr, "No early exaggeration will be used because lr_mult was passed.");
            stop_lying_iter = 0;
        } else {
            // do iteration with early exaggeration
        }
    } else {
        if (random_state != -1) {
            srand(random_state);
        }
        for (int i = 0; i < N * no_dims; ++i) {
            Y[i] = randn();
        }
    }

     if (pairs.size() > 0)
        contrib_cost_pairs *= (float)(N) / pairs.size();

    // Perform main training loop
    start = time(0);
    for (int iter = 0; iter < max_iter; iter++) {

        bool need_eval_error = (verbose && ((iter >= 0 && iter % 50 == 0) || (iter == max_iter - 1)));

        // Compute approximate gradient
        double tsne_error = computeGradient(offset_P, nns_P, val_P, Y, N, no_dims, dY, theta, need_eval_error, disjoint_set_size);
        if (need_eval_error && verbose >= 100) {
            auto result = std::minmax_element(dY, dY + N * no_dims);
            fprintf(stderr, "Min, max dY:      \t(%f, %f)\n", *result.first, *result.second);
        }

        double pairs_error = -1.0;
        if (contrib_cost_pairs > 0) {
            pairs_error = computeGradientPairs(pairs, Y, N, no_dims, dY_pairs, need_eval_error);

            for (int i = 0; i < N * no_dims; ++i) {
                dY[i] += contrib_cost_pairs * dY_pairs[i];
            }
            if (need_eval_error && verbose >= 100) {
                auto result = std::minmax_element(dY_pairs.begin(), dY_pairs.end());
                fprintf(stderr, "Min, max dY_pairs:\t(%f, %f)\n", contrib_cost_pairs * *result.first, contrib_cost_pairs * *result.second);
            }
        } else if (force_compute_pairs_error) {
            pairs_error = evaluatePairsError(pairs, Y, N, no_dims);
            if (need_eval_error && verbose >= 100) {
                auto result = std::minmax_element(dY_pairs.begin(), dY_pairs.end());
                fprintf(stderr, "Min, max dY_pairs:\t(%f, %f)\n", contrib_cost_pairs * *result.first, contrib_cost_pairs * *result.second);
            }
        }

        for (int i = 0; i < N * no_dims; ++i) {
            int point_idx = i / no_dims;
            if (lr_mult[point_idx] == 0.0) {
                // The point i frozen. No need to update it.
                continue;
            }

            // Update gains
            // If the sign of the gradient w.r.t. a parameter doesn't change,
            // we slowly increase the learning rate for that parameter.
            // If the sign switches, we rapidly reduce the learning rate for that parameter.
            // gains[i] is the learning rate multiplier for point #i.
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8 + .01);

            // Perform gradient update (with momentum and gains)
            uY[i] = momentum * uY[i] - eta * gains[i] * lr_mult[point_idx] * dY[i];
            Y[i] += uY[i];
        }

        // Make solution zero-mean
        zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if (iter == stop_lying_iter) {
            for (int i = 0; i < offset_P[N]; ++i) {
                val_P[i] /= early_exaggeration;
            }
        }
        if (iter == mom_switch_iter) {
            momentum = final_momentum;
        }

        // Print out progress
        if (need_eval_error) {
            end = time(0);

            total_time += (float)(end - start);
            fprintf(stderr, "Iteration %d: tsne_error is %f (50 iterations in %4.2f seconds)\n", iter + 1, tsne_error, (float)(end - start));
            if (contrib_cost_pairs > 0 || force_compute_pairs_error) {
                fprintf(stderr, "Iteration %d: pairs_error is %f\n", iter + 1, pairs_error);
            }

            start = time(0);
        }

    }
    end = time(0); total_time += (float) (end - start) ;

    if (final_error != NULL)
        *final_error = evaluateError(offset_P, nns_P, val_P, Y, N, no_dims, theta, disjoint_set_size);
     if (final_pairs_error != NULL) {
         *final_pairs_error = evaluatePairsError(pairs, Y, N, no_dims);
     }

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);

    free(offset_P); offset_P = NULL;
    free(nns_P); nns_P = NULL;
    free(val_P); val_P = NULL;
    if (!lr_mult_was_passed) {
        delete lr_mult;
    }

    if (verbose)
        fprintf(stderr, "Fitting performed in %4.2f seconds.\n", total_time);
}

/* Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)


 */
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::
computeGradient(int* inp_row_P, int* inp_nns_P, double* inp_val_P,
                double* Y, int N, int no_dims, double* dC, double theta, bool eval_error, int disjoint_set_size) {

    // Construct quadtree on current map
    // One for first disjoint set, one for another
    treeT* tree[2] = {new treeT(Y, disjoint_set_size, no_dims), new treeT(Y + disjoint_set_size * no_dims, N - disjoint_set_size, no_dims)};
    
    // Compute all terms required for t-SNE gradient
    double* Q = new double[N];
    double* pos_f = new double[N * no_dims]();
    double* neg_f = new double[N * no_dims]();

    double P_i_sum = 0.;
    double C = 0.;

    if (pos_f == NULL || neg_f == NULL) { 
        fprintf(stderr, "Memory allocation failed!\n"); exit(1);
    }
    
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:P_i_sum,C)
#endif
    for (int n = 0; n < N; ++n) {
        // Edge forces
        int ind1 = n * no_dims;
        for (int i = inp_row_P[n]; i < inp_row_P[n + 1]; ++i) {

            // Compute pairwise distance and Q-value
            double D = .0;
            int ind2 = inp_nns_P[i] * no_dims;
            if ((n < disjoint_set_size && inp_nns_P[i] >= disjoint_set_size) ||
                (n >= disjoint_set_size && inp_nns_P[i] < disjoint_set_size)) {
                fprintf(stderr, "Oops! Neighbor from another disjoint set!\n");
            }
            for (int d = 0; d < no_dims; ++d) {
                double t = Y[ind1 + d] - Y[ind2 + d];
                D += t * t;
            }
            
            // Sometimes we want to compute error on the go
            if (eval_error) {
                P_i_sum += inp_val_P[i];
                C += inp_val_P[i] * log((inp_val_P[i] + FLT_MIN) / ((1.0 / (1.0 + D)) + FLT_MIN));
            }

            D = inp_val_P[i] / (1.0 + D); // p_{ij}*q_{ij}*Z
            // Sum positive force
            for (int d = 0; d < no_dims; ++d) {
                pos_f[ind1 + d] += D * (Y[ind1 + d] - Y[ind2 + d]);
            }
        }
        
        // NoneEdge forces (repulsive)
        double this_Q = .0 ;
        // Do not repulse points from different disjoint sets
        if (n < disjoint_set_size) {
            tree[0]->computeNonEdgeForces(n, theta, neg_f + n * no_dims, &this_Q);
        } else {
            tree[1]->computeNonEdgeForces(n - disjoint_set_size, theta, neg_f + n * no_dims, &this_Q);
        }
        Q[n] = this_Q;
    }
    
    double sum_Q = 0.;
    // sum_Q is Z = \sum_{i!=j} (1 / (1 + d(y_i, y_j)**2))
    for (int i = 0; i < N; ++i) {
        sum_Q += Q[i];
    }
    // To compensate the normalization term when we have 2 disjoint sets.
    int block_ab_size = disjoint_set_size * (N - disjoint_set_size); // \sum_{i and j from different sets} 1 / ( 1 + 0**2)
    sum_Q += 2 * block_ab_size;

    // Compute final t-SNE gradient
    // neg_f[i] here is (q_{ij}*Z)**2 * (y_i - Y_j)
    for (int i = 0; i < N * no_dims; ++i) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }

    delete tree[0];
    delete tree[1];
    delete[] pos_f;
    delete[] neg_f;
    delete[] Q;

    C += P_i_sum * log(sum_Q);

    return C;
}


/* Compute gradient of the contrastive loss function for pairs


 */
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::
computeGradientPairs(const std::vector<std::pair<int, int> >& pairs,
                     const double* Y, int N, int no_dims, std::vector<double>& dC, bool eval_error) {
    double C = 0.;
    std::fill(dC.begin(), dC.end(), 0.0);

    // cannot parallelize if mapping is not 1 to 1.
    for(auto it = pairs.begin(); it != pairs.end(); ++it) {
        int ind1 = it->first * no_dims; // i
        int ind2 = it->second * no_dims; // j

        for (int k = 0; k < no_dims; ++k) {
            double diff = Y[ind1 + k] - Y[ind2 + k]; // y_ik - y_jk
            dC[ind1 + k] += diff; // dC/dy_ik
            dC[ind2 + k] += -diff;// dC/dy_jk

            if (eval_error) {
                C += Y[ind1 + k] * Y[ind1 + k] + Y[ind2 + k] * Y[ind2 + k] - 2 * Y[ind1 + k] * Y[ind2 + k];
            }
        }
    }
    C *= 0.5;
    return C;
}


// Evaluate t-SNE cost function (approximately)
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::evaluateError(int* offset_P, int* nns_P, double* val_P, double* Y, int N, int no_dims, double theta, int disjoint_set_size)
{

    // Get estimate of normalization term
    // One tree for first disjoint set, one for another
    treeT* tree[2] = {new treeT(Y, disjoint_set_size, no_dims), new treeT(Y + disjoint_set_size * no_dims, N - disjoint_set_size, no_dims)};

    double* buff = new double[no_dims]();
    double sum_Q = .0;
    for (int n = 0; n < disjoint_set_size; ++n) {
        tree[0]->computeNonEdgeForces(n, theta, buff, &sum_Q);
    }
    for (int n = disjoint_set_size; n < N; ++n) {
        tree[1]->computeNonEdgeForces(n - disjoint_set_size, theta, buff, &sum_Q);
    }
    sum_Q += 2 * disjoint_set_size * (N - disjoint_set_size); // \sum_{i and j from different sets} 1 / ( 1 + 0**2)
    delete tree[0];
    delete tree[1];
    delete[] buff;
    
    // Loop over all edges to compute t-SNE error
    double C = .0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:C)
#endif
    for (int n = 0; n < N; ++n) {
        int ind1 = n * no_dims;
        for (int i = offset_P[n]; i < offset_P[n + 1]; ++i) {
            double Q = .0;
            int ind2 = nns_P[i] * no_dims;
            for (int d = 0; d < no_dims; ++d) {
                double b  = Y[ind1 + d] - Y[ind2 + d];
                Q += b * b;
            }
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }
    
    return C;
}


template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::
evaluatePairsError(const std::vector<std::pair<int, int> >& pairs,
                   const double* Y, int N, int no_dims) {
    double C = 0.;

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:C)
#endif
    for(int i = 0; i < pairs.size(); ++i) {
        int ind1 = pairs[i].first * no_dims; // i
        int ind2 = pairs[i].second * no_dims; // j

        for (int k = 0; k < no_dims; ++k) {
            C += (Y[ind1 + k] * Y[ind1 + k] + Y[ind2 + k] * Y[ind2 + k] - 2 * Y[ind1 + k] * Y[ind2 + k]);
        }
    }
    C *= 0.5;
    return C;
}



/*  Compute input similarities with a fixed perplexity using ball trees
   (this function allocates memory another function should free).

   Creates a sparse matrix with results using pointers _offset_P, _nns_P, _val_P.

    Arguments:
        X - double matrix of size [N, D], points in the original space,
        N - number of input points
        D - input dimensionality
        _offset_P - pointer, used to output offsets for `_nns_P`
        _nns_P - pointer, used to output indices (j) of the K nearest neighbors for each point
        _val_P - pointer, used to output  values of pairwise similarities between points, p_{j | i};
                 (*_val_P)[offset_P[i] + k] will be the similarity of the k-th nearest neighbor
                 of point i (0 <= k < K) and point i.
        perplexity - perplexity value, a measure for information equal to 2**(Shannon entropy).
        K - number of nearest neighbors to consider for each point
        verbose - verbosity level
 */
template <class treeT, double (*dist_fn)(const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::
computeGaussianPerplexity(double* X, int N, int D, int** _offset_P, int** _nns_P,
                          double** _val_P, double perplexity, int K, int verbose, int disjoint_set_size) {

    assert(dist_fn == precomputed_distance || disjoint_set_size == 0 &&
                                               "disjoint_set_size must be 0 if distance is not precomputed");
    if (disjoint_set_size > 0) {
        if ((K > disjoint_set_size - 1) || (K > N - disjoint_set_size - 1)) {
            char msg[256];
            sprintf(msg, "number of nearest neighbors to consider (%i) is larger than disjoint set size (%i or %i)", K, disjoint_set_size, N - disjoint_set_size);
            throw std::invalid_argument(msg);
        }
    }
    if (K > N) {
        char msg[256];
        sprintf(msg, "number of nearest neighbors to consider (%i) is larger than number of points (%i )", K, N);
        throw std::invalid_argument(msg);
    }

    if (perplexity > K) fprintf(stderr, "Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_offset_P = (int *) malloc((N + 1) * sizeof(int));
    *_nns_P = (int *) calloc(N * K, sizeof(int));
    *_val_P = (double *) calloc(N * K, sizeof(double));
    if (*_offset_P == NULL || *_nns_P == NULL || *_val_P == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    /*
        offset_P -- int array of size N,  offsets for `nns_P` (i). (0, K, 2K, ... NK)
        nns_P -- int array of size N * K, stores indices (j) of the K nearest neighbors for each point;
            nns_P[offset_P[i] + k] - is the index of the k-th nearest neighbor for point i (0 <= k < K).
        val_P -- values of pairwise similarities between points, p_{j | i};
             val_P[offset_P[i] + k] - is the similarity of the k-th nearest neighbor of point i (0 <= k < K) and point i.
    */
    int *offset_P = *_offset_P;
    int *nns_P = *_nns_P;
    double *val_P = *_val_P;

    offset_P[0] = 0;
    for (int n = 0; n < N; ++n) {
        offset_P[n + 1] = offset_P[n] + K;
    }

    VpTree<DataPoint, dist_fn> *tree = NULL;
    std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for (int n = 0; n < N; ++n) {
        obj_X[n] = DataPoint(D, n, X + n * D);
    }
    if (dist_fn != precomputed_distance) {
        // Build ball tree on data set
        tree = new VpTree<DataPoint, dist_fn>();
        tree->create(obj_X);
    } else {
        assert(D == N && "X must be square distance matrix if metric='precomputed'");
    }

    // Loop over all points to find nearest neighbors
    if (verbose)
        fprintf(stderr, "Building tree...\n");

    int steps_completed = 0;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
        std::vector<double> cur_sim(K);
        std::vector<double> distances;
        std::vector<int> indices;

        // Find nearest neighbors.
        if (dist_fn != precomputed_distance) {
            // First retrieved will be obj_X[i] itself.
            std::vector<DataPoint> found_points;
            tree->search(obj_X[i], K + 1, &found_points, &distances);

            indices.reserve(found_points.size());
            std::transform(found_points.begin(), found_points.end(),
                           std::back_inserter(indices), [](DataPoint p){return p.index();});
        } else {
            double* dists_row = obj_X[i]._x;

            if (i < disjoint_set_size) {
                indices.assign(disjoint_set_size, 0);
                std::iota(indices.begin(), indices.end(), 0); // initialize indices with range [0, ... disjoint_set_size-1]
            } else {
                indices.assign(N - disjoint_set_size, 0);
                std::iota(indices.begin(), indices.end(), disjoint_set_size); // initialize indices with range [disjoint_set_size, ... N-1]
            }
            // argsort(dists_row) -> indices
            sort(indices.begin(), indices.end(),
                 [&dists_row](int idx1, int idx2) { return dists_row[idx1] < dists_row[idx2]; });
            // fill distances to the first K + 1 NNs using the corresponding indices
            std::transform(indices.begin(), indices.begin() + K + 1,
                           std::back_inserter(distances), [&dists_row](int idx) { return dists_row[idx]; });

        }
        assert(indices[0] == i);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity using binary search
        int iter = 0;
        double cur_sum_sims = DBL_MIN;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row (similarities)
            for (int m = 0; m < K; m++) {
                cur_sim[m] = exp(-beta * distances[m + 1]);
            }

            // Compute entropy of current row
            cur_sum_sims = DBL_MIN; // close to zero
            for (int m = 0; m < K; m++) {
                cur_sum_sims += cur_sim[m];
            }

            // p_{j|i} = cur_sim[j] / cur_sum_sims
            /*
               $$
               sim_{j|i} = exp(-\beta_i d_{ij}) \\  % TODO: why we use distances without squares?
               p_{j|i} = sim_{j|i} / \sum_k sim_{k|i} \\
               log(p_{j|i}) = -\beta_i d_{ij} - \ln(\sum_k sim_{k|i})
               $$

               $d_{ij}$ may be squared Euclidean distance, for example.

               Entropy of the conditional distribution $P_i$ induced by the point $i$:
               $$
                H(P_i) = - \sum_j p_{j|i} \ln(p_{j|i}) = \\
                \sum_j \frac{sim_{j|i}}{\sum_k sim_{k|i}} (\beta_i d_{ij} + \ln(\sum_k sim_{k|i})) = \\
                \frac{\sum_j \beta_i d_{ij} sim_{j|i}}{\sum_k sim_{k|i}} + \sum_j \frac{sim_{j|i} \ln(\sum_k sim_{k|i})}{\sum_k sim_{k|i}} = \\
                \frac{\sum_j \beta_i d_{ij} sim_{j|i}}{\sum_k sim_{k|i}} + \ln(\sum_k sim_{k|i}) \frac{1}{\sum_k sim_{k|i}} \sum_j sim_{j|i} = \\
                \frac{\sum_j \beta_i d_{ij} sim_{j|i}}{\sum_k sim_{k|i}} + \ln(\sum_k sim_{k|i})
               $$

            */
            double H = .0;
            for (int m = 0; m < K; m++) {
                H += beta * (distances[m + 1] * cur_sim[m]);
            }
            H = (H / cur_sum_sims) + log(cur_sum_sims); // = sum_j p_{ij} log(p_{ij})

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (int m = 0; m < K; m++) {
            cur_sim[m] /= cur_sum_sims; // = P_{j|i}
        }
        for (int m = 0; m < K; m++) {
            nns_P[offset_P[i] + m] = indices[m + 1];
            val_P[offset_P[i] + m] = cur_sim[m];
        }

        // Print progress
#ifdef _OPENMP
        #pragma omp atomic
#endif
        ++steps_completed;

        if (verbose && steps_completed % (N / 10) == 0)
        {
#ifdef _OPENMP
            #pragma omp critical
#endif
            fprintf(stderr, " - point %d of %d\n", steps_completed, N);
        }
    }

    // Clean up memory
    obj_X.clear();
    if (tree != NULL) {
        delete tree;
    }
}


/*  Symmetrize a sparse matrix P_{j|i} and get P_{ij}

    Arguments:
        _offset_P - pointer, to offsets for `_nns_P`; will be modified in-place
        _nns_P - pointer, to indices (j) of the K nearest neighbors for each point; will be modified in-place
        _val_P - pointer, to values of pairwise similarities between points, p_{j | i};
                 (*_val_P)[offset_P[i] + k] is the similarity of the k-th nearest neighbor
                 of point i (0 <= k < K) and point i;
                 will be modified in-place.
       N - number of input points
 */
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::symmetrizeMatrix(int** _offset_P, int** _nns_P, double** _val_P, int N) {

    // Get sparse matrix
    int* offset_P = *_offset_P;
    int* nns_P = *_nns_P;
    double* val_P = *_val_P;

    // Count number of elements and offsets counts of symmetric matrix
    int* num_nns = (int*) calloc(N, sizeof(int));
    if (num_nns == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int i = 0; i < N; ++i) {
        for (int pos = offset_P[i]; pos < offset_P[i + 1]; ++pos) {
            int j = nns_P[pos];

            // Check whether element (j, i) is present, i.e. point #i is among of NNs of point #j and p_{i|j} != 0.
            bool present = false;
            for (int sym_pos = offset_P[j]; sym_pos < offset_P[j + 1]; sym_pos++) {
                if (nns_P[sym_pos] == i) {
                    present = true;
                    break;
                }
            }
            num_nns[i]++;
            if (!present) {
                num_nns[j]++; // reserve space for an extra point (#i) in the nns of #cur_nn_index
            }
        }
    }
    int num_elements = 0;
    for (int n = 0; n < N; n++) {
        num_elements += num_nns[n];
    }
    // Allocate memory for symmetrized matrix
    int*    sym_offset_P = (int*)    malloc((N + 1) * sizeof(int));
    int*    sym_nns_P = (int*)    malloc(num_elements * sizeof(int));
    double* sym_val_P = (double*) malloc(num_elements * sizeof(double));
    if (sym_offset_P == NULL || sym_nns_P == NULL || sym_val_P == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }

    // Construct new offsets for symmetric matrix
    sym_offset_P[0] = 0;
    for (int i = 0; i < N; ++i) {
        sym_offset_P[i + 1] = sym_offset_P[i] + num_nns[i];
    }

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if (offset == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int i = 0; i < N; ++i) {
        for (int pos = offset_P[i]; pos < offset_P[i + 1]; ++pos) {  // considering element(i, j)
            int j = nns_P[pos];
            assert(i != j && "shouldn't have self as the nearest neighbor");
            double p_ji = val_P[pos];  // p_{j|i}

            // Check whether element (j, i) is present, i.e. p_{i|j} != 0.
            bool present = false;
            for (int sym_pos = offset_P[j]; sym_pos < offset_P[j + 1]; ++sym_pos) {
                double p_ij = val_P[sym_pos];  // p_{i|j}

                if (nns_P[sym_pos] == i) {
                    present = true;
                    if (i <= j) {  // make sure we do not add elements twice
                        sym_nns_P[sym_offset_P[i] + offset[i]] = j;
                        sym_nns_P[sym_offset_P[j] + offset[j]] = i;
                        sym_val_P[sym_offset_P[i] + offset[i]] = p_ji + p_ij;
                        sym_val_P[sym_offset_P[j] + offset[j]] = p_ji + p_ij;
                    }
                }
            }

            // If (j, i) is not present (i.e. p_{i|j} = 0), there is no addition involved
            if (!present) {
                sym_nns_P[sym_offset_P[i] + offset[i]] = j;
                sym_nns_P[sym_offset_P[j] + offset[j]] = i;
                sym_val_P[sym_offset_P[i] + offset[i]] = p_ji;
                sym_val_P[sym_offset_P[j] + offset[j]] = p_ji;
            }

            // Update offsets
            if (!present || (i <= j)) {
                offset[i]++;
                if (j != i) {
                    // shouldn't always be j!= i ?
                    offset[j]++;
                }
            }
        }
    }

    // Check that all was done right
    for (int i = 0; i < N; ++i) {
        assert(num_nns[i] == offset[i] && "corrupted sym_nns_P matrix");
    }

    // Divide the result by two
    for (int i = 0; i < num_elements; i++) {
        sym_val_P[i] /= 2.0;
    }

    // Return symmetrized matrices
    free(*_offset_P);
    free(*_nns_P);
    free(*_val_P);
    *_offset_P = sym_offset_P;
    *_nns_P = sym_nns_P;
    *_val_P = sym_val_P;

    // Free up some memory
    free(offset); offset = NULL;
    free(num_nns); num_nns  = NULL;
}


// Makes data zero-mean per dimension (axis=0)
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::zeroMean(double* X, int N, int D) {

    // Compute data mean
    double* mean = (double*) calloc(D, sizeof(double));
    if (mean == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            mean[d] += X[i * D + d];
        }
    }
    for (int d = 0; d < D; ++d) {
        mean[d] /= (double) N;
    }

    // Subtract data mean
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            X[i * D + d] -= mean[d];
        }
    }
    free(mean);
    mean = NULL;
}


// Generates a standard Gaussian random number using Box–Muller transform
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::randn() {
    double x, radius;
    do {
        x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        double y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    return x;
}

extern "C"
{
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    /*
        Arguments:
            X - double matrix of size [N, D]
            N - number of points
            D - input dimensionality
            Y - array of size [N, no_dims], to fill with the resultant embedding,
                or to use as initialization if init_from_Y == True.

            metric - one of:
                "euclidean", "sqeuclidean",
                "cosine", "cosine_prenormed",
                "angular", "angular_prenormed",
                "precomputed".
                If "cosine_distance_prenormed" or "angular_distance_prenormed" is used
                then each row of X must have norm 1.

     */
    extern void tsne_run_double(double* X, int N, int D, double* Y,
                                int no_dims = 2,
                                double perplexity = 30,
                                double theta = .5,
                                int disjoint_set_size = 0,
                                int num_threads = 1,
                                int max_iter = 1000,
                                int random_state = -1,
                                bool init_from_Y = false,
                                double* lr_mult = NULL,
                                int verbose = 0,
                                double early_exaggeration = 12,
                                double learning_rate = 200,
                                double *final_error = NULL,
                                double *final_pairs_error = NULL,
                                const char* metric = "sqeuclidean",
                                bool should_normalize_input = true,
                                double contrib_cost_pairs = 0.0,
                                int* pairs = NULL, int n_pairs = 0) {

        if (verbose)
            fprintf(stderr, "Performing t-SNE using %d cores.\n", NUM_THREADS(num_threads));

        std::string str_metric = std::string(metric);
        if ((str_metric != "precomputed") &&
            (str_metric != "euclidean") && (str_metric != "sqeuclidean")
            && (str_metric != "cosine") && (str_metric != "cosine_prenormed")
            && (str_metric != "angular") && (str_metric != "angular_prenormed")
            && (str_metric != "angular_time_prenormed")
            && (str_metric != "angular_threshold_prenormed")) {
            throw std::invalid_argument(std::string("received invalid metric name:") + str_metric);
        }

        std::vector<std::pair<int, int> > pairs_vec;
        if (pairs != NULL && n_pairs != 0) {
            pairs_vec.assign(n_pairs, std::pair<int, int>(-1, -1));
            for (int i = 0; i < n_pairs; ++i) {
                pairs_vec[i].first = pairs[i * 2];
                pairs_vec[i].second = pairs[i * 2 + 1];
            }
        }

        if (str_metric == "precomputed") {
            TSNE<SplitTree, precomputed_distance> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "euclidean") {
            TSNE<SplitTree, euclidean_distance> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "sqeuclidean") {
            TSNE<SplitTree, euclidean_distance_squared> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "cosine") {
            TSNE<SplitTree, cosine_distance> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "cosine_prenormed") {
            TSNE<SplitTree, cosine_distance_prenormed> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "angular") {
            TSNE<SplitTree, angular_distance> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "angular_prenormed") {
            TSNE<SplitTree, angular_distance_prenormed> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "angular_time_prenormed") {
            // Experimental metric. Use on your own risk.
            const int margin = 20;
            const int slope = 100;
            const int max_time_distance = 3;
            printf("Using margin=%i, slope=%i, max_time_distance=%i\n",
                    margin, slope, max_time_distance);
            TSNE<SplitTree, angular_distance_time_prenormed<margin, slope, max_time_distance> > tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        } else if (str_metric == "angular_threshold_prenormed") {
             TSNE<SplitTree, angular_threshold_prenormed_distance> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, lr_mult, verbose, early_exaggeration, learning_rate,
                     final_error, final_pairs_error, should_normalize_input, disjoint_set_size,
                     contrib_cost_pairs, pairs_vec);
        }
    }
}
