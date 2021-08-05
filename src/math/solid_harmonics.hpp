#ifndef SOLID_HARMONICS_H
#define SOLID_HARMONICS_H

// Adapted from erkale - DFT from hel.
// Changes for eigen and prefactor conventions
// Note, my translation matrix is the transpose

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

namespace hfscfmath
{

double Cnr(int n, int k);
double dblfac(int n);

/// Get transformation matrix (cart, Y_lm)
EigenMatrix<double> Ylm_transmat(int l);

/**
 * Computes cartesian coefficients of Y_lm.
 *
 * Based on the article
 * http://en.wikipedia.org/wiki/Solid_spherical_harmonics
 */
EigenVector<double> calcYlm_coeff(int l, int m);
}

#endif