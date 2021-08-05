#ifndef IRC_LINALG_H
#define IRC_LINALG_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
// #include <Eigen/Eigenvalues>
// #include <cmath>

namespace irc {

namespace linalg {

/// Size of a given container
///
/// \tparam T
/// \param a Container
/// \return Size of \param a
template<typename T>
std::size_t size(const T& a) {
  return a.size();
}

/// Number of rows of a given matrix
///
/// \tparam T
/// \param a Matrix
/// \return Number of rows of \param a
template<typename T>
std::size_t n_rows(const T& a) {
return a.rows();
}

/// Number of columns of a given matrix
///
/// \tparam T
/// \param a Matrix
/// \return Number of columns of \param a
template<typename T>
std::size_t n_cols(const T& a) {
  return a.cols();
}

/// Norm of a given vector or matrix
///
/// 2-norm for both vectors and matrices
///
/// \tparam T
/// \return Norm of \param a
template<typename T>
double norm(const T& a) {
  return a.norm();
}

/// Normalise vector or matrix
///
/// \tparam T
/// \return Normalised version of \param a
template<typename T>
T normalize(const T& a) {
  return a.normalized();
}

/// Dot product between two vectors
///
/// \tparam T
/// \param a Vector
/// \param b Vector
/// \return Dot product between \param a and \param b
template<typename Vector>
double dot(const Vector& a, const Vector& b) {
  return a.dot(b);
}

/// Cross product between two vectors
///
/// \tparam Vector3
/// \param a Vector
/// \param b Vector
/// \return Cross product between \param a and \param b
template<typename Vector3>
Vector3 cross(const Vector3& a, const Vector3& b) {
  return a.cross(b);
}

/// Allocate column vector of zeros
///
/// \tparam Vector
/// \param nelements Vector size
/// \return Column vector full of zeros
template<typename Vector>
Vector zeros(std::size_t nelements) {
  return Vector::Zero(nelements);
}

/// Allocate matrix of zeros
///
/// \tparam Matrix
/// \param nrows Number of rows
/// \param ncols Number of columns
/// \return Matrix full of zeros
template<typename Matrix>
Matrix zeros(std::size_t nrows, std::size_t ncols) {
  return Matrix::Zero(nrows, ncols);
}

/// Allocate matrix of ones
/// \tparam Matrix
/// \param nrows Number of rows
/// \param ncols Number of columns
/// \return Matrix full of ones
template<typename Matrix>
Matrix ones(std::size_t nrows, std::size_t ncols) {
  return Matrix::Ones(nrows, ncols);
}

/// Allocate identity matrix
///
/// \tparam Matrix
/// \param n Linear size of the identity matrix
/// \return Identity matrix
template<typename Matrix>
Matrix identity(std::size_t n) {
  return Matrix::Identity(n, n);
}

/// Matrix transpose
///
/// \tparam Matrix
/// \param mat Matrix
/// \return Transpose of \param mat
template<typename Matrix>
Matrix transpose(const Matrix& mat) {
  return mat.transpose();
}

/// Inverse matrix
///
/// \tparam Matrix
/// \param mat Matrix
/// \return Inverse of \param mat
template<typename Matrix>
Matrix inv(const Matrix& mat) {
  return mat.inverse();
}

/// Pseudo-inverse matrix
///
/// \tparam Matrix
/// \param mat Matrix
/// \return Pseudo-inverse of \param mat

using Eigen::Index;
template<typename Matrix>
Matrix pseudo_inverse(const Matrix& mat) {
  
  Eigen::CompleteOrthogonalDecomposition<Matrix> solver = mat.completeOrthogonalDecomposition();
  solver.setThreshold(1.0E-10);
  return solver.pseudoInverse();
  //return mat.completeOrthogonalDecomposition().psuedoInverse();
}

} // namespace linalg

} // namespace irc

#endif // IRC_LINALG_H_H
