#ifndef SCF_SOLVER_PRINT_H
#define SCF_SOLVER_PRINT_H

// comon print functions for RHF UHF

#include "../pretty_print/hfscf_pretty_print.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Matrix2D = std::vector<std::vector<T> >;

template <typename T>
using Matrix1D = std::vector<T>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using HFCOUT::pretty_print_matrix;
 
namespace HFCOUT
{
    template<typename D>
    void print_core_hamiltonian(const Eigen::Ref<const EigenMatrix<D> >& hcore_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  * Core Hamiltonian             *\n";
        std::cout << "  *                              *\n";
        std::cout << "  * H(u, v) / Eh                 *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(hcore_mat);
    }
    
    template<typename D>
    void print_overlap_matrix(const Eigen::Ref<const EigenMatrix<D> >& s_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  Overlap integrals           *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  S(u, v)                     *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(s_mat);
    }
    
    template<typename D>
    void print_ortho_overlap_matrix(const Eigen::Ref<const EigenMatrix<D> >& s_mat_sqrt)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  Orthogonal overlap matrix   *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  S(i, j) ^ -1/2              *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(s_mat_sqrt);
    }
    
    template<typename D>
    void print_npot_matrix(const Eigen::Ref<const EigenMatrix<D> >& v_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  * Nuclear attraction integrals *\n";
        std::cout << "  *                              *\n";
        std::cout << "  * V(u, v) / Eh                 *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(v_mat);
    }
    
    template<typename D>
    void print_kinetic_matrix(const Eigen::Ref<const EigenMatrix<D> >& k_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  Kinetic energy integrals    *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  K(u, v) / Eh                *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(k_mat);
    }

    template<typename I>
    void print_e_rep_matrix(const Eigen::Index num_bfs)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  2 electron integrals        *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  Ve(uv|wz) / Eh              *\n";
        std::cout << "  ********************************\n";
        Index repsize = (Eigen::Index)num_bfs * (num_bfs + 1) * (num_bfs * num_bfs + num_bfs + 2) / 8;
        std::cout << "  Total = " << repsize << " (Approx " << sizeof(double) * (repsize) / 1048576 << " MB) \n";
    }

    // symmetry versions
    
    template<typename D>
    void print_core_hamiltonian(const std::vector<EigenMatrix<D>>& vhcore_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  * Core Hamiltonian             *\n";
        std::cout << "  *                              *\n";
        std::cout << "  * H(u, v) / Eh                 *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(vhcore_mat);
    }
    
    template<typename D>
    void print_overlap_matrix(const std::vector<EigenMatrix<D>>& vs_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  Overlap integrals           *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  S(u, v)                     *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(vs_mat);
    }
    
    template<typename D>
    void print_ortho_overlap_matrix(const std::vector<EigenMatrix<D>>& vs_mat_sqrt)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  Orthogonal overlap matrix   *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  S(i, j) ^ -1/2              *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(vs_mat_sqrt);
    }
    
    template<typename D>
    void print_npot_matrix(const std::vector<EigenMatrix<D>>& vv_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  * Nuclear attraction integrals *\n";
        std::cout << "  *                              *\n";
        std::cout << "  * V(u, v) / Eh                 *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(vv_mat);
    }
    
    template<typename D>
    void print_kinetic_matrix(const std::vector<EigenMatrix<D>>& vk_mat)
    {
        std::cout << "\n  ********************************\n";
        std::cout << "  *  Kinetic energy integrals    *\n";
        std::cout << "  *                              *\n";
        std::cout << "  *  K(u, v) / Eh                *\n";
        std::cout << "  ********************************\n";

        pretty_print_matrix<D>(vk_mat);
    }
}

#endif
