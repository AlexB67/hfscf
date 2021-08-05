#ifndef DIIS_SOLVER_H
#define DIIS_SOLVER_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

namespace DIIS_SOLVER
{
    class diis_solver
    {
        public:
            explicit diis_solver(const int diis_range) : m_diis_range(diis_range) {}
            ~diis_solver() = default;
            diis_solver(const diis_solver&) = delete;
            diis_solver& operator=(const diis_solver& other) = delete;
            diis_solver(const diis_solver&&) = delete;
            diis_solver&& operator=(const diis_solver&& other) = delete;

            void set_diis_range(int diis_range) {m_diis_range = diis_range;} 
            // DIIS extrapolation routine for RHF
            void diis_extrapolate(  EigenMatrix<double>& f_mat,
                                    const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                    const Eigen::Ref<const EigenMatrix<double> >& d_mat,
                                    const Eigen::Ref<const EigenMatrix<double> >& s_sqrt,
                                    bool print = true);
            
            // DIIS extrapolation for UHF
            void diis_extrapolate(  EigenMatrix<double>& f_mat_a,
                                    EigenMatrix<double>& f_mat_b,
                                    const Eigen::Ref<const EigenMatrix<double> >& s_mat,
                                    const Eigen::Ref<const EigenMatrix<double> >& d_mat_a,
                                    const Eigen::Ref<const EigenMatrix<double> >& d_mat_b,
                                    const Eigen::Ref<const EigenMatrix<double> >& s_sqrt,
                                    bool print = true);
            
            // a list of rms value for 1..N iterations
            const std::vector<double>& get_rms_values_ref() const { return rms_values;}
            // The current rms value(s) at iteration i;
            double get_current_rms() const {return rms_diff_a;}; // RHF
            // same as
            double get_current_rms_a() const {return rms_diff_a;} // UHF
            double get_current_rms_b() const {return rms_diff_b;} // UHF

        private:
            int m_diis_range;
            double rms_diff_a{std::numeric_limits<double>::max()};
            double rms_diff_b{std::numeric_limits<double>::max()};

            std::vector<EigenMatrix<double> > e_list; // DIIS procedure error matrix
            std::vector<EigenMatrix<double> > f_list_a; // DIIS procedure alpha F matrix store (reused in RHF)
            std::vector<EigenMatrix<double> > f_list_b; // DIIS procedure beta F matrix store
            std::vector<double> rms_diis;               // rms error vector;
            std::vector<double> rms_values;             // comlpetelist of all RMS values for all iterations
    };

    constexpr double diis_eps = 1.0e-14;
}
#endif