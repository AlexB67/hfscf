#ifndef POST_RHF_CC_H
#define POST_RHF_CC_H

#include <utility>
#include <memory>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../math/hfscf_tensor4d.hpp"
#include "../math/hfscf_tensors.hpp"

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using tensor4dmath::tensor4d;
using tensor4dmath::tensor4d1221;
using tensor4dmath::tensor4d1122;
using tensormath::tensor4d1234;
using tensor4dmath::symm4dTensor;
using tensormath::tensor3d;
using Eigen::Index;

namespace POSTSCF
{
    class post_rhf_ccsd
    {
        public:
            explicit post_rhf_ccsd(Index num_orbitals, Index nelectrons, Index fcore);
            ~post_rhf_ccsd() = default;
            post_rhf_ccsd(const post_rhf_ccsd&) = delete;
            post_rhf_ccsd& operator=(const post_rhf_ccsd& other) = delete;
            post_rhf_ccsd(const post_rhf_ccsd&&) = delete;
            post_rhf_ccsd&& operator=(const post_rhf_ccsd&& other) = delete;

            [[nodiscard]] std::pair<double, double> 
                          calc_ccsd(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                    const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                    const Eigen::Ref<const EigenVector<double> >& e_rep_mat);
            
            [[nodiscard]] std::vector<double> get_ccsd_energies() const
            {
                return ccsd_energies;
            }

            [[nodiscard]] std::vector<double> get_ccsd_deltas() const
            {
                return delta_energies;
            }

            [[nodiscard]] std::vector<double> get_ccsd_rms() const
            {
                return ccsd_rms;
            }

            [[nodiscard]] std::vector<std::pair<double, std::string>> get_largest_t1() const
            {
                return largest_t1;
            }

            [[nodiscard]] std::vector<std::pair<double, std::string>> get_largest_t2() const
            {
                return largest_t2;
            }


            [[nodiscard]] int get_iterations() const { return iteration;}
            
        private:
            Eigen::Index m_num_orbitals;
            Eigen::Index m_occ{0};
            Eigen::Index m_virt{0};
            Eigen::Index m_fcore{0};
            int iteration{0};
            EigenVector<double> e_rep_mo;
            EigenMatrix<double> eps;
            EigenMatrix<double> d_ai;
            EigenMatrix<double> f_ki;
            EigenMatrix<double> f_ac;
            EigenMatrix<double> f_kc;
            EigenMatrix<double> L_ki;
            EigenMatrix<double> L_ac;
            EigenMatrix<double> ts;
            EigenMatrix<double> ts_new;
            tensor4d1122<double> d_abij;
            tensor4d1122<double> td;
            tensor4d1122<double> td_new;
            tensor4d<double> w_klij;
            tensor4d<double> w_abcd;
            tensor4d1221<double> w_akic;
            tensor4d1234<double> w_akci;

            // data which can be retrieved for displaying later
            std::vector<double> ccsd_energies;
            std::vector<double> delta_energies;
            std::vector<double> ccsd_rms;
            std::vector<std::pair<double, std::string>> largest_t1; 
            std::vector<std::pair<double, std::string>> largest_t2;


            // CCSD DIIS acceleration;
            std::vector<EigenVector<double> >  ccsd_errors;
            std::vector<tensor4d1122<double> > t2_list;
            std::vector<EigenMatrix<double> >  t1_list;

            double calc_ccsd_energy() const noexcept;
            void update() noexcept;
            void T1() noexcept;
            void T2() noexcept;
            void ccsd_diis();
            void calc_largest_T1T2();
    };
}

#endif
