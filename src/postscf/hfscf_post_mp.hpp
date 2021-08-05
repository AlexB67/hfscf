#ifndef POST_SCF_MP_H
#define POST_SCF_MP_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <memory>
#include "../math/hfscf_tensor4d.hpp"

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using tensor4dmath::tensor4d;

namespace POSTSCF
{
    class post_scf_mp
    {
        public:
            explicit post_scf_mp(Eigen::Index num_orbitals, Eigen::Index nelectrons, Eigen::Index fcore);
            ~post_scf_mp() = default;
            post_scf_mp(const post_scf_mp&) = delete;
            post_scf_mp& operator=(const post_scf_mp& other) = delete;
            post_scf_mp(const post_scf_mp&&) = delete;
            post_scf_mp&& operator=(const post_scf_mp&& other) = delete;

            [[nodiscard]] double calc_mp2_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                                                 const Eigen::Ref<const EigenVector<double> >& e_rep_mat);
            
            [[nodiscard]] Vec3D calc_ump2_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies_a,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_energies_b,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,   
                                                 const Eigen::Ref<const EigenVector<double> >& e_rep_mat,
                                                 const Index spin);
            
            [[nodiscard]] EigenVector<double> 
                                calc_ump3_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies_a,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_energies_b,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_coff_a,
                                                 const Eigen::Ref<const EigenMatrix<double> >& mo_coff_b,   
                                                 const Eigen::Ref<const EigenVector<double> >& e_rep_mat,
                                                 const Index spin);

            [[nodiscard]] std::pair<double, double> 
            calc_mp3_energy(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                            const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                            const Eigen::Ref<const EigenVector<double> >& e_rep_mat);
            
            [[nodiscard]] std::pair<double, double> 
            calc_mp3_energy_so(const Eigen::Ref<const EigenMatrix<double> >& mo_energies,
                               const Eigen::Ref<const EigenMatrix<double> >& mo_coff,
                               const Eigen::Ref<const EigenVector<double> >& e_rep_mat);
        private:
            Eigen::Index m_num_orbitals{0};
            Eigen::Index m_nelectrons{0};
            Eigen::Index m_spin_mos{0};
            Eigen::Index m_fcore{0};
            EigenVector<double> e_rep_mo;
            tensor4d<double> e_rep_smo;
            EigenVector<double> energy_smo;
    };
}

#endif
