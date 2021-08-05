#ifndef POST_SCF_CI_H
#define POST_SCF_CI_H

#include "../molecule/hfscf_molecule.hpp"

namespace POSTSCF
{
    class post_scf_cis
    {
        public:
            explicit post_scf_cis(Eigen::Index num_orbitals, Eigen::Index nelectrons)
            : m_num_orbitals(num_orbitals),
              m_nelectrons(nelectrons)
            {
                m_spin_mos = 2 * m_num_orbitals;
            }

            ~post_scf_cis() = default;
            post_scf_cis(const post_scf_cis&) = delete;
            post_scf_cis& operator=(const post_scf_cis& other) = delete;
            post_scf_cis(const post_scf_cis&&) = delete;
            post_scf_cis&& operator=(const post_scf_cis&& other) = delete;

            void calc_cis_energies(const std::shared_ptr<MOLEC::Molecule>& mol,
                                   const Eigen::Ref<const EigenMatrix<double>>& mo_energies,
                                   const Eigen::Ref<const EigenMatrix<double>>& mo_coff,
                                   const Eigen::Ref<const EigenVector<double>>& e_rep_mat);
            
            void calc_rpa_energies(const Eigen::Ref<const EigenMatrix<double>>& mo_energies,
                                   const Eigen::Ref<const EigenMatrix<double>>& mo_coff,
                                   const Eigen::Ref<const EigenVector<double>>& e_rep_mat);

        private:
            Eigen::Index m_num_orbitals{0};
            Eigen::Index m_nelectrons{0};
            Eigen::Index m_spin_mos{0};
    };
}

#endif
// 