#ifndef HFSCF_FREQ_H
#define HFSCF_FREQ_H

#include "../molecule/hfscf_molecule.hpp"
#include <memory>


using MOLEC::Molecule;

namespace FREQ
{
    int hessian_projector(const std::shared_ptr<MOLEC::Molecule>& m_mol, EigenMatrix<double>& Projector, bool project_rot = true);

    void calc_frequencies(const std::shared_ptr<MOLEC::Molecule>& m_mol, 
                          const Eigen::Ref<const EigenMatrix<double> >& hes,
                          const Eigen::Ref<const EigenMatrix<double> >& dipderiv,
                          const double E_electronic);
    
    void print_harmonic_frequencies(const Eigen::Ref<const EigenMatrix<double> >& mwhessian,
                                    const Eigen::Ref<const EigenMatrix<double> >& hes,
                                    const Eigen::Ref<const EigenMatrix<double> >& projector,
                                    const Eigen::Ref<const Eigen::VectorXcd>& evals,
                                    const Eigen::Ref<const Eigen::VectorXcd>& evals_projected,
                                    const Eigen::Ref<const EigenMatrix<double> >& norm_vectors,
                                    const Eigen::Ref<const EigenVector<double> >& reduced_mass,
                                    const Eigen::Ref<const EigenVector<double> >& ir_intensity,
                                    const std::shared_ptr<MOLEC::Molecule>& molecule,
                                    int linear_dep_dim, const double E_electronic);
    
    void print_thermo_chemistry(const std::shared_ptr<MOLEC::Molecule>& molecule, 
                                const Eigen::Ref<const Eigen::VectorXcd>& evals,
                                const double E_electronic, int linear_dep_dim);
}

#endif