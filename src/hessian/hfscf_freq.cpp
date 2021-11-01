#include "../molecule/hfscf_elements.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../molecule/hfscf_constants.hpp"
#include "hfscf_freq.hpp"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>


using MOLEC::Molecule;
using Eigen::Index;
using HF_SETTINGS::hf_settings;

int FREQ::hessian_projector(const std::shared_ptr<Molecule>& m_mol, EigenMatrix<double>& Projector, bool project_rot)
{
     // Build projection
    const int natoms = static_cast<int>(m_mol->get_atoms().size());
    const auto& zval = m_mol->get_z_values();
    EigenVector<double> x3 =  EigenVector<double>(3 * natoms);
    EigenVector<double> y3 =  EigenVector<double>(3 * natoms);
    EigenVector<double> z3 =  EigenVector<double>(3 * natoms);
    EigenVector<double> sqrtmasses =  EigenVector<double>(3 * natoms);
    EigenVector<double> sqrtmassesinv =  EigenVector<double>(3 * natoms);
    EigenVector<double> ux =  EigenVector<double>::Zero(3 * natoms);
    EigenVector<double> uy =  EigenVector<double>::Zero(3 * natoms);
    EigenVector<double> uz =  EigenVector<double>::Zero(3 * natoms);

    for(Index i = 0; i < natoms; ++i)
    {
        x3(3 * i) = m_mol->get_geom()(i, 0);
        x3(3 * i + 2) = x3(3 * i + 1) = x3(3 * i);
        y3(3 * i) = m_mol->get_geom()(i, 1);
        y3(3 * i + 2) = y3(3 * i + 1) = y3(3 * i);
        z3(3 * i) = m_mol->get_geom()(i, 2);
        z3(3 * i + 2) = z3(3 * i + 1) = z3(3 * i);
        sqrtmasses(3 * i) = sqrt(ELEMENTDATA::masses[zval[i] - 1]);
        sqrtmasses(3 * i + 2) = sqrtmasses(3 * i + 1) = sqrtmasses(3 * i);
        ux(3 * i) = 1; uy(3 * i + 1) = 1; uz(3 * i + 2) = 1;
    }

    EigenVector<double> T1 = sqrtmasses.cwiseProduct(ux);
    EigenVector<double> T2 = sqrtmasses.cwiseProduct(uy);
    EigenVector<double> T3 = sqrtmasses.cwiseProduct(uz);
    EigenVector<double> R4, R5, R6;
    EigenMatrix<double> TR;

    if (project_rot)
    {
        TR = EigenMatrix<double>(6, 3 * natoms);
        R4 = sqrtmasses.cwiseProduct(y3.cwiseProduct(uz) - z3.cwiseProduct(uy));
        R5 = sqrtmasses.cwiseProduct(z3.cwiseProduct(ux) - x3.cwiseProduct(uz));
        R6 = sqrtmasses.cwiseProduct(x3.cwiseProduct(uy) - y3.cwiseProduct(ux));

        for(Index i = 0; i < 3 * natoms; ++i)
        {
            TR(0, i) = T1(i); TR(1, i) = T2(i); TR(2, i) = T3(i);
            TR(3, i) = R4(i); TR(4, i) = R5(i); TR(5, i) = R6(i);
        }
    }
    else
    {
        TR = EigenMatrix<double>(3, 3 * natoms);

        for(Index i = 0; i < 3 * natoms; ++i)
        {
            TR(0, i) = T1(i); TR(1, i) = T2(i); TR(2, i) = T3(i);
        }
    }

    Eigen::JacobiSVD<EigenMatrix<double> > svd(TR.transpose(), Eigen::ComputeThinU);
    const Eigen::VectorXd sv = svd.singularValues();
    EigenMatrix<double> U = svd.matrixU();

    constexpr double tol = 0.01;
    int dim = 0;
    for(Index i = 0; i < sv.size(); ++i)
        if(sv(i) > tol) ++dim;
    
    if(dim != sv.size()) U.conservativeResize(Eigen::NoChange, dim);

    EigenMatrix<double> TRindependent = U.transpose();
    Projector = EigenMatrix<double>::Identity(3 * natoms, 3 * natoms);

    for(Index i = 0; i < dim; ++i)
         for(Index j = 0; j < 3 * natoms; ++j)
            for(Index k = 0; k < 3 * natoms; ++k)
                Projector(j, k) -= TRindependent(i, j) * TRindependent(i, k);
    
    return dim;
}


void FREQ::calc_frequencies(const std::shared_ptr<Molecule>& m_mol, 
                            const Eigen::Ref<const EigenMatrix<double> >& hes,
                            const Eigen::Ref<const EigenMatrix<double> >& dipderiv,
                            const double E_electronic)
{
    // Mass weighted Hessian
    const auto& atoms = m_mol->get_atoms();
    const int natoms = static_cast<int>(atoms.size());
    const auto& zval = m_mol->get_z_values();
    EigenMatrix<double> mwhessian = hes;
    EigenVector<double> sqrtmassesinv(3 * natoms);

	for(int i = 0; i < natoms; ++i)
	{
        const double m_i = ELEMENTDATA::masses[zval[i] - 1];
    	for(Index j = 0; j < natoms; ++j)
		{
            const double m_j = ELEMENTDATA::masses[zval[j] - 1];
            const double srqtmu = std::sqrt(m_i * m_j);

      		mwhessian(3 * i, 3 * j) /=  srqtmu;
			mwhessian(3 * i, 3 * j + 1) /= srqtmu; 
			mwhessian(3 * i, 3 * j + 2) /= srqtmu;
            mwhessian(3 * i + 1, 3 * j) /=  srqtmu;
			mwhessian(3 * i + 1, 3 * j + 1) /= srqtmu;
			mwhessian(3 * i + 1, 3 * j + 2) /= srqtmu;
            mwhessian(3 * i + 2, 3 * j) /=  srqtmu; 
			mwhessian(3 * i + 2, 3 * j + 1) /= srqtmu;
			mwhessian(3 * i + 2, 3 * j + 2) /= srqtmu;
    	}
        
        sqrtmassesinv(3 * i) = 1.0 / std::sqrt(m_i);
        sqrtmassesinv(3 * i + 1) = 1.0 / std::sqrt(m_i);
        sqrtmassesinv(3 * i + 2) = 1.0 / std::sqrt(m_i);
  	}

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver_unprojected(mwhessian);
    Eigen::VectorXcd evals = solver_unprojected.eigenvalues();
    EigenMatrix<double> evecs_unprojected = solver_unprojected.eigenvectors();

    EigenMatrix<double> Projector;
    const bool project_rot = !hf_settings::get_project_hessian_translations_only();
    int dim = FREQ::hessian_projector(m_mol, Projector, project_rot);

    const EigenMatrix<double>& mwhessian_projected = Projector.transpose() * mwhessian * Projector;
    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver_projected(mwhessian_projected);
    const Eigen::VectorXcd& evals_projected = solver_projected.eigenvalues();
    EigenMatrix<double> evecs_projected = solver_projected.eigenvectors();

    EigenMatrix<double> wL = EigenMatrix<double>(3 * natoms, 3 * natoms);
    EigenMatrix<double> wL_unproj = EigenMatrix<double>(3 * natoms, 3 * natoms);

    for(Index i = 0; i < 3 * natoms; ++i)
        for(Index j = 0; j < 3 * natoms; ++j)
        {
            wL(i, j) = evecs_projected(i, j) * sqrtmassesinv(i);
            wL_unproj(i, j) = evecs_unprojected(i, j) * sqrtmassesinv(i);
        }

    EigenVector<double> reduced_mass = EigenVector<double>(3 * natoms);
    for(Index i = 0; i < 3 * natoms; ++i)
    {
        EigenVector<double> tmp = EigenVector<double>(3 * natoms);
        for(Index j = 0; j < 3 * natoms; ++j)
            tmp(j) = wL(j, i);
        
        reduced_mass(i) = 1.0 / tmp.squaredNorm();
    }

    EigenVector<double> IR_intensity;

    if (dipderiv.size()) // only for the analytic hessian
    {
        EigenMatrix<double> dipderiv_reshape = EigenMatrix<double>::Zero(3, 3 * natoms);

        for (Index at = 0; at < natoms; ++at)
        { 
            dipderiv_reshape(0, 3 * at) = dipderiv(3 * at, 0);
            dipderiv_reshape(0, 3 * at + 1) = dipderiv(3 * at + 1, 0);
            dipderiv_reshape(0, 3 * at + 2) = dipderiv(3 * at + 2, 0);

            dipderiv_reshape(1, 3 * at) = dipderiv(3 * at, 1);
            dipderiv_reshape(1, 3 * at + 1) = dipderiv(3 * at + 1, 1);
            dipderiv_reshape(1, 3 * at + 2) = dipderiv(3 * at + 2, 1);

            dipderiv_reshape(2, 3 * at) = dipderiv(3 * at, 2);
            dipderiv_reshape(2, 3 * at + 1) = dipderiv(3 * at + 1, 2);
            dipderiv_reshape(2, 3 * at + 2) = dipderiv(3 * at + 2, 2);
        }

        EigenMatrix<double> dDipq = dipderiv_reshape * wL;
        IR_intensity = EigenVector<double>(3 * natoms);

        for(Index i = 0; i < 3 * natoms; ++i)
            IR_intensity(i) = dDipq(0, i) * dDipq(0, i) + dDipq(1, i) * dDipq(1, i) + dDipq(2, i) * dDipq(2, i);
            // std::cout << IR_intensity[i] * 974.8801263471677 << "\n";
    }

    EigenMatrix<double> xL = EigenMatrix<double>(3 * natoms, 3 * natoms - dim);
     for(Index i = 0; i < 3 * natoms; ++i)
        for(Index j = dim; j < 3 * natoms; ++j)
            xL(i, j - dim) = wL(i, j) * sqrt(reduced_mass(j));

    FREQ::print_harmonic_frequencies(mwhessian, hes, Projector, evals, evals_projected, xL,
                                     reduced_mass, IR_intensity, m_mol, dim, E_electronic);
}

using HF_SETTINGS::hf_settings;
using namespace MOLEC_CONSTANTS;
using HFCOUT::pretty_print_matrix;

void FREQ::print_harmonic_frequencies(const Eigen::Ref<const EigenMatrix<double> >& mwhessian,
                                     const Eigen::Ref<const EigenMatrix<double> >& hes,
                                     const Eigen::Ref<const EigenMatrix<double> >& projector,
                                     const Eigen::Ref<const Eigen::VectorXcd>& evals,
                                     const Eigen::Ref<const Eigen::VectorXcd>& evals_projected,
                                     const Eigen::Ref<const EigenMatrix<double> >& norm_vectors,
                                     const Eigen::Ref<const EigenVector<double> >& reduced_mass,
                                     const Eigen::Ref<const EigenVector<double> >& ir_intensity,
                                     const std::shared_ptr<MOLEC::Molecule>& molecule,
                                     int linear_dep_dim, const double E_electronic)
{
    bool mp2 = false;
    
    if ("MP2" == hf_settings::get_frequencies_type()) mp2 = true;

    if(hf_settings::get_verbosity() > 1)
    {  
        std::cout << "\n  ************************************\n";
        std::cout << "  *  Hessian matrix                  *\n";

        if (mp2) 
            std::cout << "  *  Level: MP2                      *\n";
        else
            std::cout << "  *  Level: SCF                      *\n";      
        
        std::cout << "  *                                  *\n";
        std::cout << "  *  H(i, j) / (Eh a0^-2)            *\n";
        std::cout << "  ************************************\n";
        pretty_print_matrix<double>(hes);
    }

    std::cout << "\n  ************************************\n";
    std::cout << "  *  Mass weighted hessian matrix    *\n";

    if (mp2) 
        std::cout << "  *  Level: MP2                      *\n";
    else
        std::cout << "  *  Level: SCF                      *\n";      
            
    std::cout << "  *                                  *\n";
    std::cout << "  *  F(i, j) / (Eh / (amu a0^-2 ))   *\n";
    std::cout << "  ************************************\n";
    pretty_print_matrix(mwhessian);

    if(hf_settings::get_verbosity() > 2)
    {  
        std::cout << "\n  ************************************\n";
        std::cout << "  *        Projection Matrix         *\n";
        std::cout << "  *   Linear dependencies: " << linear_dep_dim;
        std::cout << "         *\n";
        std::cout << "  ************************************\n";
        pretty_print_matrix(projector);
    }

    std::cout << "\n  ___Frequencies___\n";
    std::cout << "\n  Pre-projected frequencies / cm^-1\n";
    std::cout << "  (";

    for (int j = 0; j < evals.size(); ++j)
    {
        if (0 == j % 9 && j > 0)
        {
            std::cout << '\n';
            std::cout << "   ";
        }
        
        if(fabs(sqrt(evals(j)).imag()) < 1E-06)
            std::cout << std::right << std::fixed << std::setprecision(4) 
            << hartree_to_wavenumber * std::sqrt(evals(j)).real();
        else
             std::cout << std::right << std::fixed << std::setprecision(4)
            << hartree_to_wavenumber * std::sqrt(evals(j)).imag() << "i";

        if(j < evals.size() - 1) std::cout << ", ";
    }

    std::cout << ")";
    std::cout << "\n\n  Linear dependencies: " << linear_dep_dim <<  "\n";
    
    if (linear_dep_dim > 3)
        std::cout << "  Projecting Hessian: Removing translational and rotational modes:\n\n";
    else
        std::cout << "  Projecting Hessian: Removing translational modes only:\n\n";

    std::cout << "  Projected frequencies / cm^-1\n";
    std::cout << "  (";

    for (int j = 0; j < evals_projected.size(); ++j)
    {
        if (0 == j % 9 && j > 0)
        {
            std::cout << '\n';
            std::cout << "   ";
        }
        
        if(fabs(sqrt(evals_projected(j)).imag()) < 1E-06)
            std::cout << std::right << std::fixed << std::setprecision(4) 
            << hartree_to_wavenumber * std::sqrt(evals_projected(j)).real();
        else
             std::cout << std::right << std::fixed << std::setprecision(4)
            << hartree_to_wavenumber * std::sqrt(evals_projected(j)).imag() << "i";

        if(j < evals_projected.size() - 1) std::cout << ", ";
    }

    std::cout << ")\n\n"; 

    std::cout << "  ***********************************************************************************************\n";
	std::cout << "  *                           Harmonic frequency analysis                                       *\n";
    if (mp2) 
        std::cout << "  *                                    Level: MP2                                               *\n";
    else
        std::cout << "  *                                    Level: SCF                                               *\n";

    std::cout << "  *                                                                                             *\n";
	std::cout << "  * #    \u03C9 / cm^-1        \u03BC / amu    F/(mDyne/A)     TP(v=0)/a0    IR/(km/mol)      Char. T/K   *\n";
	std::cout << "  ***********************************************************************************************\n";

    Index F = (linear_dep_dim > 3) ? linear_dep_dim : 0;
  
	for (Index i = evals.size() - 1; i >= F; --i)
	{

        double freq_cm = 0;

		if(fabs(sqrt(evals_projected(i)).imag()) < 1E-06)
        {
            freq_cm = hartree_to_wavenumber * std::sqrt(evals_projected(i)).real();
            if (freq_cm < 10 && F == 0) continue; // skip if printing rotation type modes
            std::cout << std::setw(5) << i + 1;
            std::cout << std::right << std::fixed << std::setprecision(4) << std::setw(13) << freq_cm;
        }
        else
        {
            std::cout << std::setw(5) << i + 1;
            std::cout << std::right << std::fixed << std::setprecision(3) << std::setw(12) 
            << hartree_to_wavenumber * std::sqrt(evals_projected(i)).imag() << "i";
        }

        std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(6) << reduced_mass(i);
        std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(6) 
                  <<  freq_cm * freq_cm * reduced_mass(i) * force_constant_mdyne;

        if(fabs(sqrt(evals_projected(i)).imag()) < 1E-06)
            std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(6) 
                    << 1.0 / (sqrt(freq_cm * reduced_mass(i)) * convs);
        else
            std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(6) << 0.0;
        
        if (ir_intensity.size())
            std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(4) 
                      << ir_intensity(i) * 974.8801263471677; // TODO add to constants
        else
            std::cout << std::setw(15) << std::right << "TODO"; 

        std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(4) 
                  << freq_cm * convk << '\n';
	}

    if (molecule->get_atoms().size() > 2)
    {
        std::cout << "\n  *********************************************************************************************";
        std::cout << "\n  *                       Normalized unmass weighted normal modes                             *";
        std::cout << "\n  *                                                                                           *";
        std::cout << "\n  * # atom X / a0   Y / a0   Z / a0     X / a0   Y / a0   Z / a0     X / a0   Y / a0   Z / a0 *";
        std::cout << "\n  *********************************************************************************************\n";
    }
    else
    {
        std::cout << "\n  ************************************";
        std::cout << "\n  *  Normalized unmass weighted      *";
        std::cout << "\n  *        Normal modes              *";
        std::cout << "\n  *                                  *";
        std::cout << "\n  * # atom X / a0   Y / a0   Z / a0  *";
        std::cout << "\n  ************************************\n";
    }

    const auto skip = [&](int i) -> bool
    {
        if (F != 0 || i < 0) return false;

        else if(fabs(sqrt(evals_projected(i)).imag()) < 1E-06
                && hartree_to_wavenumber * std::sqrt(evals_projected(i)).real() < 10)
                    return true;

        return false;
    };

    const auto& zval = molecule->get_z_values();
    int mode = evals_projected.size();
    for (int i = 3 * static_cast<int>(zval.size()) - F - 1; i >= 0; i -= 3)
    {
        std::cout << "           Mode:" << std::setw(3) << mode;

        if (molecule->get_atoms().size() > 2 || F == 0)
        {
            if (i > 1)
                std::cout << "                     Mode:" << std::setw(3) << mode - 1;
            
            if (i > 0)
                std::cout << "                     Mode:" << std::setw(3) << mode - 2;
        }

        std::cout << "\n";

        for (int j = 0; j < static_cast<int>(zval.size()); ++j)
        {
            std::cout << std::setw(5)  << std::right << j + 1; 
            std::cout << std::setw(3)  << std::right << ELEMENTDATA::atom_names[zval[j] - 1];
            
            for (int xyz = 0; xyz < 3; ++xyz)
            {
                if (skip(i) == false)
                    std::cout <<  std::right << std::setw(9) << std::setprecision(4) << norm_vectors(3 * j + xyz, i);
                else
                    std::cout <<  std::right << std::setw(9) << std::setprecision(4) << 0.;
            }

            if (i - 1 >= 0)
                for (int xyz = 0; xyz < 3; ++xyz)
                {
                    int w = (xyz == 0) ? 11 : 9;
                    if (skip(i - 1) == false)
                        std::cout <<  std::right << std::setw(w) << std::setprecision(4) << norm_vectors(3 * j + xyz, i - 1);
                    else
                        std::cout <<  std::right << std::setw(w) << std::setprecision(4) << 0.;
                }

            if (i - 2 >= 0)
                for (int xyz = 0; xyz < 3; ++xyz)
                {   int w = (xyz == 0) ? 11 : 9;
                    if (skip(i - 2) == false)
                        std::cout <<  std::right << std::setw(w) << std::setprecision(4) << norm_vectors(3 * j + xyz, i - 2);
                    else
                        std::cout <<  std::right << std::setw(w) << std::setprecision(4) << 0.;
                }

            std::cout << "\n";
        }

        std::cout << "\n\n";
        mode -= 3;
    }

    print_thermo_chemistry(molecule, evals_projected, E_electronic, linear_dep_dim);
}

using hfscfmath::pi;

void FREQ::print_thermo_chemistry(const std::shared_ptr<MOLEC::Molecule>& molecule, 
                                  const Eigen::Ref<const Eigen::VectorXcd>& evals,
                                  const double E_electronic, int linear_dep_dim)
{
    if(!molecule->get_atoms().size()) return;

    std::cout << "  ___Thermochem___\n\n";

    Index F = (linear_dep_dim > 3) ? linear_dep_dim : 0;

    double T_ = hf_settings::get_thermo_chem_temperature();
    double P_ = hf_settings::get_thermo_chem_pressure();
    const double beta = 1.0 / (Kb * T_);

    const double mass = molecule->get_mass();
    const double qelec = log(static_cast<double>(molecule->get_spin()) + 1.0);
    const double qtran = std::pow(2 * pi * mass * amu_to_kg / (beta * h * h), 1.5) * Na / (beta * P_);
    const double convert = (R / hartree_to_kjmol);
    const double Stran  = (5.0 / 2 + log(qtran / Na)) * convert;
    const double Cvtran = (3.0 / 2) * convert;
    const double Cptran = (5.0 / 2) * convert;
    const double Utran  = Cvtran * T_;
    const double Htran  = Cptran * T_;

    int sigma = molecule->get_sigma(); // Get symmetry number
    double Srot = 0; double Cprot = 0; double Cvrot = 0; double Urot = 0;
    double phiA = 0; double phiB = 0; double phiC = 0;

    if(molecule->molecule_is_linear())
    {
        Srot =  (1.0 - log(beta * sigma * 100.0 * c * h * molecule->get_rotational_constants()(1))) * convert;
        Cprot = convert;
        Cvrot = convert; 
        Urot = convert * T_;
    }
    else
    {
        phiA = (100.0 * c * h / Kb) * molecule->get_rotational_constants()(0);
        phiB = (100.0 * c * h / Kb) * molecule->get_rotational_constants()(1);
        phiC = (100.0 * c * h / Kb) * molecule->get_rotational_constants()(2);
        const double qrot = std::sqrt(pi) * std::pow(T_, 1.5 ) / (sigma * std::sqrt(phiA * phiB * phiC));
        Srot = (3.0 / 2 + log(qrot)) * convert;
        Cprot = (3.0 / 2) * convert;
        Cvrot = (3.0 / 2) * convert;
        Urot = (3.0 / 2) * convert * T_;
    }

    double Svib = 0; double Cpvib = 0; double Cvvib = 0; double Uvib = 0; double Hvib = 0;
    double zpe = 0;

    for (int  i = 0; i < evals.size(); ++i)
    {
        if(fabs(sqrt(evals(i)).imag()) < 1E-06)
        {
            const double freq_cm = hartree_to_wavenumber * std::sqrt(evals(i)).real();
            if (freq_cm < 10 && F == 0) continue; // skip if printing rotation type modes
        }
        
        if (evals(i).real() > evals(i).imag())
        {
            if (i >= F)
            {
                const double theta = convk * hartree_to_wavenumber * std::sqrt(evals(i)).real() / T_;
                Svib  += theta / (std::exp(theta) - 1.0) - log(1.0 - std::exp(-theta));
                Cvvib += std::exp(theta) * std::pow((theta / (std::exp(theta) - 1.0)), 2);
                zpe += theta * T_ / 2;
                Uvib += T_ * theta * (1.0 / 2.0 + 1.0 / (std::exp(theta) - 1.0));
            }

            if(hartree_to_wavenumber * std::sqrt(evals(i)).real() < low_mode && i >= linear_dep_dim)
                std::cout << "  Note: Low mode in thermochemistry: " 
                << hartree_to_wavenumber * std::sqrt(evals(i)).real() << "\n";
        }
    }

    Svib  *= convert;
    Cvvib *= convert;
    Cpvib  = Cvvib;
    Uvib  *= convert;
    Hvib   = Uvib;
    zpe   *= convert;
    const double S_elec = qelec * convert;

    std::cout << "  Multiplicity = q(e) = " << molecule->get_spin() + 1 
              << "\n  S(electronic) / (mEh / K) = "
              << std::setprecision(8) << S_elec << "\n"
              << "  ZPE / Eh = " << zpe / 1000 << "\n";

    std::cout << "\n  **********************************************************************************\n";
    std::cout << "  *                       Thermochemistry @                                        *\n";

    std::cout << "  *                       P / Pa  = " << std::setprecision(3)
              << P_ << " T / K = " << std::setprecision(3) << T_ << "                     *\n";

    std::cout << "  *                                                                                *\n";
    std::cout << "  * Property              Trans          Rot(\u03C3 = "
              << sigma;
              
    (sigma > 9) ? std::cout << ")    Vib            Total       *\n"
                : std::cout << ")     Vib            Total       *\n";
    std::cout << "  **********************************************************************************\n";

    std::cout << "    G  / Eh        = " << std::setw(15) << std::right << std::setprecision(8)
              << 0.001 * (Htran - T_ * Stran) << std::setw(15) 
              << 0.001 * (Urot - T_ * Srot)  << std::setw(15)
              << 0.001 * (Hvib - T_ * Svib)  << std::setw(15) 
              << 0.001 * (Htran  + Urot + Hvib - T_ * (Stran + Srot + Svib)) << '\n';

    std::cout << "    H  / Eh        = " << std::setw(15) << std::right << std::setprecision(8)
              << Htran / 1000 << std::setw(15) << Urot / 1000 << std::setw(15) << Hvib / 1000 
              << std::setw(15) << Htran / 1000 + Urot / 1000 + Hvib / 1000 << '\n';
    
    std::cout << "    U  / Eh        = " << std::setw(15) << std::right << std::setprecision(8) 
              << Utran / 1000 << std::setw(15) << Urot / 1000 << std::setw(15) << Uvib / 1000 
              << std::setw(15) << Utran / 1000 + Urot / 1000 + Uvib / 1000 << '\n';
    
    std::cout << "    S  / (mEh / K) = " << std::setw(15) << std::right << std::setprecision(8) 
              << Stran << std::setw(15) << Srot << std::setw(15) << Svib 
              << std::setw(15) << Stran + Srot + Svib << '\n';
    
    std::cout << "    Cp / (mEh / K) = " << std::setw(15) << std::right << std::setprecision(8) 
              << Cptran << std::setw(15) << Cprot << std::setw(15) << Cpvib
              << std::setw(15) << Cptran + Cprot + Cpvib << '\n';
    
    std::cout << "    Cv / (mEh / K) = " << std::setw(15) << std::right << std::setprecision(8) 
              << Cvtran << std::setw(15) << Cvrot << std::setw(15) << Cvvib
              << std::setw(15) << Cvtran + Cvrot + Cvvib << "\n";
    std::cout << "  **********************************************************************************\n"; 
    // SI
    std::cout << "    G  / (kJ / mol)    = " << std::setw(11) << std::right << std::setprecision(4)
              << 0.001 * (Htran - T_ * Stran) * hartree_to_kjmol << std::setw(15)
              << 0.001 * (Urot - T_ * Srot)  * hartree_to_kjmol << std::setw(15)
              << 0.001 * (Hvib - T_ * Svib)  * hartree_to_kjmol << std::setw(15) 
              << 0.001 * (Htran  + Urot + Hvib - T_ * (Stran + Srot + Svib)) * hartree_to_kjmol << '\n';

    std::cout << "    H  / (kJ / mol)    = " << std::setw(11) << std::right << std::setprecision(4)
              << hartree_to_kjmol * Htran / 1000 << std::setw(15) << hartree_to_kjmol * Urot / 1000 
              << std::setw(15) << hartree_to_kjmol * Hvib / 1000 
              << std::setw(15) << hartree_to_kjmol * (Htran / 1000 + Urot / 1000 + Hvib / 1000) << '\n';
    
    std::cout << "    U  / (kJ / mol)    = " << std::setw(11) << std::right << std::setprecision(4) 
              << Utran / 1000 << std::setw(15) << Urot / 1000 << std::setw(15) << Uvib / 1000 
              << std::setw(15) << Utran / 1000 + Urot / 1000 + Uvib / 1000 << '\n';
    
    std::cout << "    S  / (J / (K mol)) = " << std::setw(11) << std::right << std::setprecision(4) 
              << hartree_to_kjmol * Stran << std::setw(15) << hartree_to_kjmol * Srot << std::setw(15) 
              << hartree_to_kjmol * Svib  << std::setw(15) << hartree_to_kjmol * (Stran + Srot + Svib) << '\n';
    
    std::cout << "    Cp / (J / (K mol)) = " << std::setw(11) << std::right << std::setprecision(4) 
              << hartree_to_kjmol * Cptran << std::setw(15) << hartree_to_kjmol * Cprot 
              << std::setw(15) << hartree_to_kjmol * Cpvib
              << std::setw(15) << hartree_to_kjmol * (Cptran + Cprot + Cpvib) << '\n';
    
    std::cout << "    Cv / (J / (K mol)) = " << std::setw(11) << std::right << std::setprecision(4) 
              << hartree_to_kjmol * Cvtran << std::setw(15) << hartree_to_kjmol * Cvrot << std::setw(15) 
              << hartree_to_kjmol * Cvvib << std::setw(15) << hartree_to_kjmol * (Cvtran + Cvrot + Cvvib) << "\n\n";


    
    std::cout << "\n********************************";
    ("SCF" == hf_settings::get_frequencies_type()) 
    ? std::cout << "\n  Energies summary, Level: SCF" : std::cout << "\n  Energies summary, Level: MP2";
    std::cout << "\n********************************";
    
    ("SCF" == hf_settings::get_frequencies_type()) 
    ? std::cout << "\n  E(SCF) / Eh = "
    : std::cout << "\n  E(MP2) / Eh = ";
    std::cout << std::setprecision(8) << E_electronic;
    
    std::cout << "\n   + ZPE / Eh = " << E_electronic + zpe / 1000;
    std::cout << "\n       U / Eh = " << E_electronic + 0.001 * (Utran  + Urot + Uvib);
    std::cout << "\n       H / Eh = " << E_electronic + 0.001 * (Htran  + Urot + Hvib);
    std::cout << "\n       G / Eh = " 
              << E_electronic + 0.001 * (Htran  + Urot + Hvib - T_ * (Stran + Srot + Svib + S_elec));
              if (molecule->get_spin()) std::cout << " (includes S(electronic))";

    std::cout << "\n\n";
}