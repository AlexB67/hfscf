#include "hfscf_molecule.hpp"
#include "hfscf_elements.hpp"
#include "../settings/hfscf_settings.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <Eigen/Geometry>
#include <algorithm>
#include "hfscf_constants.hpp"
 
using namespace HF_SETTINGS;
using namespace MOLEC_CONSTANTS;
using Eigen::Index;

void MOLEC::Molecule::print_info(const bool post_geom_opt)
{	
	std::cout << "**************************************************************************\n";
    std::cout << "  Input parameters\n";
    std::cout << "**************************************************************************\n";
	std::cout << "  Comment line               = " << comment << "\n";
	std::cout << "  Eigen Max Threads          = " << Eigen::nbThreads() << '\n';
	
	#ifdef HAS_OPENMP
	std::cout << "  OMP Max threads            = " << omp_get_max_threads() << '\n';
	std::cout << "  OMP Max processes          = " << omp_get_num_procs() << '\n'; 
	#endif
	
	std::cout << "  SCF type                   = " << hf_settings::get_hf_type() << '\n';
	std::cout << "  Initial guess method       = " << hf_settings::get_guess_type() << '\n';

	if (hf_settings::get_guess_type() == "SAD")
	{
		std::cout << "  Max SAD SCF iterations     = " << hf_settings::get_max_sad_iterations() << '\n';
		std::cout << "  SAD SCF Energy tolerance   = " <<
		std::setprecision(1) << std::scientific << hf_settings::get_sad_energy_tol() << '\n';
		std::cout << "  SAD SCF RMS tolerance      = " <<
		std::setprecision(1) << std::scientific << hf_settings::get_sad_rms_tol() << '\n';
	}

	std::cout << "  Max SCF iterations         = " << hf_settings::get_max_scf_iterations() << '\n';
	std::cout << "  SCF RMS tolerance          = " <<
	std::setprecision(1) << std::scientific << hf_settings::get_rms_tol() << '\n';
	std::cout << "  SCF Energy tolerance       = " << hf_settings::get_energy_tol() << '\n';
	
	if(0 == hf_settings::get_diis_size())
		std::cout << "  DIIS                       = off" << '\n';
	else
	    std::cout << "  DIIS range                 = " << hf_settings::get_diis_size() << '\n';
	
	
	if (hf_settings::get_soscf())
	{
		std::cout << "  SOSCF                      = on\n";
		std::cout << "  SOSCF max micro iter       = " << hf_settings::get_max_soscf_iterations() << "\n";
		std::cout << "  SOSCF RMS tol              = "
		          <<  std::scientific << std::setprecision(1) << hf_settings::get_soscf_rms_tol() << "\n";
	}

	std::string sym_geom = "false";
	if (hf_settings::get_symmetrize_geom()) sym_geom = "true";
	std::cout << "  Symmetrize geometry        = " << sym_geom <<'\n';
	// if (hf_settings::get_point_group_equivalence_threshold() != "DEFAULT")
	std::cout << "  Point group equivalence    = " 
		      << hf_settings::get_point_group_equivalence_threshold() << '\n';
			  
	std::cout << "  Use point group symmetry   = ";
	(hf_settings::get_use_symmetry()) ? std::cout << "true\n" : std::cout << "false\n";

	if(hf_settings::get_geom_opt().length())
	{
		std::cout << "  Geometry optimization      = " << hf_settings::get_geom_opt() << "\n";
		std::cout << "  Geomopt method             = " << hf_settings::get_geom_opt_algorithm() << "\n";
		std::cout << "  Geomopt tolerance          = " << hf_settings::get_geom_opt_tol() << '\n';

		if (hf_settings::get_geom_opt_write_trajectory())
			std::cout << "  Geomopt write xyz file     = true\n";

		if(hf_settings::get_geom_opt_algorithm() == "RFO")
		{
			std::cout << "  Geomopt RFO stepsize       = automatic\n";
			std::cout << "  Geomopt guess Hessian      = " << hf_settings::get_geom_opt_guess_hessian() << "\n";
		}
		else
			std::cout << "  Geomopt stepsize           = "  <<  hf_settings::get_geom_opt_stepsize() << "\n";
	}

	if(hf_settings::get_hf_type() == "UHF")
	{
		std::cout << "  UHF symmetry breaking      = ";
		if(hf_settings::get_uhf_guess_mix())
			std::cout << "on\n";
		else
		std::cout << "off\n";
	}
    
	std::string post_scf = hf_settings::get_post_scf_type();
	if (post_scf.length())
		std::cout << "  Post SCF method            = " << post_scf << '\n';

	std::string post_ci = hf_settings::get_ci_type();
	if (post_ci.length())
		std::cout << "  CI method                  = " << post_ci << '\n';

	if (hf_settings::get_post_scf_type().length())
	{
		std::cout << "  Freeze core                = ";

		if(hf_settings::get_freeze_core())
			std::cout << "true\n";
		else
			std::cout << "false\n";	
	}

	if(post_scf.substr(0, 4) == "CCSD")
	{
		std::cout << "  CCSD DIIS range            = ";
		(hf_settings::get_ccsd_diis_size() == 0) ? std::cout << "off" : std::cout << hf_settings::get_ccsd_diis_size();
		std::cout << '\n';
		std::cout << "  CCSD Max iterations        = " << hf_settings::get_max_ccsd_iterations() << "\n";
		std::cout << "  CCSD Max RMS               = "
		          << std::setprecision(2) << std::scientific << hf_settings::get_ccsd_rms_tol() << "\n";
		std::cout << "  CCSD energy tolerance      = " 
		          << std::setprecision(2) << std::scientific << hf_settings::get_ccsd_energy_tol() << "\n";
	}

	if(hf_settings::get_gradient_type().length())
	{
		std::cout << "  Gradient type              = " <<  hf_settings::get_gradient_type() << '\n';
		std::cout << "  Gradient method            = ";
		if("MP2" == hf_settings::get_gradient_type() && "UHF" == hf_settings::get_hf_type()) std::cout << "numeric\n";
		else std::cout << "analytic\n";
	}

	if(hf_settings::get_do_mol_props())
	{
		std::cout << "  Population analysis        = true\n";
		std::cout << "  Dipole moment              = true\n";
		std::cout << "  Quadrupole moment          = true\n";

		("RHF" == hf_settings::get_hf_type()) 
		? std::cout << "  static polarizabilities    = true\n"
		: std::cout << "  static polarizabilities    = false\n";

		if (hf_settings::get_molprops_cphf_iter() && "RHF" == hf_settings::get_hf_type()) 
			std::cout << "  CPHF method                = iterative\n";
		else if (!hf_settings::get_molprops_cphf_iter() && "RHF" == hf_settings::get_hf_type())
			std::cout << "  CPHF method                = direct\n";
	}
	
	if(hf_settings::get_frequencies_type().length())
	{
		std::cout << "  Frequency analysis         = " << hf_settings::get_frequencies_type() << '\n';

		if ("UHF" == hf_settings::get_hf_type() && "MP2" == hf_settings::get_frequencies_type())
		{
			std::cout << "  Hessian 2nd derivative     = Numeric from SCF-MP2 energy\n";
			std::cout << "  Hessian energy stepsize    = " << std::scientific << std::setprecision(2) <<
			hf_settings::get_hessian_step_from_energy() << "\n";
		}
		else
		{
			if ("RHF" == hf_settings::get_hf_type() && hf_settings::get_frequencies_type().length())
				std::cout << "  Hessian 2nd derivative     = analytic\n";
			else
				std::cout << "  Hessian 2nd derivative     = numeric\n"
				          << "  Hessian gradient stepsize  = " << std::scientific << std::setprecision(2) <<
				hf_settings::get_hessian_step_from_grad() << "\n";
		}

		if (hf_settings::get_freq_write_molden())
			std::cout << "  Write Molden file          = true\n";
	}
	
	std::cout << "  Print level verbosity      = " << hf_settings::get_verbosity() << '\n';

	std::string scf_direct;
	(hf_settings::get_scf_direct()) ? scf_direct = "true" : scf_direct = "false";
	std::cout << "  SCF Direct                 = " << scf_direct << '\n';

	std::cout << "  SCF Density damping        = "; 
	(std::fabs(hf_settings::get_scf_damping_factor()) > 0.9)? std::cout << "off\n" 
								   							: std::cout << std::setprecision(2) 
															<< hf_settings::get_scf_damping_factor() << '\n';
	
	if (hf_settings::get_scf_direct() || hf_settings::get_screen_eri())
	{
		std::cout << "  Screen eri                 = true\n";
		// To bring back
		// std::cout << "  Cauchy Schwartz cutoff     = " << std::scientific 
		//           << std::setprecision(1) << 100 * hf_settings::get_integral_tol() << '\n';
	}
	else
		std::cout << "  Screen eri                 = false\n";

	std::cout << "  Integral cutoff tolerance  = " << hf_settings::get_integral_tol() << '\n';
	std::cout << "  Basis set                  = " << hf_settings::get_basis_set_name() << '\n';
	std::cout << "  Basis set type             = " << hf_settings::get_basis_coord_type() << '\n';
	std::cout << "  Use pure angular momentum  = ";
	
	if (!use_pure_am()) std::cout <<  "false\n";
	else  std::cout <<  "true\n";

	std::cout << "  Number of basis primitives = " << num_basis_gtos << '\n';
	//std::cout << "  Number of actual GTOs      = " << num_gtos << '\n';
	std::cout << "  Cartesian basis functions  = " <<
	shells[shells.size() - 1].get_idx() + shells[shells.size() - 1].get_cirange() << '\n';

	if (use_pure_am() || hf_settings::get_basis_coord_type() == "spherical")
		std::cout << "  Spherical basis functions  = " << num_orbitals << '\n';

	std::cout << "  Number of shells           = " << num_shells << '\n';
	std::cout << "  Number of electrons        = " << nelectrons << '\n';

	if ("UHF" == hf_settings::get_hf_type())
	{
		Index alpha_size, beta_size;
	    alpha_size =  static_cast<Index>(std::floor(0.5 * static_cast<double>(nelectrons + spin)));
	    beta_size = nelectrons - alpha_size;

		if(beta_size > alpha_size) std::swap(alpha_size, beta_size);

		std::cout << "  Alpha electrons            = " << alpha_size << '\n';
		std::cout << "  Beta  electrons            = " << beta_size  << '\n';
		std::cout << "  Doubly occupied            = " << beta_size  << '\n';
		std::cout << "  Singly occupied            = " << alpha_size - beta_size << '\n';
	}
	if ("RHF" == hf_settings::get_hf_type()) 
	{
		std::cout << "  Doubly occupied            = " << nelectrons / 2  << '\n';
		std::cout << "  Singly occupied            = 0\n";
	}

	std::cout << "  Charge                     = " << charge << '\n';
	std::cout << "  Multiplicity (2S + 1)      = " << spin + 1; 
	
	if(0 == spin) std::cout << " (closed shell)" << '\n';
	else std::cout << " (open shell)" << '\n';

	if (hf_settings::get_verbosity() > 1)
	{
		std::cout << "\n*************************************************************************\n";
		write_basis_set_info();
	}

	if(!post_geom_opt)
	{
		if(zmat.size())
		{
			std::cout << "\n*************************************************************************\n";
			std::cout << "  Input type: Z matrix (internal coordinates) detected.\n";
			std::cout << "  Input Z matrix:\n\n";
			std::cout << "  Atom  #    r_ij / angstrom";
			
			if(natoms > 2)
				std::cout << "   #  theta / deg";

			if(natoms > 3) 
				std::cout << "         #  phi / deg";

			std::cout << '\n';
			std::cout << "*************************************************************************\n";

			double convert = 1.0;
			if(hf_settings::get_unit_type() == "bohr") // Will always display Z matrix in angstroms
				convert = bohr_to_angstrom;

			for(size_t i = 0; i < zmat.size(); ++i)
			{
				(zmat[i].atom_name.length() == 1) ? std::cout << "   " : std::cout << "  ";
				std::cout << zmat[i].atom_name;
				
				if(i >= 1)
				{
					std::cout << std::right << std::setw(5) << zmat[i].r_ij_num + 1;
					std::cout << std::right << std::fixed << std::setprecision(12) 
							<< std::setw(18) << zmat[i].r_ij * convert;
				}
				
				if(i >= 2)
				{
					std::cout << std::right << std::setw(5) << zmat[i].angle_num + 1;
					std::cout << std::right << std::fixed << std::setprecision(12) 
							<< std::setw(18) << zmat[i].angle * ZMAT::rad_to_deg;
				}
				
				if (i >= 3)
				{
					std::cout << std::right << std::setw(5) << zmat[i].dihedral_num + 1;
					std::cout << std::right << std::fixed << std::setprecision(12) << std::setw(18) 
							<< zmat[i].dihedral_angle * ZMAT::rad_to_deg;
				}		

				std::cout << '\n';
			}
		}
		else
		{
			std::cout << "\n***********************************************************************\n";
			std::cout << "  Input type: Cartesian coordinates detected.\n"; 
			std::cout << "  Input coordinates:\n\n";
			std::cout << "  Atom  X / bohr          Y / bohr          Z / bohr          Mass\n";
			std::cout << "***********************************************************************\n";

			for(size_t i = 0; i < zval.size(); ++i)
			{
				std::cout << "  " << ELEMENTDATA::atom_names[zval[i] - 1];
				std::cout << std::right << std::fixed << std::setprecision(12) << std::setw(18) << start_geom(i, 0)
						<< std::right << std::fixed << std::setw(18) << start_geom(i, 1) 
						<< std::right << std::fixed << std::setw(18) << start_geom(i, 2)
						<< std::right << std::fixed << std::setw(13) << std::setprecision(7) 
						<< ELEMENTDATA::masses[zval[i] - 1] << '\n';
			}
		}
		
		start_geom.resize(0, 0);
	}

	std::cout << "**************************************************************************\n";

    std::cout << "\n  ___Geometry analysis___\n\n";
	std::cout << "  Number of atoms: " << natoms << '\n';
	
	if (m_point_group.length() == 0)
	{
		m_point_group = "C1";
		if (!hf_settings::get_use_symmetry()) m_sub_group = "none";
	}
	
	std::cout << "  Full Point group: " << m_point_group << '\n';

	if (hf_settings::get_use_symmetry())
		std::cout << "  Using Point group: " << m_sub_group << '\n';
	else
		std::cout << "  Using Point group: none\n";

	if (hf_settings::get_symmetrize_geom()) 
		std::cout << "  Geometry symmetrized with deviation: " << m_sym_deviation << '\n';

	std::cout << "  Relative molecular mass: " << std::right << std::fixed 
	          << std::setprecision(7) << molecular_mass << "\n";
	std::cout << "  COM Cartesian coordinates";

	if (isaligned) 
		std::cout <<  ", aligned to principal axes:\n\n";
	else
		std::cout << ":\n\n";
	
	std::cout << "  Atom  X / bohr          Y / bohr          Z / bohr          Mass\n";
	std::cout << "***********************************************************************\n";
    
	for(size_t i = 0; i < zval.size(); ++i)
	{
		std::cout << "  " << ELEMENTDATA::atom_names[zval[i] - 1];
		std::cout << std::right << std::fixed << std::setprecision(12) << std::setw(18) << geom(i, 0)
				  << std::right << std::fixed << std::setw(18) << geom(i, 1)
				  << std::right << std::fixed << std::setw(18) << geom(i, 2)
				  << std::right << std::fixed << std::setw(13) << std::setprecision(7) 
				  << ELEMENTDATA::masses[zval[i] - 1] << '\n';
	}
	
	if(natoms == 1) return;
	
	std::cout << "\n  ___Rotor analysis___\n";
	std::cout << "\n  *********************************************************************\n";
	std::cout << "  *             Princpial Moments of inertia / amu bohr^2             *\n";
	std::cout << "  *********************************************************************\n";
	std::cout << "   Ia(e) = ";
	std::cout << std::fixed << std::setprecision(8) << i_tensor_evals[0];
	std::cout << "  Ib(e) = ";
	std::cout << std::fixed << std::setprecision(8) << i_tensor_evals[1];
	std::cout << "  Ic(e) = "; 
	std::cout << std::fixed << std::setprecision(8) << i_tensor_evals[2] << '\n';
	
	std::cout << "\n  *********************************************************************\n";
	std::cout << "  *           Princpial Moments of inertia / amu Angstrom^2           *\n";
	std::cout << "  *********************************************************************\n";
	std::cout << "   Ia(e) = ";
	std::cout << std::fixed << std::setprecision(8)
			  << bohr_to_angstrom * bohr_to_angstrom * i_tensor_evals(0);
	std::cout << "  Ib(e) = ";
	std::cout << std::fixed << std::setprecision(8)
			  << bohr_to_angstrom * bohr_to_angstrom * i_tensor_evals(1);
	std::cout << "  Ic(e) = ";
	std::cout << std::fixed << std::setprecision(8)
			  << bohr_to_angstrom * bohr_to_angstrom * i_tensor_evals(2) << "\n";

	std::cout << "\n  *********************************************************************\n";
	std::cout << "  *                   Rotational constants / MHz                      *\n";
	std::cout << "  *********************************************************************\n";
	std::cout << "   A(e) = ";
					if (i_tensor_evals(0) < i_tensor_epsilon)  
						std::cout << std::fixed << std::setprecision(4) << 0.0; 
					else 
						std::cout << std::fixed << std::setprecision(4) 
								  << convert_to_hertz / i_tensor_evals(0); 
					
	std::cout << "  B(e) = "; 
					if (i_tensor_evals(1) < i_tensor_epsilon)  
						std::cout << std::fixed << std::setprecision(4) << 0.0;  
					else 
						std::cout << std::fixed << std::setprecision(4) 
								  << convert_to_hertz / i_tensor_evals(1); 

	std::cout << "  C(e) = "; 
					if (i_tensor_evals(2) < i_tensor_epsilon)  
						std::cout << std::fixed << std::setprecision(4) << 0.0; 
					else 
						std::cout << std::fixed << std::setprecision(4) 
						          << convert_to_hertz / i_tensor_evals(2); 


	std::cout << "\n\n  *********************************************************************\n";
	std::cout << "  *                     Rotational constants / cm^-1                  *\n";
	std::cout << "  *********************************************************************\n";
	std::cout << "   A(e) = ";
					if (i_tensor_evals(0) < i_tensor_epsilon)  
						std::cout << std::fixed << std::setprecision(8); 
					else
						std::cout << std::fixed << std::setprecision(8) 
								  << convert_to_wavenumber / i_tensor_evals(0); 
					
	std::cout << "  B(e) = "; 
					if (i_tensor_evals(1) < i_tensor_epsilon)  
						std::cout << std::fixed << std::setprecision(8) << 0.0; 
					else 
						std::cout << std::fixed << std::setprecision(8) 
								  << convert_to_wavenumber / i_tensor_evals(1); 
	std::cout << "  C(e) = "; 
					if (i_tensor_evals(2) < i_tensor_epsilon)
						std::cout << std::fixed << std::setprecision(8) << 0.0; 
					else 
						std::cout << std::fixed << std::setprecision(8) 
								  << convert_to_wavenumber / i_tensor_evals(2); 
	

	if(zval.size() == 2)
	{ 
		std::cout << "\n\n  Molecule is a diatomic.\n";
		islinear = true;
	}
  	else if(i_tensor_evals(0) <  i_tensor_epsilon) 
	{
	  std::cout << "\n\n  Molecule is linear.\n";
	  islinear = true;
	}
	else if ((  std::fabs(i_tensor_evals(0) - i_tensor_evals(1)) <  i_tensor_epsilon) 
			&& (std::fabs(i_tensor_evals(1) - i_tensor_evals(2)) <  i_tensor_epsilon))
	{
		std::cout << "\n\n  Molecule is a spherical top.\n";
	}
	else if(   (std::fabs(i_tensor_evals(0) - i_tensor_evals(1)) <  i_tensor_epsilon) 
			&& (std::fabs(i_tensor_evals(1) - i_tensor_evals(2)) >  i_tensor_epsilon))
	{
		const double kappa = (2.0 * i_tensor_evals(1) -   i_tensor_evals(0) -  i_tensor_evals(2)) 
						   / ( i_tensor_evals(0) -  i_tensor_evals(2));

		std::cout << "\n\n  Molecule is an oblate symmetric top. kappa = " << kappa << "\n";
	}
	else if(   (std::fabs(i_tensor_evals(0) - i_tensor_evals(1))  >  i_tensor_epsilon) 
			&& (std::fabs(i_tensor_evals(1) - i_tensor_evals(2))  <  i_tensor_epsilon))
	{
		const double kappa = (2.0 * i_tensor_evals(1) -   i_tensor_evals(0) -  i_tensor_evals(2)) 
						   / ( i_tensor_evals(0) -  i_tensor_evals(2));

		std::cout << "\n\n  Molecule is a prolate symmetric top. kappa = " << kappa << "\n";
	}
	else
	{
		const double kappa = (2.0 * i_tensor_evals(1) -   i_tensor_evals(0) -  i_tensor_evals(2)) 
						   / ( i_tensor_evals(0) -  i_tensor_evals(2));

		std::cout << "\n\n  Molecule is an asymmetric top. kappa = " << kappa << "\n";
	}
}

using Eigen::Index;
void MOLEC::Molecule::write_basis_set_info() const
{
	const std::array Lidx = {'S', 'P', 'D', 'F', 'G', 'H', 'I'};

	std::cout << "  Using basis-set data from " << hf_settings::get_basis_set_name() << "\n";
	std::cout << "  " << hf_settings::get_basis_coord_type() << "\n";


	for (size_t atom = 0; atom < atoms.size(); ++atom)
	{
		// skip duplicate atoms
		repeat:
		for (size_t at = 0; at < atom; ++at)
		{
			if (at == static_cast<size_t>(natoms) - 1) return;
			else if (zval[at] == zval[atom]){++atom; goto repeat;} // note: unsorted atoms so we repeat
		}

		std::cout << "  ***\n";
		std::cout << " " << ELEMENTDATA::atom_names[zval[atom] - 1] << "\n";

		for(size_t s = 0; s < shells.size(); ++s)
		{
			if (atom == static_cast<size_t>(shell_centers[s]))
			{
				std::cout << "  " << Lidx[shells[s].L()] << "  " << shells[s].c().size() << "\n";
                for (int k = 0; k < shells[s].c().size(); ++k) 
				{
                    std::cout << std::setw(20) << std::scientific 
					          << std::setprecision(9) << shells[s].alpha()(k)
                              << std::setw(20) << std::scientific 
							  << std::setprecision(9) << shells[s].c_unscaled()(k) << "\n";
                }
			}
		}
	}

	return;
}
