#include "hfscf_molecule.hpp"
#include "hfscf_parser.hpp"
#include "hfscf_constants.hpp"
#include "../math/solid_harmonics.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <Eigen/Geometry>
#include <algorithm>
#include "../symmetry/msym_helper.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace BASIS;
using namespace MOLPARSE;
using namespace MOLEC_CONSTANTS;

void MOLEC::Molecule::init_molecule(const std::string& filename)
{
    std::ifstream inputfile(filename, std::ifstream::in);
    std::string line;

	std::getline(inputfile, line);
    comment = line;

	if (comment.substr(0, 1) != "#") // comment line must always be present
	{
		std::cout <<  "  Error: Invalid comment line, expecting \"# string\"\n";
		std::cout <<  "         First line must be a description or comment line.\n";
		exit(1);
	}

	// strip out excess leading spaces
	comment.replace(0, 1, "");
	while (comment.substr(0, 1) == " ") 
		comment.replace(0, 1, "");

	int count = 0;
	// Iterate through the input file and look for keywords. if not found use the defaults.
	do
	{
		std::getline(inputfile, line);

		if('{' == get_keyword(line).c_str()[0])
			break;
		else if(line.empty() || line.substr(0 , 1) == "#")
			continue; // skip empty or comment lines.
		else
		{
			if (get_keyword(line) == "multiplicity") // special case (molecule class member)
			{
				multiplicity = std::stoi(parse_value_from_keyword(line).value());
				spin = multiplicity - 1;
			}
			else if (get_keyword(line) == "charge") // special case (molecule class member)
				charge = std::stoi(parse_value_from_keyword(line).value());
			else
    			parse_value_from_keyword(line); // a HF_SETTING setting

			++count;
		}

		if(count == 1000) // Dho: we got stuck. possible syntax error
		{
			std::cout << "\n\n  Error: Parsing input file failed. Last known line:\n\n";
			std::cout << "         " << line << "\n\n";
			std::cout << "         or missing an opening \"{\" close \"}\" section in the input file.  Aborting.\n";
			exit(EXIT_FAILURE);
		}

	} while(true);

	// sensible post settings based on gradient and frequencies.
	if(hf_settings::get_frequencies_type() == "MP2") hf_settings::set_do_post_scf("MP2");
	//if(hf_settings::get_gradient_type() == "MP2") hf_settings::set_do_post_scf("MP2");

	Keywords::lookup_keyword.clear();
	Keywords::err_mesg.clear();
	Keywords::set_setting_str.clear();
	Keywords::set_setting<double>.clear();
	Keywords::set_setting<bool>.clear();
	Keywords::set_setting<int>.clear();

	int i = 0;
	count = -1; // can't rely on i as a counter because it may get stuck
	bool cartesian_format = true;
	std::unique_ptr<ZMAT::ZtoCart> zmat_ptr;

	double unit_convert = 1.0;
	if (hf_settings::get_unit_type() == "angstrom") 
		unit_convert = angstrom_to_bohr;

	for(;;)
	{
		// Check input file to prevent an endless loop or too many atoms.
		if(count >= 100)
		{	
			std::cout << "\n\n  Error: Maximum number of supported atoms is 100,\n";
			std::cout << "         or missing an opening \"{\" close \"}\" section in the input file.  Aborting.\n\n";
			std::cout << "         last known data: \n\n";
			std::cout << "         ";
			(line.empty()) ? std::cout << "end of file or empty string" : std::cout << line;
			std::cout << '\n';
			exit(EXIT_FAILURE);
		}

		++count; 
		
		std::getline(inputfile, line);
		
		if(line.empty()) // skip empty lines
			continue;
		else if(line.find_first_not_of(" \t") == std::string::npos) // skip if line is just spaces or tabs
			continue;
		else if(line.find_first_of("}") != std::string::npos)
			break;

		if(!line.empty())	
		{
		    if(0 == i) cartesian_format = check_for_cartesian_format(line);

			std::string atom_symbol;

			if (cartesian_format)
		    {
				std::istringstream data(line);

			    std::string x, y, z;
				data >> atom_symbol; // discard
			    geom.conservativeResize(i + 1, 3);
			    data >> x;
				check_coord_is_double(x, line);
			    geom(i, 0) = std::stod(x) * unit_convert;
				data >> y;
				check_coord_is_double(y, line);
			    geom(i, 1) = std::stod(y) * unit_convert;
				data >> z;
				check_coord_is_double(z, line);
			    geom(i, 2) = std::stod(z) * unit_convert;
			}
			else
            {
				std::istringstream data(line);
				std::string r_num;
				std::string angle_num;
				std::string dihedral_num;
				std::string r_ij;
				std::string angle;
				std::string dangle;
				data >> atom_symbol;

            	if(0 == i)
					zmat_ptr = std::make_unique<ZMAT::ZtoCart>(hf_settings::get_unit_type());
			   	
				if(i >= 1)
			   	{
					data >> r_num;
					data >> r_ij;
			   	}
				
				if(i >= 2)
			   	{
					data >> angle_num; 
					data >> angle;
			   	}
				
				if(i >= 3)
				{
					data >> dihedral_num;
					data >> dangle;
				}
                
				zmat_ptr->add_row(atom_symbol, r_num, angle_num, dihedral_num, r_ij, angle, dangle);
            }

            if(atom_symbol != "X") // skip dummy atoms
			{
				const int Z = parse_element(line);
				zval.emplace_back(Z);
				// zval.conservativeResize(zval.size() + 1);
				// zval(zval.size()) = Z;
				nelectrons += Z;
				molecular_mass += ELEMENTDATA::masses[zval[static_cast<size_t>(natoms)] - 1];
				++natoms;
			}

			++i;
		}
	}

	inputfile.close();

	if(0 == natoms)
	{
		std::cout << "\n\n  Error:  No atoms found. Aborting.\n\n";
		exit(EXIT_FAILURE);
	}
    
    if (!cartesian_format)
    {
        zmat_ptr->get_cartesians_from_zarray(geom);
		zmat = zmat_ptr->get_zmatrix();
      	zmat_ptr.reset();
    }    
	
	nelectrons -= charge;

	start_geom = geom;

	do_geometry_analysis();

	std::string path;

	if(hf_settings::get_basis_set_path().length())
		path = hf_settings::get_basis_set_path() + "/" 
		     +  make_file_name_from_basis_set_name(hf_settings::get_basis_set_name());
    else 
		path = path_root + "/basis/" + make_file_name_from_basis_set_name(hf_settings::get_basis_set_name());


	if(false == std::filesystem::exists(path))
	{
		std::cout << "\n\n  Error: Unsupported basis set, or invalid path to Basis set at: ";
		std::cout << path << '\n';
		std::cout << "  Note: You can use the -p flag to specify a custom basis set path if required.\n\n";
        exit(EXIT_FAILURE);
	}
	
    Basisset basisset(path, zval);
	
	// Used by gradient calculations
	MOLEC::mask cmask(0, 0);
	MOLEC::mask smask(0, 0);

    for (i = 0; i < natoms; ++i)
	{
        Vec3D r(geom(i, 0), geom(i, 1), geom(i, 2));
        Atom atom(zval[i], r);
        atoms.emplace_back(atom);

		const auto& gshells = basisset.get_shells(i, r);
		int atom_cart_nbfs = 0;
		int atom_pure_nbfs = 0;
		for (const auto &j : gshells)
		{
			shells.emplace_back(j);
			shell_centers.emplace_back(i);
			atom_cart_nbfs += j.get_cirange();
			atom_pure_nbfs += j.get_sirange();
		}
		// Not used right now, for SAD
		if (hf_settings::get_guess_type() == "SAD")
		{
			atoms[i].set_cart_basis_params(gshells[0].get_idx(), atom_cart_nbfs);
			atoms[i].set_pure_basis_params(gshells[0].get_ids(), atom_pure_nbfs);
		}

		// Used by gradient hessian calculations - cartesian basis
		int cstep = gshells[gshells.size() - 1].get_idx() 
		          + gshells[gshells.size() - 1].get_cirange() - gshells[0].get_idx();
	
		cmask.mask_end += cstep;
		atom_cmask.emplace_back(cmask.mask_start, cmask.mask_end - 1);
		cmask.mask_start += cstep;

		// Used by gradient hessian calculations - spherical basis
		int sstep = gshells[gshells.size() - 1].get_ids() 
		          + gshells[gshells.size() - 1].get_sirange() - gshells[0].get_ids();

		smask.mask_end += sstep;
		atom_smask.emplace_back(smask.mask_start, smask.mask_end - 1);
		smask.mask_start += sstep;
    }

	m_basis_coord_type = basisset.get_basis_coord_type();
	hf_settings::set_basis_coord_type(basisset.get_basis_coord_type());
	const bool is_pure = use_pure_am(); 

    num_gtos = basisset.get_num_gtos();
	num_basis_gtos = basisset.get_num_unique_gtos();
    (is_pure) ? num_orbitals = shells[shells.size() - 1].get_ids() + shells[shells.size() - 1].get_sirange()
	          : num_orbitals = shells[shells.size() - 1].get_idx() + shells[shells.size() - 1].get_cirange();
	num_shells = basisset.get_num_shells();

	create_shell_pairs();
    calc_nuclear_repulsion_energy();

	if(hf_settings::get_geom_opt_write_trajectory() && hf_settings::get_geom_opt().length())
	{
		const auto file = MOLPARSE::set_output_path(filename, ".xyz");
		hf_settings::set_geom_opt_trajectory_file(file);
	}

	if(hf_settings::get_freq_write_molden() && hf_settings::get_frequencies_type().length())
	{
		const auto file = MOLPARSE::set_output_path(filename, ".molden_normal_modes");
		hf_settings::set_freq_molden_file(file);
	}

	// overrides use_symmetry, may remove this flag once symmetry code is solid
	if (is_pure && hf_settings::get_point_group().length()) hf_settings::set_use_symmetry(true);

	if (!use_pure_am() && hf_settings::get_use_symmetry())
	{
		std::cout << "\n  Error: Symmetry support is for pure (spherical) basis sets only, or if pure_am is enabled.\n\n";
		exit(EXIT_FAILURE);
	}

	if (hf_settings::get_use_symmetry() && hf_settings::get_hf_type() == "UHF"
	                                    && hf_settings::get_gradient_type() == "MP2")
	{
		std::cout << "\n  Error: Symmetry enabled UHF wth MP2 numeric gradients not supported.\n\n";
		exit(EXIT_FAILURE);
	}

	if (hf_settings::get_use_symmetry() && hf_settings::get_hf_type() == "UHF"
	                                    && hf_settings::get_frequencies_type().length())
	{
		std::cout << "\n  Error: Symmetry enabled UHF wth numeric frequencies not supported.\n\n";
		exit(EXIT_FAILURE);
	}

	if (hf_settings::get_use_symmetry() && hf_settings::get_hf_type() == "UHF"
	                                    && hf_settings::get_uhf_guess_mix())
	{
		std::cout << "\n  Error: To enable a UHF guess mix symmetry must be disabled.\n\n";
		exit(EXIT_FAILURE);
	}


	if (is_pure && hf_settings::get_use_symmetry())
		do_salc_analysis();
}

void MOLEC::Molecule::calc_nuclear_repulsion_energy()
{
	e_nuc = 0.0;
    for (int i = 0; i < natoms; ++i)
	{
        for (int j = i + 1; j < natoms; ++j)
        {
            double x = (geom(i, 0) - geom(j, 0));
            double y = (geom(i, 1) - geom(j, 1));
            double z = (geom(i, 2) - geom(j, 2));
            e_nuc  += static_cast<double>(zval[static_cast<size_t>(i)]) 
					* static_cast<double>(zval[static_cast<size_t>(j)]) / std::sqrt(x * x + y * y + z * z);
        }
	}
}

void MOLEC::Molecule::update_geom(const Eigen::Ref<const EigenMatrix<double> >& gradient, 
                                  const double stepsize, bool do_geom_analysis)
{
	//Steepest descent
	geom += -stepsize * gradient;

	if(do_geom_analysis) do_geometry_analysis();

	for(int i = 0; i < natoms; ++i)
	{
		Vec3D new_r(geom(i, 0), geom(i, 1), geom(i, 2));
        atoms[i].set_r(new_r);
		
		for(size_t s = 0; s < shells.size(); ++s)
			if (i == shell_centers[s])
				shells[s].set_r(new_r);
	}

	for (auto& k : shpair)
		k.set_params();

	calc_nuclear_repulsion_energy();
}

void MOLEC::Molecule::update_geom(const Eigen::Ref<const EigenMatrix<double> >& gradient_current,
                                  const Eigen::Ref<const EigenMatrix<double> >& gradient_previous,
                                  const double stepsize, bool do_geom_analysis)
{
	// Conjugate cradient Fletcher-Reeve method
	const double gamma = gradient_current.squaredNorm() / gradient_previous.squaredNorm();	
	geom += -stepsize * (gradient_current + gamma * gradient_previous);

	if(do_geom_analysis) do_geometry_analysis();

	for(int i = 0; i < natoms; ++i) // Update basis and atom data
	{
		Vec3D new_r(geom(i, 0), geom(i, 1), geom(i, 2));
        atoms[i].set_r(new_r);
		
		for (size_t s = 0; s < shells.size(); ++s)
			if (i == shell_centers[s])
				shells[s].set_r(new_r);

	}

	for (auto& k : shpair)
		k.set_params();

	calc_nuclear_repulsion_energy();
}

void MOLEC::Molecule::update_geom(const double delta, const int atom_num, const int cart_dir)
{
	geom(atom_num, cart_dir) += delta;

	for(int i = 0; i < natoms; ++i)
	{	
		Vec3D new_r(geom(i, 0), geom(i, 1), geom(i, 2));
        atoms[i].set_r(new_r);
		
		for(size_t s = 0; s < shells.size(); ++s)
			if (i == shell_centers[s])
				shells[s].set_r(new_r);
	}

	for (auto& k : shpair)
		k.set_params();

	calc_nuclear_repulsion_energy();
}

void MOLEC::Molecule::update_geom(const Eigen::Ref<const EigenMatrix<double> >& new_geom, bool do_geom_analysis)
{ 
	geom = new_geom;

	if(do_geom_analysis) do_geometry_analysis();
	
	for(int i = 0; i < natoms; ++i)
	{	
		Vec3D new_r(geom(i, 0), geom(i, 1), geom(i, 2));
        atoms[i].set_r(new_r);
		
		for(size_t s = 0; s < shells.size(); ++s)
			if (i == shell_centers[s])
				shells[s].set_r(new_r);
	}

	for (auto& k : shpair)
		k.set_params();

	calc_nuclear_repulsion_energy();
}

void MOLEC::Molecule::do_geometry_analysis()
{
	std::unique_ptr<MSYM::Msym> msym_ptr = std::make_unique<MSYM::Msym>(natoms);
	msym_ptr->do_symmetry_analysis(geom, zval);
	m_point_group = msym_ptr->get_point_group();
	sigma = msym_ptr->get_sigma();
	m_sym_deviation = msym_ptr->get_sym_deviation();

	// Set center of charge
	double qx_center = 0.0;
	double qy_center = 0.0;
	double qz_center = 0.0;
	double total_charge = 0;

	for (int i = 0; i < natoms; ++i)
	{
		double nuc_charge = zval[i];
		total_charge += zval[i];
		qx_center += nuc_charge * geom(i, 0);
		qy_center += nuc_charge * geom(i, 1);
		qz_center += nuc_charge * geom(i, 2);
	}
	
	qx_center /= total_charge;
	qy_center /= total_charge;
	qz_center /= total_charge;

	Q_c = Vec3D(qx_center, qy_center, qz_center);
	
	// We could get this from libmsym - todo
	// msym will already center the molecule
	// compute center of mass
	double x_com = 0.0;
	double y_com = 0.0;
	double z_com = 0.0;

	for (int i = 0; i < natoms; ++i)
	{
		double m_i = ELEMENTDATA::masses[zval[i] - 1];
		x_com += m_i * geom(i, 0);
		y_com += m_i * geom(i, 1);
		z_com += m_i * geom(i, 2);
	}

	x_com /= molecular_mass;
	y_com /= molecular_mass;
	z_com /= molecular_mass;

	center_of_mass = Vec3D(x_com, y_com, z_com);

	i_tensor.setZero();

	for(int i = 0; i < natoms; ++i) 
	{
    	double m_i = ELEMENTDATA::masses[zval[i] - 1];

		i_tensor(0, 0) += m_i * (geom(i, 1) * geom(i, 1) + geom(i, 2) * geom(i, 2));
		i_tensor(1, 1) += m_i * (geom(i, 0) * geom(i, 0) + geom(i, 2) * geom(i, 2));
		i_tensor(2, 2) += m_i * (geom(i, 0) * geom(i, 0) + geom(i, 1) * geom(i, 1));
		i_tensor(0, 1) += m_i * geom(i, 0) * geom(i, 1);
		i_tensor(0, 2) += m_i * geom(i, 0) * geom(i, 2);
		i_tensor(1, 2) += m_i * geom(i, 1) * geom(i, 2);
  	}
 
	i_tensor(1, 0) = i_tensor(0, 1);
	i_tensor(2, 0) = i_tensor(0, 2);
	i_tensor(2, 1) = i_tensor(1, 2);

	Eigen::SelfAdjointEigenSolver<InertiaTensor> solver(i_tensor);
	i_tensor_evals = solver.eigenvalues();
	i_tensor_evecs = solver.eigenvectors();
	(i_tensor_evals(0) > i_tensor_epsilon) ? Be(0) = convert_to_wavenumber / i_tensor_evals(0) : 0;
	(i_tensor_evals(1) > i_tensor_epsilon) ? Be(1) = convert_to_wavenumber / i_tensor_evals(1) : 0;
	(i_tensor_evals(2) > i_tensor_epsilon) ? Be(2) = convert_to_wavenumber / i_tensor_evals(2) : 0;

	calc_nuclear_repulsion_energy();
}

const std::string& MOLEC::Molecule::get_point_group() const
{
	return m_point_group;
}

int MOLEC::Molecule::get_sigma()
{
	return sigma;
}

double MOLEC::Molecule::get_mass() const 
{
	return molecular_mass;
}

bool MOLEC::Molecule::molecule_is_linear() const 
{ 
	return islinear;
}

const std::vector<MOLEC::mask>& MOLEC::Molecule::get_atom_mask() const 
{
	return atom_cmask;
}

const std::vector<MOLEC::mask>& MOLEC::Molecule::get_atom_spherical_mask() const 
{
	return atom_smask;
}

double MOLEC::Molecule::get_enuc() const 
{ 
	return e_nuc;
}

const Eigen::Ref<const Vec3D> MOLEC::Molecule::get_center_of_charge_vector() const 
{ 
	return Q_c;
}

const Eigen::Ref<const Vec3D> MOLEC::Molecule::get_center_of_mass_vector() const 
{ 
	return center_of_mass;
}

const Eigen::Ref<const Vec3D> MOLEC::Molecule::get_rotational_constants() const 
{ 
	return Be;
}

Eigen::Index MOLEC::Molecule::get_num_electrons() const
{ 
	return nelectrons;
}

Eigen::Index MOLEC::Molecule::get_num_orbitals() const 
{ 
	return num_orbitals;
}

Eigen::Index  MOLEC::Molecule::get_num_cart_orbitals() const
{
	return shells[shells.size() - 1].get_idx() + shells[shells.size() - 1].get_cirange();
}

Eigen::Index MOLEC::Molecule::get_num_shells() const 
{ 
	return num_shells;
}

const std::vector<int>& MOLEC::Molecule::get_z_values() const 
{ 
	return zval;
}

Eigen::Index MOLEC::Molecule::get_num_gtos() const 
{ 
	return num_gtos;
}

const ShellVector& MOLEC::Molecule::get_shells() const
{
	return shells;
}

const ShellPairVector& MOLEC::Molecule::get_shell_pairs() const
{
	return shpair;
}

Eigen::Index MOLEC::Molecule::get_spin() const 
{
	return spin;
}

Eigen::Index MOLEC::Molecule::get_multiplicity() const 
{
	return multiplicity;
}

using Eigen::Index;
Eigen::Index MOLEC::Molecule::get_nalpha() const
{
	Index nalpha = static_cast<Index>(std::floor(0.5 * static_cast<double>(nelectrons + get_multiplicity() - 1)));
    Index nbeta = nelectrons - nalpha;

	if (nbeta > nalpha) std::swap(nalpha, nbeta);

	return nalpha;
}

Eigen::Index MOLEC::Molecule::get_nbeta() const
{
	Index nalpha = static_cast<int>(std::floor(0.5 * static_cast<double>(nelectrons + get_multiplicity() - 1)));
    Index nbeta = nelectrons - nalpha;

	if (nbeta > nalpha) std::swap(nalpha, nbeta);

	return nbeta;
}

const std::vector<MOLEC::Atom>& MOLEC::Molecule::get_atoms() const 
{ 
	return atoms; 
}

const Eigen::Ref<const EigenMatrix<double> > MOLEC::Molecule::get_geom() const 
{
	return geom; 
}

EigenMatrix<double> MOLEC::Molecule::get_geom_copy() const 
{
	return geom;
}

void MOLEC::Molecule::create_shell_pairs()
{	
	shpair.reserve(shells.size() * shells.size());
	for(const auto &i : shells)
		for(const auto &j : shells)
		{
			shpair.emplace_back(ShellPair(i, j));
		}
}

bool MOLEC::Molecule::use_pure_am() const
{
	bool is_pure = m_basis_coord_type != "cartesian";

	if (hf_settings::get_use_pure_angular_momentum().has_value())
		is_pure = hf_settings::get_use_pure_angular_momentum().value(); // optional override
	
	return is_pure;
}

void MOLEC::Molecule::do_salc_analysis()
{
    std::unique_ptr<MSYM::Msym> msym_ptr = std::make_unique<MSYM::Msym>(natoms);
    int ret = msym_ptr->build_salcs(geom, zval, shells, shell_centers);

    if (ret)
	{
        std::cout << "\n  Error: Failure in SALC determination.\n";
        exit(EXIT_FAILURE);
    }

    symblocks = msym_ptr->get_sym_blocks();
    sym_species = msym_ptr->get_sym_species();
    irrep_sizes = msym_ptr->get_irrep_sizes();
    m_sub_group = msym_ptr->get_sub_group();
}

// using Eigen::Index;
// void MOLEC::Molecule::create_spherical_trans_form()
// {
// 	Index cartdim = shells[shells.size() - 1].get_idx() + shells[shells.size() - 1].get_cirange();
// 	strans = EigenMatrix<double>::Zero(cartdim, num_orbitals); // (l + 1) x (l + 2) / 2 by (2l + 1) 

// 	for (const auto& i : shells)
// 	{
// 		const Eigen::Ref<const EigenMatrix<double>> Ylmn = i.get_spherical_form();

// 		for (int k = 0; k < Ylmn.rows(); ++k)
// 			for (int l = 0; l < Ylmn.cols(); ++l)
// 				strans(i.get_idx() + k, i.get_ids() + l) = Ylmn(k, l);
// 	}
// }