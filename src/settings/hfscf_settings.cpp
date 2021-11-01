#include "hfscf_settings.hpp"
#include <cmath>
#include <iostream>

using namespace HF_SETTINGS;

void hf_settings::set_basis_set_name(const std::string& basis_name)
{
    m_basis_set_name = basis_name;
}

void hf_settings::set_basis_set_path(const std::string& basis_set_path)
{
    m_basis_set_path = basis_set_path;
}

void hf_settings::set_basis_coord_type(const std::string& basis_coord_type)
{
    m_basis_coord_type = basis_coord_type;
}

void hf_settings::set_hf_type(const std::string& hf_type)
{
    m_hf_type = hf_type;
}

void hf_settings::set_guess_type(const std::string& inital_guess)     
{
    m_initial_guess = inital_guess;
}

void hf_settings::set_geom_opt_guess_hessian(const std::string& guess_hessian)     
{
    m_geom_opt_guess_hessian = guess_hessian;
}

void hf_settings::set_unit_type(const std::string& unit)     
{
    m_unit_type = unit;

    if (unit != "bohr" && unit != "angstrom")
    {
        std::clog << "  Error: Invalid unit " << unit << "\n";
        exit(EXIT_FAILURE);
    }
}

void hf_settings::set_gradient_type(const std::string& gradient_type)     
{
    m_gradient_type = gradient_type;
}

void hf_settings::set_ci_type(const std::string& ci_type)     
{
    m_ci_type = ci_type;
}

void hf_settings::set_point_group_equivalence_threshold(const std::string& equivalence_threshold)
{
    m_point_group_equivalence_threshold = equivalence_threshold;
}

void hf_settings::set_point_group(const std::string& group)
{
    m_point_group = group;
}

void hf_settings::set_scf_damping_factor(const double scf_damping)    
{
    if (std::fabs(scf_damping) < 0.2) 
    {
        m_scf_damping = 0.2;
        std::cout << "\n  Warning: damping < 0.2, damping set to 0.2\n";
    }
    else if(std::fabs(scf_damping) > 1.0)
    {
         m_scf_damping = 1.0;
        std::cout << "\n  Warning: damping > 1.0, damping set to 1.0. i.e no damping.\n";
    }
    else
        m_scf_damping = fabs(scf_damping);
}

void hf_settings::set_diis_size(const int diis_size)
{
    if (abs(diis_size) > 8)
    {
        std::cout << "  Warning:  Max DIIS range adjuted to 8.\n";
        m_diis_size = 8;
    }
    else
        m_diis_size = abs(diis_size);
}

void hf_settings::set_ccsd_diis_size(const int diis_size)
{
    if (abs(diis_size) > 6)
    {
        std::cout << "\n  Warning:  Max CCSD DIIS range adjuted to 6.\n";
        m_ccsd_diis_size = 6;
    }
    else
        m_ccsd_diis_size = abs(diis_size);
}

void hf_settings::set_max_scf_iterations(const int max_scf_iter)
{
    m_max_scf_iter = abs(max_scf_iter);
}

void hf_settings::set_max_ccsd_iterations(const int max_ccsd_iter)
{
    m_max_ccsd_iter = abs(max_ccsd_iter);
}

void hf_settings::set_max_sad_iterations(const int max_sad_iter)
{
    m_max_sad_iter = abs(max_sad_iter);
}


void hf_settings::set_max_soscf_iterations(const int max_soscf_iter)
{
    m_max_soscf_iter = abs(max_soscf_iter);

    if (m_max_soscf_iter < 3) m_max_soscf_iter = 3;
    else if (m_max_soscf_iter > 6) m_max_soscf_iter = 6;
}

void hf_settings::set_max_geomopt_iterations(const int max_geomopt_iter)
{
    m_max_geomopt_iter = abs(max_geomopt_iter);
}

void hf_settings::set_rms_tol(const double rms_tol) 
{ 
    if (fabs(rms_tol) > 1e-5)
    {
        std::cout << "  Warning: SCF RMS tolerance too large. Adjusted to 1E-5\n";
        m_rms_tol = 1e-5;
    }
    else
        m_rms_tol = fabs(rms_tol); 
}

void hf_settings::set_ccsd_rms_tol(const double ccsd_rms_tol) 
{ 
    if (fabs(ccsd_rms_tol) > 1e-6)
    {
        std::cout << "  Warning: SCF RMS tolerance too large. Adjusted to 1E-5\n";
        m_ccsd_rms_tol = 1e-6;
    }
    else
        m_ccsd_rms_tol = fabs(ccsd_rms_tol); 
}

void hf_settings::set_sad_rms_tol(const double sad_rms_tol) 
{ 
    if (fabs(sad_rms_tol) > 1e-5)
    {
        std::cout << "  Warning: SAD SCF RMS tolerance too large. Adjusted to 1E-5\n";
        m_sad_rms_tol = 1e-5;
    }
    else
        m_sad_rms_tol = fabs(sad_rms_tol);
}

void hf_settings::set_soscf_rms_tol(const double soscf_rms_tol) 
{ 
    if (fabs(soscf_rms_tol) > 1e-2)
    {
        std::cout << "  Warning: SOSCF RMS tolerance too large. Adjusted to 1E-2\n";
        m_soscf_rms_tol = 1e-2;
    }
    else
        m_soscf_rms_tol = fabs(soscf_rms_tol);
    
    if (m_soscf_rms_tol < 1e-04)  m_soscf_rms_tol = 1e-04;
}

void hf_settings::set_energy_tol(const double energy_tol) 
{ 
    if (fabs( energy_tol) > 1e-6) 
    {
        std::cout << "\n  Warning: SCF energy tolerance too large. Adjusted to 1E-6\n";
        m_energy_tol = 1e-6;
    } 
    else
        m_energy_tol = fabs(energy_tol);
}

void hf_settings::set_ccsd_energy_tol(const double ccsd_energy_tol) 
{ 
    if (fabs(ccsd_energy_tol) > 1e-6) 
    {
        std::cout << "\n  Warning: CCSD energy tolerance too large. Adjusted to 1E-6\n";
        m_ccsd_energy_tol = 1e-6;
    } 
    else
        m_ccsd_energy_tol = fabs(ccsd_energy_tol);
}

void hf_settings::set_sad_energy_tol(const double sad_energy_tol) 
{ 
    if (fabs(sad_energy_tol) > 1e-4) 
    {
        std::cout << "\n  Warning: SAD SCF energy tolerance too large. Adjusted to 1E-5\n";
        m_sad_energy_tol = 1e-5;
    } 
    else
        m_sad_energy_tol = fabs(sad_energy_tol);
}

void hf_settings::set_integral_tol(const double threshold)
{
    if (fabs(threshold) > 1e-10) 
    {
        std::cout << "\n  Warning: integral threshold too large. Adjusted to 1E-10\n";
        m_integral_tol = 1e-10;
    } 
    else
        m_integral_tol = fabs(threshold);
}

void hf_settings::set_hessian_step_from_grad(const double grad_step)
{
    if (fabs(grad_step) < 1e-06) 
    {
        std::cout << "\n  Warning: stepsize less than allowed range. Adjusted to 1E-06\n";
        m_hessian_step_from_grad = 1e-06;
    } 
    else if (fabs(grad_step) > 1e-04) 
    {
        std::cout << "\n  Warning: stepsize less than allowed range. Adjusted to 1E-04\n";
        m_hessian_step_from_grad = 1e-04;
    }
    else
        m_hessian_step_from_grad = fabs(grad_step);
}

void hf_settings::set_hessian_step_from_energy(const double energy_step)
{
    if (fabs(energy_step) < 1e-04)
	{
		std::cout << "\n  Warning: stepsize less than allowed range. Adjusted to 1E-04\n";
		m_hessian_step_from_energy = 1e-04;
	}
	else if (fabs(energy_step) > 1e-02)
	{
		std::cout << "\n  Warning: stepsize less than allowed range. Adjusted to 1E-02\n";
		m_hessian_step_from_energy = 1e-02;
	}
    else
        m_hessian_step_from_energy = fabs(energy_step);
}

void hf_settings::set_grad_step_from_energy(const double energy_step)
{
    if (fabs(energy_step) < 1e-06) 
    {
        std::cout << "  Warning: stepsize less than allowed range. Adjusted to 1E-06\n";
        m_grad_step_from_energy = 1e-06;
    } 
    else if (fabs(energy_step) > 1e-04) 
    {
        std::cout << "  Warning: stepsize less than allowed range. Adjusted to 1E-04\n";
        m_grad_step_from_energy = 1e-04;
    }
    else
        m_grad_step_from_energy = fabs(energy_step);
}

void hf_settings::set_thermo_chem_pressure(const double pressure) 
{
    m_thermo_chem_P = pressure;
}

void hf_settings::set_thermo_chem_temperature(const double temperature) 
{
    m_thermo_chem_T = temperature;
}

void hf_settings::set_verbosity(const short verbosity) 
{ 
    m_verbosity = verbosity; 
}

void hf_settings::set_freeze_core(const bool freeze_core)
{
    m_freeze_core = freeze_core;
}

void hf_settings::set_do_molprops(const bool  mol_props)
{
    m_molprops = mol_props;
}

void hf_settings::set_molprops_cphf_iter(const bool cphf_iterative)
{
    m_molprops_cphf_iterative = cphf_iterative;
}

void hf_settings::set_use_pure_angular_momentum(const bool use_pure_am)
{
    m_use_pure_am = use_pure_am;
}

void hf_settings::set_geom_opt_write_trajectory(const bool geom_opt_write_trajectory)
{
    m_geom_opt_write_trajectory = geom_opt_write_trajectory;
}

void hf_settings::set_align_geom(const bool align_geom)
{
    m_align_geom = align_geom;
}

void hf_settings::set_use_symmetry(const bool use_symmetry)
{
    m_use_symmetry = use_symmetry;
}

void hf_settings::set_soscf(const bool soscf)
{
    m_use_soscf = soscf;
}

void hf_settings::set_project_hessian_translations_only(const bool project_trans_only)
{
    m_project_hessian_trans_only = project_trans_only;
}

void hf_settings::set_scf_direct(const bool scf_direct) 
{ 
    m_scf_direct = scf_direct; 
}

void hf_settings::set_uhf_guess_mix(const bool uhf_guess_mix) 
{ 
    m_uhf_guess_mix = uhf_guess_mix;
}

void hf_settings::set_geom_opt(const std::string& geom_opt) 
{ 
    m_geom_opt = geom_opt;
}

void hf_settings::set_geom_opt_trajectory_file(const std::string& geom_opt_trajectory_file) 
{ 
    m_geom_opt_trajectory_file = geom_opt_trajectory_file;
}

void hf_settings::set_geom_opt_algorithm(const std::string& geom_opt_algorithm) 
{ 
    m_geom_opt_algorithm = geom_opt_algorithm;
}

void hf_settings::set_geom_opt_stepsize(const double geom_opt_stepsize) 
{ 
    if (std::fabs(geom_opt_stepsize) > 1.0) 
    {
        std::cout << "\n  Warning: geometry optimizaton stepszie > 1.0, adjusted to 1.0\n";
        m_geom_opt_stepsize = 1.0;
    }
    else if(std::fabs(geom_opt_stepsize) < 0.05)
    {
        std::cout << "\n  Warning: geometry optimizaton stepszie < 0.05, adjusted to 0.05\n";
        m_geom_opt_stepsize = 0.05;
    }
    else
        m_geom_opt_stepsize = std::fabs(geom_opt_stepsize);             
}

void hf_settings::set_symmetrize_geom(const bool symmetrize)
{
    m_symmetrize_geom = symmetrize;
}

void hf_settings::set_geomopt_constrain__angles(const bool constrain)
{
    m_constrain_angles = constrain;
}

void hf_settings::set_geomopt_constrain__dihedral_angles(const bool constrain)
{
    m_constrain_dihedral_angles = constrain;
}

void hf_settings::set_geomopt_constrain__oop_angles(const bool constrain)
{
    m_constrain_oop_angles = constrain;
}

void hf_settings::set_screen_eri(const bool screen_eri)
{
    m_screen_eri = screen_eri;
}

void hf_settings::set_do_post_scf(const std::string& post_scf_type)
{
    m_post_scf_type = post_scf_type;
}

void hf_settings::set_frequencies_type(const std::string& freq_type)
{
    m_frequencies = freq_type;
}

void hf_settings::set_geom_opt_tol(const std::string& geom_opt_tol)
{
    m_geom_opt_tol = geom_opt_tol;
}

std::string& hf_settings::get_basis_set_name()
{ 
    return m_basis_set_name; 
}

std::string& hf_settings::get_basis_set_path()
{ 
    return m_basis_set_path; 
}

std::string& hf_settings::get_frequencies_type()
{ 
    return m_frequencies; 
}

std::string& hf_settings::get_basis_coord_type()
{
    return m_basis_coord_type;
}

std::string& hf_settings::get_hf_type() 
{ 
    return m_hf_type; 
}

std::string& hf_settings::get_post_scf_type()
{
    return m_post_scf_type;
}

std::string& hf_settings::get_geom_opt_tol()
{
    return m_geom_opt_tol;
}

std::string& hf_settings::get_guess_type() 
{ 
    return m_initial_guess; 
}

std::string& hf_settings::get_unit_type() 
{ 
    return m_unit_type; 
}

std::string& hf_settings::get_gradient_type() 
{ 
    return m_gradient_type; 
}

std::string& hf_settings::get_ci_type() 
{ 
    return m_ci_type; 
}

std::string& hf_settings::get_geom_opt_guess_hessian()
{ 
    return m_geom_opt_guess_hessian; 
}

double hf_settings::get_ccsd_rms_tol() 
{ 
    return m_ccsd_rms_tol; 
}

double hf_settings::get_rms_tol() 
{ 
    return m_rms_tol; 
}

double hf_settings::get_sad_rms_tol() 
{ 
    return m_sad_rms_tol; 
}

double hf_settings::get_soscf_rms_tol() 
{ 
    return m_soscf_rms_tol;
}

double hf_settings::get_ccsd_energy_tol() 
{ 
    return m_ccsd_energy_tol; 
}

double hf_settings::get_energy_tol() 
{ 
    return m_energy_tol; 
}

double hf_settings::get_sad_energy_tol() 
{ 
    return m_sad_energy_tol; 
}

double hf_settings::get_scf_damping_factor() 
{ 
    return m_scf_damping; 
}

double hf_settings::get_integral_tol()
{
    return m_integral_tol;
}

double hf_settings::get_hessian_step_from_grad()
{
    return m_hessian_step_from_grad;
}

double hf_settings::get_hessian_step_from_energy()
{
    return m_hessian_step_from_energy;
}

double hf_settings::get_grad_step_from_energy()
{
    return m_grad_step_from_energy;
}

double hf_settings::get_geom_opt_stepsize()
{
    return m_geom_opt_stepsize;
}

double hf_settings::get_thermo_chem_pressure()
{
    return m_thermo_chem_P;
}

double hf_settings::get_thermo_chem_temperature()
{
    return m_thermo_chem_T;
}

bool hf_settings::get_screen_eri() 
{ 
    return m_screen_eri;
}


bool hf_settings::get_scf_direct()
{ 
    return m_scf_direct;
}

bool hf_settings::get_symmetrize_geom()
{ 
    return m_symmetrize_geom;
}

bool hf_settings::get_geomopt_constrain__angles()
{
    return m_constrain_angles;
}

bool hf_settings::get_geomopt_constrain__dihedral_angles()
{
    return m_constrain_dihedral_angles;
}

bool hf_settings::get_geomopt_constrain__oop_angles()
{
    return m_constrain_oop_angles;
}

bool hf_settings::get_align_geom()
{ 
    return m_align_geom;
}

bool hf_settings::get_use_symmetry()
{ 
    return m_use_symmetry;
}

bool hf_settings::get_soscf()
{ 
    return m_use_soscf;
}

bool hf_settings::get_project_hessian_translations_only()
{ 
    return m_project_hessian_trans_only;
}

bool hf_settings::get_freeze_core()
{ 
    return m_freeze_core;
}

bool hf_settings::get_do_mol_props()
{ 
    return m_molprops;
}

bool hf_settings::get_molprops_cphf_iter()
{ 
    return m_molprops_cphf_iterative;
}

std::optional<bool> hf_settings::get_use_pure_angular_momentum()
{ 
    if (m_use_pure_am)
        return m_use_pure_am;
    else
        return std::nullopt;
}

bool hf_settings::get_geom_opt_write_trajectory()
{
    return m_geom_opt_write_trajectory;
}

bool hf_settings::get_uhf_guess_mix()
{ 
    return m_uhf_guess_mix;
}

std::string& hf_settings::get_geom_opt_trajectory_file()
{
    return m_geom_opt_trajectory_file;
}

std::string& hf_settings::get_geom_opt()
{ 
    return m_geom_opt;
}

std::string& hf_settings::get_geom_opt_algorithm()
{ 
    return m_geom_opt_algorithm;
}

std::string& hf_settings::get_point_group_equivalence_threshold()
{ 
    return m_point_group_equivalence_threshold; 
}

std::string& hf_settings::get_point_group()
{ 
    return m_point_group; 
}


int hf_settings::get_diis_size() 
{ 
    return m_diis_size; 
}

int hf_settings::get_max_scf_iterations()
{
    return m_max_scf_iter;
}

int hf_settings::get_max_ccsd_iterations()
{
    return m_max_ccsd_iter;
}

int hf_settings::get_max_sad_iterations()
{
    return m_max_sad_iter;
}

int hf_settings::get_max_soscf_iterations()
{
    return m_max_soscf_iter;
}

int hf_settings::get_max_geomopt_iterations()
{
    return m_max_geomopt_iter;
}

int hf_settings::get_ccsd_diis_size() 
{ 
    return m_ccsd_diis_size; 
}

short hf_settings::get_verbosity() 
{ 
    return m_verbosity; 
}