#ifndef KEYWORDS_MAP_H
#define KEYWORDS_MAP_H

#include <map>
#include <string>
#include <string_view>
#include "../settings/hfscf_settings.hpp"
using HF_SETTINGS::hf_settings;

namespace Keywords
{
    inline std::multimap<std::string, std::string> lookup_keyword
    {
        {"basis_set", "*"}, // * denotes no specific value is specified
        {"point_group_override", "*"},
        {"post_scf", "MP2"}, {"post_scf", "MP3"}, {"post_scf", "CCSD"}, {"post_scf", "CCSD(T)"},
        {"scf_type", "RHF"}, {"scf_type", "UHF"},
        {"initial_guess", "CORE"}, {"initial_guess", "SAD"},
        {"gradient_type", "SCF"}, {"gradient_type", "MP2"},
        {"ci_type", "CIS"}, {"ci_type", "RPA"},
        {"geom_opt_tol", "VERYHIGH"}, {"geom_opt_tol", "HIGH"}, {"geom_opt_tol", "MEDIUM"}, {"geom_opt_tol", "LOW"},
        {"geom_opt", "SCF"}, {"geom_opt", "MP2"},
        {"geom_opt_algorithm", "RFO"}, {"geom_opt_algorithm", "CGSD"},
        {"frequencies", "SCF"}, {"frequencies", "MP2"},
        {"units", "bohr"},  {"units", "angstrom"},
        {"electronic_properties", "true"}, {"electronic_properties", "false"},
        {"mol_props_cphf_iter", "true"}, {"mol_props_cphf_iter", "false"},
        {"use_puream", "true"}, {"use_puream", "false"},
        {"align_geom", "true"},  {"align_geom", "false"}, // developers only, not fully implemented
        {"use_symmetry", "true"},  {"use_symmetry", "false"},
        {"geom_opt_constrain_angles", "true"},  {"geom_opt_constrain_angles", "false"},
        {"geom_opt_constrain_dihedrals", "true"},  {"geom_opt_constrain_dihedrals", "false"},
        {"geom_opt_constrain_oop", "true"},  {"geom_opt_constrain_oop", "false"},
        {"soscf", "true"},  {"soscf", "false"},
        {"geom_opt_write_xyz", "true"}, {"geom_opt_write_xyz", "false"},
        {"eri_screen", "true"}, {"eri_screen", "false"},
        {"freeze_core", "true"}, {"freeze_core", "false"},
        {"uhf_guess_mix", "true"}, {"uhf_guess_mix", "false"},
        {"scf_direct", "true"}, {"scf_direct", "false"},
        {"symmetrize_geom", "true"}, {"symmetrize_geom", "false"},
        {"point_group_threshold", "TIGHT"}, {"point_group_threshold", "DEFAULT"}, {"point_group_threshold", "RELAXED"},
        {"diis_range", "int"},
        {"ccsd_diis_range", "int"},
        {"max_ccsd_iter", "int"},
        {"max_scf_iter", "int"},
        {"max_sad_iter", "int"},
        {"max_soscf_iter", "int"},
        {"max_geomopt_iter", "int"},
        {"charge", "int"},
        {"multiplicity", "int"},
        {"ccsd_energy_tol", "double"},
        {"energy_tol", "double"},
        {"sad_energy_tol", "double"},
        {"ccsd_rms_tol", "double"},
        {"rms_tol", "double"},
        {"sad_rms_tol", "double"},
        {"soscf_rms_tol", "double"},
        {"density_damp", "double"},
        {"integral_tol", "double"},
        {"geom_opt_stepsize", "double"},
        {"grad_step_from_energy", "double"},
        {"hessian_step_from_grad", "double"},
        {"hessian_step_from_energy", "double"},
        {"thermo_chem_pressure", "double"},
        {"thermo_chem_temperature", "double"}
    };

    inline std::map<std::string, std::string_view> err_mesg =
    {
        {"post_scf", " expecting \"post_scf = MP2|MP3|CCSD|CCSD(T)\""},
        {"scf_type", " expecting \"scf_type = UHF|RHF\""},
        {"initial_guess", " expecting \"initial_guess = CORE|SAD\""},
        {"gradient_type", " expecting \"gradient_type = SCF|MP2\""},
        {"ci_type", "  expecting \"ci_type = CIS|RPA\""},
        {"geom_opt_tol", "  expecting \"geom_opt_tol = LOW|MEDIUM|HIGH|VERYHIGH\""},
        {"geom_opt", "  expecting \"geom_opt = SCF|MP2\""},
        {"geom_opt_algorithm", "  expecting \"geom_opt_algorithm = RFO|CGSD\""},
        {"frequencies", " expecting \"frequencies = SCF|MP2\""},
        {"units"," expecting \"units = bohr|angstrom\""},
        {"electronic_properties", " expecting \"electronic_properties = true|false\""},
        {"mol_props_cphf_iter", " expecting \"mol_props_cphf_iter = true|false\""},
        {"use_puream", " expecting \"use_puream = true|false\""},
        {"geom_opt_write_xyz", " expecting \"geom_opt_write_xyz = true|false\""},
        {"align_geom", " expecting \"align_geom = true|false\""},
        {"use_symmetry", " expecting \"use_symmetry = true|false\""},
        {"geom_opt_constrain_angles", " expecting \"geom_opt_constrain_angles = true|false\""},
        {"geom_opt_constrain_dihedrals", " expecting \"geom_opt_constrain_dihedrals = true|false\""},
        {"geom_opt_constrain_oop", " expecting \"geom_opt_constrain_oop = true|false\""},
        {"soscf", " expecting \"soscf = true|false\""},
        {"freeze_core", " expecting \"freeze_core = true|false\""},
        {"uhf_guess_mix", " expecting \"uhf_guess_mix = true|false\""},
        {"eri_screen", " expecting \"eri_screen = true|false\""},
        {"scf_direct", " expecting \"scf_direct = true|false\""},
        {"symmetrize_geom", " expecting \"symmetrize_geom = true|false\""},
        {"point_group_threshold", " expecting \"point_group_threshold = RELAXED|DEFAULT|TIGHT\""},
        {"diis_range", " expecting \"diis_range = integer\" (allowed range: 2 to 8)"},
        {"ccsd_diis_range", " expecting \"ccsd_diis_range = integer\" (allowed range: 2 to 6)"},
        {"max_ccsd_iter", " expecting \"max_ccsd_iter = integer\""},
        {"max_scf_iter", " expecting \"max_scf_iter = integer\""},
        {"max_sad_iter", " expecting \"max_sad_iter = integer\""},
        {"max_soscf_iter", " expecting \"max_soscf_iter = integer (3 - 6)\""},
        {"max_geomopt_iter", " expecting \"max_geomopt_iter = integer\""},
        {"charge", " expecting \"charge = integer\""},
        {"multiplicity", " expecting \"multiplicity = integer\""},
        {"ccsd_energy_tol", " expecting \"ccsd_energy_tol = double\""},
        {"energy_tol", " expecting \"energy_tol = double\""},
        {"sad_energy_tol", " expecting \"sad_energy_tol = double\""},
        {"ccsd_rms_tol", " expecting \"ccsd_rms_tol = double\""},
        {"rms_tol", " expecting \"rms_tol = double\""},
        {"sad_rms_tol", " expecting \"sad_rms_tol = double\""},
        {"soscf_rms_tol", " expecting \"soscf_rms_tol = double (1e-4 to 1e-2)\""},
        {"integral_tol", " expecting \"integral_tol = double\""},
        {"density_damp", " expecting \"density_damp = double\" (damping range > 0.2 < 0.9. 1.0 = off)"},
        {"geom_opt_stepsize", " expecting \"geom_opt_stepsize = double\" (damping range > 0.05 < 1.0)"},
        {"grad_step_from_energy", " expecting \"grad_step_from_energy = double\" (range 1.0E-04 - 1.0E-06)"},
        {"hessian_step_from_grad", " expecting \"hessian_step_from_grad = double\" (range 1.0E-04 - 1.0E-06)"},
        {"hessian_step_from_energy", " expecting \"hessian_step_from_energy = double\" (range 1.0E-02 - 1.0E-04)"},
        {"thermo_chem_pressure", "expecting \"thermo_chem_pressure = double\""},
        {"thermo_chem_temperature", "expecting \"thermo_chem_temperature = double\""}
    };

    inline std::map<std::string, std::function<void(const std::string& )> > set_setting_str =
    {
        {"scf_type", &hf_settings::set_hf_type},
        {"post_scf", &hf_settings::set_do_post_scf},
        {"initial_guess", &hf_settings::set_guess_type},
        {"gradient_type", &hf_settings::set_gradient_type},
        {"frequencies", &hf_settings::set_frequencies_type},
        {"geom_opt", &hf_settings::set_geom_opt},
        {"geom_opt_algorithm", &hf_settings::set_geom_opt_algorithm},
        {"geom_opt_tol", &hf_settings::set_geom_opt_tol},
        {"ci_type", &hf_settings::set_ci_type},
        {"units", &hf_settings::set_unit_type},
        {"basis_set", &hf_settings::set_basis_set_name},
        {"point_group_threshold", &hf_settings::set_point_group_equivalence_threshold},
        {"point_group_override", &hf_settings::set_point_group},
    };

    template<typename T>
    std::map<std::string, std::function<void(const T )> > set_setting
    {
        {"electronic_properties", &hf_settings::set_do_molprops},
        {"mol_props_cphf_iter", &hf_settings::set_molprops_cphf_iter},
        {"use_puream", &hf_settings::set_use_pure_angular_momentum},
        {"align_geom", &hf_settings::set_align_geom},
        {"use_symmetry", &hf_settings::set_use_symmetry},
        {"soscf", &hf_settings::set_soscf},
        {"geom_opt_write_xyz", &hf_settings::set_geom_opt_write_trajectory},
        {"freeze_core", &hf_settings::set_freeze_core},
        {"uhf_guess_mix", &hf_settings::set_uhf_guess_mix},
        {"eri_screen", &hf_settings::set_screen_eri},
        {"symmetrize_geom", &hf_settings::set_symmetrize_geom},
        {"geom_opt_constrain_angles", &hf_settings::set_geomopt_constrain__angles},
        {"geom_opt_constrain_dihedrals", &hf_settings::set_geomopt_constrain__dihedral_angles},
        {"geom_opt_constrain_oop", &hf_settings::set_geomopt_constrain__oop_angles},
        {"scf_direct", &hf_settings::set_scf_direct},
        {"diis_range", &hf_settings::set_diis_size},
        {"ccsd_diis_range", &hf_settings::set_ccsd_diis_size},
        {"max_ccsd_iter", &hf_settings::set_max_ccsd_iterations},
        {"max_scf_iter", &hf_settings::set_max_scf_iterations},
        {"max_sad_iter", &hf_settings::set_max_sad_iterations},
        {"max_soscf_iter", &hf_settings::set_max_soscf_iterations},
        {"max_gem_iter", &hf_settings::set_max_geomopt_iterations},
        {"ccsd_energy_tol", &hf_settings::set_ccsd_energy_tol},
        {"energy_tol", &hf_settings::set_energy_tol},
        {"sad_energy_tol", &hf_settings::set_sad_energy_tol},
        {"ccsd_rms_tol", &hf_settings::set_ccsd_rms_tol},
        {"rms_tol", &hf_settings::set_rms_tol},
        {"sad_rms_tol", &hf_settings::set_sad_rms_tol},
        {"soscf_rms_tol", &hf_settings::set_soscf_rms_tol},
        {"density_damp", &hf_settings::set_scf_damping_factor},
        {"integral_tol", &hf_settings::set_integral_tol},
        {"geom_opt_stepsize", &hf_settings::set_geom_opt_stepsize},
        {"grad_step_from_energy", &hf_settings::set_grad_step_from_energy},
        {"hessian_step_from_grad", &hf_settings::set_hessian_step_from_grad},
        {"hessian_step_from_energy", &hf_settings::set_hessian_step_from_energy},
        {"thermo_chem_pressure", &hf_settings::set_thermo_chem_pressure},
        {"thermo_chem_temperature", &hf_settings::set_thermo_chem_temperature}
        // multiplicity:   Not a setting, but is a keyword molecule member, no function needed 
        // should prob'lly be moved inside coordinates section
        // charge: Not a setting, but is a keyword molecule member, no function needed
        // should prob'lly be moved inside coordinates section
    };
}

#endif