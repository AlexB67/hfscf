#ifndef HFSCF_SETTINGS_H
#define HFSCF_SETTINGS_H

#include <string>
#include <optional>

namespace HF_SETTINGS
{
    class hf_settings
    {
        private:
            inline static std::string   m_hf_type{"RHF"};
            inline static std::string   m_initial_guess{"SAD"};
            inline static std::string   m_basis_set_name{"STO-3G"};
            inline static std::string   m_basis_set_path{""};
            inline static std::string   m_geom_opt_trajectory_file{""};
            inline static std::string   m_basis_coord_type{"cartesian"};
            inline static std::string   m_post_scf_type{""};
            inline static std::string   m_unit_type{"bohr"};
            inline static std::string   m_gradient_type{};
            inline static std::string   m_ci_type{}; // Configuration interaction
            inline static std::string   m_geom_opt_tol{"MEDIUM"};
            inline static std::string   m_geom_opt{""};
            inline static std::string   m_geom_opt_algorithm{"RFO"};
            inline static std::string   m_frequencies{""};
            inline static std::string   m_point_group_equivalence_threshold{"DEFAULT"};
            inline static std::string   m_point_group{""};

            inline static double        m_scf_damping{1.0}; // i.e no damping
            inline static double        m_rms_tol{1.0E-07};
            inline static double        m_ccsd_rms_tol{1.0E-07};
            inline static double        m_soscf_rms_tol{5.0E-03};
            inline static double        m_sad_rms_tol{1.0E-06};
            inline static double        m_energy_tol{1.0E-07};
            inline static double        m_ccsd_energy_tol{1.0E-7};
            inline static double        m_sad_energy_tol{1.0E-05};
            inline static double	    m_integral_tol{1.0E-14};
            inline static double	    m_geom_opt_stepsize{0.5}; // CG only
            inline static double        m_hessian_step_from_grad{1.0E-6};
            inline static double        m_hessian_step_from_energy{1.0E-3};
            inline static double        m_grad_step_from_energy{1.0E-6};
            inline static double        m_thermo_chem_T{298.15};
            inline static double        m_thermo_chem_P{101325.0};
            inline static int           m_diis_size{5};
            inline static int           m_ccsd_diis_size{5};
            inline static int           m_max_scf_iter{50};
            inline static int           m_max_ccsd_iter{30};
            inline static int           m_max_sad_iter{25};
            inline static int           m_max_soscf_iter{4};
            inline static int           m_max_geomopt_iter{15};
            inline static short         m_verbosity{1};
            inline static bool          m_scf_direct{false};
            inline static bool          m_symmetrize_geom{false};
            inline static bool          m_constrain_angles{false};
            inline static bool          m_constrain_dihedral_angles{false};
            inline static bool          m_constrain_oop_angles{false};
            inline static bool          m_align_geom{true}; // devs only
            inline static bool          m_freeze_core{false};
            inline static bool          m_uhf_guess_mix{false};
            inline static bool          m_screen_eri{true};
            inline static bool          m_molprops{false};
            inline static bool          m_molprops_cphf_iterative{false};
            inline static bool          m_geom_opt_write_trajectory{false};
            inline static std::optional<bool>  m_use_pure_am;
            inline static bool          m_use_symmetry{false};
            inline static bool          m_use_soscf{false};
        
        public:
            hf_settings() = delete;
            hf_settings(const hf_settings&) = delete;
            hf_settings& operator=(const hf_settings& other) = delete;
            hf_settings(const hf_settings&&) = delete;
            hf_settings&& operator=(const hf_settings&& other) = delete;

            static void set_basis_set_name(const std::string& basis_name);
            static void set_basis_set_path(const std::string& basis_set_path);
            static void set_basis_coord_type(const std::string& basis_coord_type);
            static void set_hf_type(const std::string& hf_type);
            static void set_guess_type(const std::string& inital_guess);
            static void set_scf_damping_factor(const double scf_damping);
            static void set_diis_size(const int diis_size);
            static void set_max_scf_iterations(const int max_scf_iter);
            static void set_max_ccsd_iterations(const int max_ccsd_iter);
            static void set_max_sad_iterations(const int max_sad_iter);
            static void set_max_soscf_iterations(const int max_soscf_iter);
            static void set_max_geomopt_iterations(const int max_geomopt_iter);
            static void set_ccsd_diis_size(const int diis_size);
            static void set_rms_tol(const double rms_tol);
            static void set_ccsd_rms_tol(const double ccsd_energy_tol);
            static void set_sad_rms_tol(const double sad_rms_tol);
            static void set_soscf_rms_tol(const double soscf_rms_tol);
            static void set_integral_tol(const double threshold);
            static void set_hessian_step_from_grad(const double grad_step);
            static void set_hessian_step_from_energy(const double energy_step);
            static void set_grad_step_from_energy(const double energy_step);
            static void set_energy_tol(const double scf_energy_tol);
            static void set_ccsd_energy_tol(const double ccsd_energy_tol);
            static void set_sad_energy_tol(const double sad_energy_tol);
            static void set_geom_opt_stepsize(const double geom_opt_stepsize);
            static void set_thermo_chem_temperature(const double temperture);
            static void set_thermo_chem_pressure(const double pressure);
            static void set_do_post_scf(const std::string& post_scf_type);
            static void set_geom_opt_tol(const std::string& geom_opt_tol);
            static void set_frequencies_type(const std::string& freq_type);
            static void set_point_group_equivalence_threshold(const std::string& equivalence_threshold);
            static void set_point_group(const std::string& group);
            static void set_verbosity(short verbosity);
            static void set_unit_type(const std::string& unit);
            static void set_gradient_type(const std::string& gradient_type);
            static void set_ci_type(const std::string& ci_type);
            static void set_scf_direct(const bool scf_direct);
            static void set_screen_eri(const bool screen_eri);
            static void set_symmetrize_geom(const bool symmetrize);
            static void set_geomopt_constrain__angles(const bool constrain);
            static void set_geomopt_constrain__dihedral_angles(const bool constrain);
            static void set_geomopt_constrain__oop_angles(const bool constrain);
            static void set_freeze_core(const bool freeze_core);
            static void set_uhf_guess_mix(const bool uhf_guess_mix);
            static void set_do_molprops(const bool molprops);
            static void set_molprops_cphf_iter(const bool cphf_iter);
            static void set_use_pure_angular_momentum(const bool pure_am);
            static void set_geom_opt_write_trajectory(const bool geom_opt_write_trajectory);
            static void set_geom_opt_trajectory_file(const std::string& trajectory_file);
            static void set_geom_opt(const std::string& geom_opt);
            static void set_geom_opt_algorithm(const std::string& geom_opt_algorithm);
            static void set_align_geom(const bool align_geom);
            static void set_use_symmetry(const bool use_symmetry);
            static void set_soscf(const bool enabe_soscf);

            static std::string& get_basis_set_name();
            static std::string& get_basis_set_path();
            static std::string& get_basis_coord_type();
            static std::string& get_hf_type();
            static std::string& get_guess_type();
            static std::string& get_post_scf_type();
            static std::string& get_unit_type();
            static std::string& get_gradient_type();
            static std::string& get_ci_type();
            static std::string& get_geom_opt_tol();
            static std::string& get_frequencies_type();
            static std::string& get_geom_opt();
            static std::string& get_geom_opt_algorithm();
            static std::string& get_point_group_equivalence_threshold();
            static std::string& get_point_group();
            static std::string& get_geom_opt_trajectory_file();
            static double get_ccsd_rms_tol();
            static double get_rms_tol();
            static double get_sad_rms_tol();
            static double get_soscf_rms_tol();
            static double get_ccsd_energy_tol();
            static double get_energy_tol();
            static double get_sad_energy_tol();
            static double get_scf_damping_factor();
            static double get_integral_tol();
            static double get_hessian_step_from_grad();
            static double get_hessian_step_from_energy();
            static double get_grad_step_from_energy();
            static double get_geom_opt_stepsize();
            static double get_thermo_chem_temperature();
            static double get_thermo_chem_pressure();
            static int get_diis_size();
            static int get_ccsd_diis_size();
            static int get_max_scf_iterations();
            static int get_max_ccsd_iterations();
            static int get_max_sad_iterations();
            static int get_max_soscf_iterations();
            static int get_max_geomopt_iterations();
            static short get_verbosity();
            static bool get_scf_direct();
            static bool get_symmetrize_geom();
            static bool get_geomopt_constrain__angles();
            static bool get_geomopt_constrain__dihedral_angles();
            static bool get_geomopt_constrain__oop_angles();
            static bool get_freeze_core();
            static bool get_do_mol_props();
            static bool get_molprops_cphf_iter();
            static std::optional<bool> get_use_pure_angular_momentum();
            static bool get_uhf_guess_mix();
            static bool get_screen_eri();
            static bool get_geom_opt_write_trajectory();
            static bool get_align_geom();
            static bool get_use_symmetry();
            static bool get_soscf();
    };
}

#endif
