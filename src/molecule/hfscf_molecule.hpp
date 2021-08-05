#ifndef MOLEC_MOLECULE_H
#define MOLEC_MOLECULE_H

#include "../basis/hfscf_basis.hpp"
#include "../utils/hfscf_zmat.hpp"
#include "../integrals/hfscf_shellpair.hpp"
#include "hfscf_atom.hpp"
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <vector>
#include <array>
#include <string>
#include <typeinfo>

template<typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using InertiaTensor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using ShellPairVector = std::vector<BASIS::ShellPair>;

namespace MOLEC
{
    struct mask
    {
        explicit mask(int start, int end)
        : mask_start(start), mask_end(end){}
	    int mask_start;
	    int mask_end;
    };

    class Molecule
    {
        public:
            Molecule() = delete;
            explicit Molecule(const std::string& path) : path_root(path) {} 
            Molecule(const Molecule&) = delete;
            Molecule& operator=(const Molecule& other) = delete;
            Molecule(const Molecule&&) = delete;
            Molecule&& operator=(const Molecule&& other) = delete;
            
            void init_molecule(const std::string& filename);

            // Get the classical nuclear repulsion energy
            double get_enuc() const;

            // Get the center of charge vector. Useful for dipoles or other properties
            const Eigen::Ref<const Vec3D> get_center_of_charge_vector() const;

             // Get the center of mass vector.
            const Eigen::Ref<const Vec3D> get_center_of_mass_vector() const;

            // Get the rotational constants for the current geometry (not zero point)
            const Eigen::Ref<const Vec3D> get_rotational_constants() const;

            // Return the total number of electrons
            Eigen::Index get_num_electrons() const;

            // Return the total number or catersian basis functions if the basis is cartesian, or 
            // pure if the basis is spherical. total pure is <= total cartesians. Pure has 5d, 7f. 
            // Cartesian 6d, 10f etc.
            Eigen::Index get_num_orbitals() const;

            // Return the total number of catersian basis functions for pure and non pure basis sets.
            Eigen::Index get_num_cart_orbitals() const;
            
            // Return the total number of shells. A shell is s or px py pz etc. Water sto-3g has 5 shells.
            Eigen::Index get_num_shells() const;

            // Return an array of atmic numbers for the molcule.
            const std::vector<int>& get_z_values() const;

            // Return an array of all shells
            const ShellVector& get_shells() const;

             // Return an array of all shell pairs
            const ShellPairVector& get_shell_pairs() const;

            //! Return the electronic spin 2S (note, multiplicity is 2S + 1)
            Eigen::Index get_spin() const;

            //! Return the electronic multiplicity is 2S + 1
            Eigen::Index get_multiplicity() const;

            //! Return the nubber of alpha electrons deduced from multiplicity and Z
            Eigen::Index get_nalpha() const;

            //! Return the nubber of beta electrons deduced from multiplicity and Z
            Eigen::Index get_nbeta() const;

            // Print molecule and setings info: 
            // input: bool post_geom_opt. Are we at the post geometry optimisation stage.
            void print_info(const bool post_geom_opt);

            // Return an array of all atoms in the molecule
            const std::vector<Atom>& get_atoms() const;

            // Return a reference to the cartesian geometry for the molecule
            const Eigen::Ref<const EigenMatrix<double> > get_geom() const;

            // Return a copy of the cartesian geometry for the molecule
            EigenMatrix<double> get_geom_copy() const;

            // Return a mask vector to denote what basis functions belong to a given atomic center.
            // Cartesian basis functions
            // sto-3g water with 7 basis functions would look like
            // 1H: 1 0 0 0 0 0 0 1s
            // 2H: 0 1 0 0 0 0 0 1s
            // 3O: 0 0 1 1 1 1 1 1s 2s 3x2p
            const std::vector<mask>& get_atom_mask() const; // mask for cartesian basis

            // Return a mask vector to denote what basis functions belong to a given atomic center.
            // Pure basis functions
            // sto-3g water with 7 basis functions would look like
            // 1H: 1 0 0 0 0 0 0 1s
            // 2H: 0 1 0 0 0 0 0 1s
            // 3O: 0 0 1 1 1 1 1 1s 2s 3x2p
            const std::vector<mask>& get_atom_spherical_mask() const;  // mask for spherical basis

            // Steepest descent
            void update_geom(const Eigen::Ref<const EigenMatrix<double> >& gradient, 
                             const double stepsize, bool do_geom_analysis = false);
            // Conjugate gradient
            void update_geom(const Eigen::Ref<const EigenMatrix<double> >& gradient_current,
                             const Eigen::Ref<const EigenMatrix<double> >& gradient_previous,
                             const double stepsize, bool do_geom_analysis = false);
            
            void update_geom(const double delta, const int atom_num, const int cart_dir);
            void update_geom(const Eigen::Ref<const EigenMatrix<double> >& new_geom, bool do_geom_analysis = false);

            // Return if the molecule is linear
            bool molecule_is_linear() const;

            // Return the relative atomic mass of the molecule
            double get_mass() const;
            
            // Return the rotational symmetry number of the molecule.
            int get_sigma();

            // Return the point group of the molecule
            const std::string& get_point_group() const;

            // Return the symmetry blocks per irrep
            const std::vector<EigenMatrix<double>>& get_sym_blocks() const { return symblocks;}

            // Return symmetry species per irrep
            const std::vector<std::string>& get_symmetry_species() const { return sym_species;}

            // Return irep dimensions
            const std::vector<int>& get_irreps() const { return irrep_sizes;}

            // Return whether to use pure angular momentum basis functions
            bool use_pure_am() const;
        
        private:
            std::string path_root;
            bool isaligned{false};
            bool islinear{false};
            Eigen::Index num_gtos{0};
            Eigen::Index num_basis_gtos{0};
            Eigen::Index num_orbitals{0};
            Eigen::Index num_shells{0};
            Eigen::Index nelectrons{0};
            Eigen::Index spin{0};
            Eigen::Index multiplicity{1};
            int charge{0};
            int natoms{0};
            int sigma{0}; // symmetry number since version 0.4.2
            double e_nuc{0};
            double molecular_mass{0};
            double m_sym_deviation{0};
            std::string comment;
            std::string m_point_group{};
            std::string m_sub_group{};
            std::string m_basis_coord_type{};
            Vec3D Q_c; // center of charge
            Vec3D center_of_mass; // center of mass
            Vec3D Be; // rotational constants
            std::vector<int> zval;
            ShellVector shells;
            ShellPairVector shpair;
            std::vector<Atom> atoms;
            std::vector<mask> atom_cmask; // Used by derivatives/gradients - cartesian basis
            std::vector<mask> atom_smask; // Used by derivatives/gradients - spherical basis
            std::vector<int> shell_centers;
            EigenMatrix<double> geom;
            EigenMatrix<double> start_geom;
            EigenMatrix<double> strans;
            std::vector<ZMAT::zparams> zmat;
            InertiaTensor i_tensor;
            Eigen::Vector3d i_tensor_evals;
            EigenMatrix<double>i_tensor_evecs;
            std::vector<EigenMatrix<double>> symblocks;
            std::vector<std::string> sym_species;
            std::vector<int> irrep_sizes;

            void do_geometry_analysis();
            void do_salc_analysis();
            void calc_nuclear_repulsion_energy();
            void set_geom_opt_trajectory_params(const std::string& filename) const;
            void write_basis_set_info() const;
            void create_shell_pairs();
            Eigen::Index get_num_gtos() const;
    };
}

#endif
// End MOLEC_MOLECULE
