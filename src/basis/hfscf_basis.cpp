#include "../basis/hfscf_basis.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../settings/hfscf_settings.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

using hfscfmath::fac;
using hfscfmath::pi;

BASIS::Basisset::Basisset(const std::string& basis_set_name, const Matrix1D<int>& zval)
: m_basis_set_name(basis_set_name), m_zval(zval)
{
    load_basis_set(m_basis_set_name.c_str());
}

// Format psi4 compatible

void BASIS::Basisset::load_basis_set(const char *filename)
{
    std::ifstream inputfile(filename, std::ifstream::in);

    int Z;
    int endmarker;
    size_t unique_atoms = 0;
    size_t unique_total = 0;
    double scale_factor;
    std::string atom_name;
    std::string line;
    std::string orbitaltype;
    std::vector<double> vec_alpha;
    std::vector<double> vec_c1;
    std::vector<double> vec_c2;

    constexpr auto find_unique_atoms = [](const std::vector<int>& zvals) -> size_t
    {
        size_t total = 0;
        for(size_t i = 0; i < zvals.size(); ++i)
        {
            size_t j;
            for(j = 0; j < i; ++j) if (zvals[j] == zvals[i]) break;
            if(i == j) total++;
        }

        return total;
    };

    int Zmax = 0;
    for(const auto& Z_current : m_zval) if (Z_current > Zmax) Zmax = Z_current;

    unique_total = find_unique_atoms(m_zval);

    // Get the header line for the basis, spherical or cartesian if present
    getline(inputfile, line);
    
    if(!line.empty() && "!" != line.substr(0,1))
    {
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        m_basis_coord_type = line;
    } 

    while (inputfile.good())
    {
        // account for atoms of same type
        if (unique_atoms == unique_total)
        {
            inputfile.close();
            return;
        }

        getline(inputfile, line);

        if ("!" == line.substr(0,1) || line.empty() || "*" == line.substr(0, 1))
        {
            continue;
        }
        else
        {
            bool been_once = false;
            std::istringstream data(line);
            data >> atom_name;
            data >> endmarker; // unused

            const auto iter = ELEMENTDATA::name_to_Z.find(atom_name);

            if(iter == ELEMENTDATA::name_to_Z.end())
            {
                std::cout << "\n\n  Error: Unsupported element: " << atom_name << '\n';
                std::cout << "  Aborting.\n";
                exit(1);
            }

            Z = iter->second;

            // sanity check if atom not in basis
            if(Z > Zmax) // If this happens we missed an atom
            {
                std::cout << "\n\n  Error: One or more atoms for the given molecule are not present in basis:\n";
                std::cout << "  " << filename << '\n';
                std::cout << "  Last processed atom in basis file: " << atom_name << '\n';  
                std::cout << "  Aborting.\n";
                exit(EXIT_FAILURE);
            }

            getline(inputfile, line);
            std::istringstream orbtype(line);
            orbtype >> orbitaltype;

            size_t num_cof;
            orbtype >> num_cof;
            orbtype >> scale_factor; // currently unused

            for (;;)
            {
                vec_alpha.clear();
                vec_c1.clear();
                vec_c2.clear();

                for (size_t i = 0; i < num_cof; ++i) 
                {
                    getline(inputfile, line);
                    // Translate Fortran to C++ numbers
                    std::for_each(line.begin(), line.end(), [] (char& c){ if (c == 'D') c = 'E'; });
                    std::istringstream alpha_cof_data(line);

                    double alpha, cof1, cof2;
                    alpha_cof_data >> alpha;

                    if("SP" == orbitaltype)
                    {
                        alpha_cof_data >> cof1;
                        alpha_cof_data >> cof2;
                        vec_alpha.emplace_back(alpha);
                        vec_c1.emplace_back(cof1);
                        vec_c2.emplace_back(cof2);
                    }
                    else
                    {
                        alpha_cof_data >> cof1;
                        vec_alpha.emplace_back(alpha);
                        vec_c1.emplace_back(cof1);
                    }
                }

                size_t atom_num = 0;
                for (const auto &Z_i : m_zval) 
                {
                    if (Z == Z_i) 
                    {
                        if ("S" == orbitaltype)
                        {
                            // Shell
                            num_shells += 1;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 0, vec_alpha, vec_c1));

                            num_gtos += 1 * static_cast<int>(vec_c1.size());
                            num_sgtos += 1 * static_cast<int>(vec_c1.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size());

                            if(!been_once)
                            {
                                ++unique_atoms; // assume at least 1 S on every unique atom
                                been_once = true;
                            }
                        } 
                        else if ("P" == orbitaltype)
                        {
                            num_shells += 1;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 1, vec_alpha, vec_c1));
                            
                            num_gtos += 3 * static_cast<int>(vec_c1.size());
                            num_sgtos += 3 * static_cast<int>(vec_c1.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size());
                        }
                        else if ("SP" == orbitaltype)
                        {
                            num_shells += 2;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 0, vec_alpha, vec_c1));
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 1, vec_alpha, vec_c2));
                            
                            num_gtos += static_cast<int>(vec_c1.size()) + 3 * static_cast<int>(vec_c2.size());
                            num_sgtos += static_cast<int>(vec_c1.size()) + 3 * static_cast<int>(vec_c2.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size()) + static_cast<int>(vec_c2.size());
                        }
                        else if ("D" == orbitaltype)
                        {
                            num_shells += 1;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 2, vec_alpha, vec_c1));

                            num_gtos += 6 * static_cast<int>(vec_c1.size());
                            num_sgtos += 5 * static_cast<int>(vec_c1.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size());
                        }
                        else if ("F" == orbitaltype)
                        {
                            num_shells += 1;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 3, vec_alpha, vec_c1));

                            num_gtos += 10 * static_cast<int>(vec_c1.size());
                            num_sgtos += 7 * static_cast<int>(vec_c1.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size());
                        }
                        else if ("G" == orbitaltype)
                        {
                            num_shells += 1;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 4, vec_alpha, vec_c1));

                            num_gtos += 15 * static_cast<int>(vec_c1.size());
                            num_sgtos += 9 * static_cast<int>(vec_c1.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size());
                        }
                        else if ("H" == orbitaltype)
                        {
                            num_shells += 1;
                            m_shell_basis.emplace_back(Basisshell_data(Z, atom_num, 5, vec_alpha, vec_c1));

                            num_gtos += 21 * static_cast<int>(vec_c1.size());
                            num_sgtos += 11 * static_cast<int>(vec_c1.size());
                            num_unique_gtos += static_cast<int>(vec_c1.size());
                        }
                        else 
                        {
                            std::cout << "\n  Error: unrecognised basis. S, P, D, F, G and H supported only. Aborting.\n";
                            std::cout << "  last known data:\n"; 
                            std::cout << "  " << line << '\n';
                            exit(EXIT_FAILURE);
                        }
                    }
                    ++atom_num;
                }

                getline(inputfile, line);
                orbtype.clear();
                orbtype.str(line);
                orbtype >> orbitaltype;
                if("*" == orbitaltype.substr(0, 1)) break;
                orbtype >> num_cof;
                orbtype >> scale_factor;
            }
        }
    }
}

using HF_SETTINGS::hf_settings;
ShellVector BASIS::Basisset::get_shells(const size_t atom_number, const Eigen::Ref<const Vec3D>& r)
{
    ShellVector shells;
    bool ispure = get_basis_coord_type() != "cartesian";

	if (hf_settings::get_use_pure_angular_momentum().has_value())
		ispure = hf_settings::get_use_pure_angular_momentum().value(); // optional override

    for (const auto &i : m_shell_basis)
    {   
        if( atom_number == i.get_atom_num()) // Note atom_number is an index not Z(atomic number)
        {
            int amsize1 = (i.get_L() + 1) * (i.get_L() + 2) / 2;
            int amsize2 = (2 * i.get_L() + 1);
            
            EigenVector<double> c = EigenVector<double>(i.get_alpha_size());;
            EigenVector<double> alpha = EigenVector<double>(i.get_alpha_size());

            for (int j = 0; j < (int) i.get_alpha_size(); ++j)
            {
                c(j) = i.get_c(j); alpha(j) = i.get_alpha(j);
            }

            shells.emplace_back(Shell(ispure, i.get_L(), shell_idx, shell_ids, r, c, alpha));
            shell_idx += amsize1;
            shell_ids += amsize2;
        }
    }

    return shells;
}
