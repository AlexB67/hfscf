#include "hfscf_sad.hpp"
#include "../integrals/hfscf_osnuclear.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include "../molecule/hfscf_elements.hpp"
#include <Eigen/Eigenvalues>

using Eigen::Index;
using HF_SETTINGS::hf_settings;
using hfscfmath::index_ij;
using hfscfmath::index_ijkl;

void Mol::sad_uhf_solver::init_data(bool print)
{
    diis_log = print;
    print_sad = print;

    //if (hf_settings::get_verbosity() > 4 && print) diis_log = true;
    diis_log = false;
    
    const auto& at = molecule->get_atoms()[atom];

    if (molecule->use_pure_am())
    {
        norbitals = at.get_pure_nbfs();
        bf_offset = at.get_pure_offset();
    }
    else
    {   
        norbitals = at.get_cart_nbfs();
        bf_offset = at.get_cart_offset();
    }

    hcore_mat = molkin.block(bf_offset, bf_offset, norbitals, norbitals);

    calc_potential_matrix();
    
    s_mat = molovlap.block(bf_offset, bf_offset, norbitals, norbitals);
    s_mat_sqrt = EigenMatrix<double>::Zero(norbitals, norbitals);
    fp_mat = EigenMatrix<double>::Zero(norbitals, norbitals);
    fp_mat_b = EigenMatrix<double>::Zero(norbitals, norbitals);
    f_mat = EigenMatrix<double>::Zero(norbitals, norbitals);
    f_mat_b = EigenMatrix<double>::Zero(norbitals, norbitals);
    d_mat = EigenMatrix<double>::Zero(norbitals, norbitals);
    d_mat_b = EigenMatrix<double>::Zero(norbitals, norbitals);
    e_rep_mat = EigenVector<double>(norbitals * (norbitals + 1)
              * (norbitals * norbitals + norbitals + 2) / 8);

    const auto & zvals = molecule->get_z_values();
    atom_name = ELEMENTDATA::atom_names[zvals[atom] - 1];
    atom_name.erase(std::remove(atom_name.begin(), atom_name.end(), ' '), atom_name.end());

    if (hf_settings::get_verbosity() > 1 && print_sad)
        std::cout << "\n  Performing a UHF coputation for unique atom " << atom_name << "\n";

    calc_initial_fock_matrix();
    calc_density_matrix();

    if (hf_settings::get_verbosity() > 3 && print_sad)
    {
        std::cout << "\n iteration: " << iteration << "\n";

        std::cout << "\n  ***********";
        std::cout << "\n       S";
        std::cout << "\n  ***********\n";
        HFCOUT::pretty_print_matrix<double>(s_mat);

        std::cout << "\n  **********";
        std::cout << "\n    S^-1/2";
        std::cout << "\n  **********\n";
        HFCOUT::pretty_print_matrix<double>(s_mat);

        std::cout << "\n  **********";
        std::cout << "\n    H(core)";
        std::cout << "\n  **********\n";
        HFCOUT::pretty_print_matrix<double>(hcore_mat);

        std::cout << "\n  ***********";
        std::cout << "\n    C_alpha";
        std::cout << "\n  ***********\n";
        HFCOUT::pretty_print_matrix<double>(mo_coff);

        std::cout << "\n  ***********";
        std::cout << "\n    C_beta";
        std::cout << "\n  ***********\n";
        HFCOUT::pretty_print_matrix<double>(mo_coff_b);

        std::cout << "\n  ***********";
        std::cout << "\n    D_alpha";
        std::cout << "\n  ***********\n";
        HFCOUT::pretty_print_matrix<double>(d_mat);

        std::cout << "\n  ***********";
        std::cout << "\n    D_beta";
        std::cout << "\n  ***********\n";
        HFCOUT::pretty_print_matrix<double>(d_mat_b);
    }
}

void Mol::sad_uhf_solver::calc_initial_fock_matrix()
{
    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(s_mat);
    const EigenMatrix<double>& s_evecs = solver.eigenvectors();
    const EigenMatrix<double>& s_evals = solver.eigenvalues();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i)
         for (Index j = 0; j < norbitals; ++j)
         {
             s_mat_sqrt(i, j) = 0;
             for (Index k = 0; k < norbitals; ++k)
                s_mat_sqrt(i, j) += (1.0 / sqrt(s_evals(k))) * s_evecs(i, k) * s_evecs(j, k);
         }

  
    fp_mat.noalias() = s_mat_sqrt.transpose() * hcore_mat * s_mat_sqrt;
}

void Mol::sad_uhf_solver::calc_density_matrix()
{
    if(0 == iteration)
    {
        nelectrons = molecule->get_atoms()[atom].get_Z();
        const double nalpha = 0.5 * static_cast<double>(nelectrons);
        const double nbeta = nalpha;
        occ = get_pop_numbers();

        if (hf_settings::get_verbosity() > 2 && print_sad)
        {
            EigenMatrix<double> coords = molecule->get_geom();
            Eigen::RowVector3d coord(coords(atom, 0), coords(atom, 1), coords(atom, 2));
            std::cout << "\n  Data for atom " << atom_name << "\n";
            std::cout << "  coordinates / bohr  = " << std::left << std::setprecision(6) << coord << "\n";
            std::cout << "  basis set           = " << hf_settings::get_basis_set_name() << "\n";
            std::cout << "  Use pure AM         = "; 
            (molecule->use_pure_am()) ? std::cout << "true\n" : std::cout << "false\n";
            std::cout << "  basis functions     = " << std::setprecision(2) << norbitals << "\n";
            std::cout << "  nalpha              = " << std::setprecision(2) << nalpha << "\n";
            std::cout << "  nbeta               = " << std::setprecision(2) << nbeta << "\n";
            std::cout << "  Fully occupied      = " << nfrozen << "\n";
            std::cout << "  Partially occupied  = " << nactive << "\n  with ";
            std::cout << (nalpha - (double)nfrozen) / (double) nactive 
                    << " alpha and " << (nbeta - (double)nfrozen) / (double) nactive  
                    << " beta.\n\n";
        }

        Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(fp_mat, Eigen::ComputeEigenvectors);
        EigenMatrix<double> f_mat_evecs = solver.eigenvectors();
        mo_coff = s_mat_sqrt * f_mat_evecs * pop_index.asDiagonal();
        mo_coff_b = mo_coff;
    }
    else
    {
        d_mat_previous = d_mat;
        d_mat_previous_b = d_mat_b;
        Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver_alpha(fp_mat, Eigen::ComputeEigenvectors);
        const EigenMatrix<double>& f_mat_evecs_alpha = solver_alpha.eigenvectors();
        mo_coff = s_mat_sqrt * f_mat_evecs_alpha * pop_index.asDiagonal();
        
        Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver_beta(fp_mat_b, Eigen::ComputeEigenvectors);
        const EigenMatrix<double>& f_mat_evecs_beta = solver_beta.eigenvectors();
        mo_coff_b = s_mat_sqrt * f_mat_evecs_beta * pop_index.asDiagonal();
    }
    

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i) 
        for (Index j = 0; j < norbitals; ++j) 
        {
            d_mat(i, j) = 0;
            for (Index k = 0; k < occ; ++k)
                d_mat(i, j) += mo_coff(i, k) * mo_coff(j, k);
        }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (Index i = 0; i < norbitals; ++i) 
        for (Index j = 0; j < norbitals; ++j) 
        {
            d_mat_b(i, j) = 0;
            for (int k = 0; k < occ; ++k) 
                d_mat_b(i, j) += mo_coff_b(i, k) * mo_coff_b(j, k);
        }

    calc_scf_energy();
}

void Mol::sad_uhf_solver::calc_scf_energy()
{
    scf_energy_previous = scf_energy_current;

    (iteration > 0) ? scf_energy_current = 0.5 * (d_mat.cwiseProduct(hcore_mat + f_mat).sum()
                                         + d_mat_b.cwiseProduct(hcore_mat + f_mat_b).sum())
                    : scf_energy_current = d_mat.cwiseProduct(hcore_mat).sum();
                                        // f_mat = hcore_mat at iteration 0

    scf_energies.emplace_back(scf_energy_current);
}

Index Mol::sad_uhf_solver::get_pop_numbers()
{
    int Z = molecule->get_atoms()[atom].get_Z();
    double charge = 0; // TODO;

    constexpr std::array<Index, 8> zvalues = {0, 2, 10, 18, 36, 54, 86, 118};
    // Find the noble gas core.
    auto zval = std::lower_bound(zvalues.begin(), zvalues.end(), Z);
    if (zval == zvalues.end()) 
    {
        std::clog << "\n\n  Error: Z <= 118 for SAD guess only. got " 
                   << Z << " for atom " << atom << "\n\n";
         exit(EXIT_FAILURE);
    } 
  
    if (*zval > Z) zval--;

    // Number of frozen and active orbitals
    if ((*zval) == Z) 
    {
        nfrozen = 0;
        nactive = (*zval) / 2;
    } 
    else 
    {
        nfrozen = (*zval) / 2;
        nactive = (*(++zval)) / 2 - nfrozen;
    }

    // Sanity check: should never happen ?
    if (nactive > norbitals - nfrozen)
        nactive = norbitals - nfrozen;

    pop_index = EigenVector<double>::Zero(norbitals);

    const double frac_occ = std::sqrt(static_cast<double>((nelectrons - nfrozen * 2.0 + charge) / (2.0 * (double) nactive)));

    for (Index i = 0; i < nfrozen; ++i)  pop_index[i] = 1.0;

    for (Index i = nfrozen; i < nactive + nfrozen; ++i)
        pop_index[i] = frac_occ;

    return nfrozen + nactive;
}

void Mol::sad_uhf_solver::calc_potential_matrix()
{
    Index nshells = molecule->get_num_shells();
    const auto& sp = molecule->get_shell_pairs();
    std::unique_ptr<OSNUCLEAR::OSNuclear> osnuc_ptr 
    = std::make_unique<OSNUCLEAR::OSNuclear>(molecule->use_pure_am());

    std::vector<MOLEC::Atom> current_atom;
    current_atom.emplace_back(molecule->get_atoms()[atom]);

    Index mol_nbfs = molecule->get_num_orbitals();
    EigenMatrix<double> molvmat = EigenMatrix<double>::Zero(mol_nbfs, mol_nbfs);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif 
    for (Index i = 0; i < nshells; ++i)
        for (Index j = i; j < nshells; ++j)
        {   // skip over shells we don't need
            if (molecule->use_pure_am())
            {
                if (sp[i * nshells + j].m_s1.get_ids() >=  bf_offset + norbitals) continue;
                else if (sp[i * nshells + j].m_s2.get_ids() >=  bf_offset + norbitals) continue;
                else if (sp[i * nshells + j].m_s1.get_ids() < bf_offset) continue;
                else if (sp[i * nshells + j].m_s2.get_ids() < bf_offset) continue;
            }
            else
            {
                if (sp[i * nshells + j].m_s1.get_idx() >=  bf_offset + norbitals) continue;
                else if (sp[i * nshells + j].m_s2.get_idx() >=  bf_offset + norbitals) continue;
                else if (sp[i * nshells + j].m_s1.get_idx() < bf_offset) continue;
                else if (sp[i * nshells + j].m_s2.get_idx() < bf_offset) continue;
            }

            osnuc_ptr->compute_contracted_shell(molvmat, sp[i * nshells + j], current_atom);
        }

    hcore_mat += molvmat.block(bf_offset, bf_offset, norbitals, norbitals);
}

void Mol::sad_uhf_solver::update_fock_matrix()
{
   ++iteration;
    #ifdef _OPENMP
    #pragma omp parallel for schedule (dynamic)
    #endif
    for (Index i = 0; i < norbitals; ++i) 
    {
        for (Index j = 0; j < norbitals; ++j) 
        {
            f_mat(i, j) = hcore_mat(i, j);
            f_mat_b(i, j) = hcore_mat(i, j);
            for (Index k = 0; k < norbitals; ++k) 
            {
                for (Index l = 0; l < norbitals; ++l) 
                {
                    Index ijkl = index_ijkl(i + bf_offset, j + bf_offset, k + bf_offset, l + bf_offset);
                    Index ikjl = index_ijkl(i + bf_offset, k + bf_offset, j + bf_offset, l + bf_offset);
                    f_mat(i, j)   += d_mat(k, l)   * (moleri(ijkl) - moleri(ikjl)) 
                                   + d_mat_b(k, l) * moleri(ijkl);
                    f_mat_b(i, j) += d_mat_b(k, l) * (moleri(ijkl) - moleri(ikjl)) 
                                   + d_mat(k, l)   * moleri(ijkl);
                }
            }
        }
    }

    diis_ptr->diis_extrapolate(f_mat, f_mat_b, s_mat, d_mat, d_mat_b, s_mat_sqrt, diis_log);
    fp_mat.noalias() = s_mat_sqrt.transpose() * f_mat * s_mat_sqrt;
    fp_mat_b.noalias() = s_mat_sqrt.transpose() * f_mat_b * s_mat_sqrt;

    calc_density_matrix();
}

bool Mol::sad_uhf_solver::sad_run()
{

    bool converged = true;
    double rms_tol = hf_settings::get_sad_rms_tol();
    double e_tol = hf_settings::get_sad_energy_tol();
    const int diis_steps = 6;
    const int maxiter = hf_settings::get_max_sad_iterations();

    diis_ptr.reset();
    diis_ptr = std::make_unique<DIIS_SOLVER::diis_solver>(diis_steps);
    
    for(;;)
    {
        update_fock_matrix();

        if (hf_settings::get_verbosity() > 4 && print_sad)
        {
            std::cout << "\n  iteration: " << iteration << "\n";

            std::cout << "\n  *****************";
            std::cout << "\n     F_alpha";
            std::cout << "\n  *****************\n";
            HFCOUT::pretty_print_matrix<double>(f_mat);

            std::cout << "\n  *****************";
            std::cout << "\n     F_beta";
            std::cout << "\n  *****************\n";
            HFCOUT::pretty_print_matrix<double>(f_mat_b);

            std::cout << "\n  *****************";
            std::cout << "\n  C_alpha";
            std::cout << "\n  *****************\n";
            HFCOUT::pretty_print_matrix<double>(mo_coff);

            std::cout << "\n  *****************";
            std::cout << "\n  C_beta";
            std::cout << "\n  *****************\n";
            HFCOUT::pretty_print_matrix<double>(mo_coff_b);

            std::cout << "\n  *****************";
            std::cout << "\n  D_alpha";
            std::cout << "\n  *****************\n";
            HFCOUT::pretty_print_matrix<double>(d_mat);

            std::cout << "\n  *****************";
            std::cout << "\n  D_beta";
            std::cout << "\n  *****************\n";
            HFCOUT::pretty_print_matrix<double>(d_mat_b);
        }

        double rms_diff_a = diis_ptr->get_current_rms_a();
        double rms_diff_b = diis_ptr->get_current_rms_b();

        if (rms_diff_a < rms_tol && rms_diff_b < rms_tol 
                                 && std::fabs(scf_energy_current - scf_energy_previous) < e_tol)
                break;
        else if ((rms_diff_a + rms_diff_b) < 1E-14) break;
        else if (iteration == maxiter)
        {
            std::cout << "!!! SAD Reached a " << maxiter << " iterations. Failed to converge !!!\n";
            converged = false;
            exit(EXIT_FAILURE);
        }

        if(!converged) break;
    }

    if (hf_settings::get_verbosity() > 3 && print_sad)
    {
        std::cout << "\n  Final iteration: " << iteration << "\n";

        std::cout << "\n  *****************";
        std::cout << "\n    F_alpha";
        std::cout << "\n  *****************\n";
        HFCOUT::pretty_print_matrix<double>(f_mat);

        std::cout << "\n  *****************";
        std::cout << "\n    F_beta";
        std::cout << "\n  *****************\n";
        HFCOUT::pretty_print_matrix<double>(f_mat_b);

        std::cout << "\n  *****************";
        std::cout << "\n    C_alpha";
        std::cout << "\n  *****************\n";
        HFCOUT::pretty_print_matrix<double>(mo_coff);

        std::cout << "\n  *****************";
        std::cout << "\n    C_beta";
        std::cout << "\n  *****************\n";
        HFCOUT::pretty_print_matrix<double>(mo_coff_b);

        std::cout << "\n  *****************";
        std::cout << "\n    D_alpha";
        std::cout << "\n  *****************\n";
        HFCOUT::pretty_print_matrix<double>(d_mat);

        std::cout << "\n  *****************";
        std::cout << "\n    D_beta";
        std::cout << "\n  *****************\n";
        HFCOUT::pretty_print_matrix<double>(d_mat_b);
    }

    if (hf_settings::get_verbosity() > 1 && print_sad)
    {
        std::cout << "*******************************************************************************\n";
        std::cout << "  Iteration   E(e) / Eh            dE(e) / Eh           dRMS             DIIS\n";
        std::cout << "*********************************************************************************\n";

        for(size_t i = 0; i < scf_energies.size(); ++i)
        {
            if (0 == i)
            {
                std::cout << "  ";
                std::cout << std::setw(5) << std::left << i << std::right << std::fixed 
                << std::setprecision(12) << std::setw(22) << scf_energies[0]  << '\n';
            }
            else
            {
                // rms values offset by one (a bit clumsy) since they start at iteration 1 ..
                std::cout << "  ";
                std::cout << std::setw(5) << std::left << i << std::right << 
                std::fixed << std::setprecision(12) << std::setw(22) << scf_energies[i] << 
                std::right << std::fixed << std::setprecision(12) << std::setw(21) << 
                scf_energies[i] - scf_energies[i - 1] << 
                std::right << std::fixed << std::setprecision(12)
                << std::setw(21) << diis_ptr->get_rms_values_ref()[i - 1];

                if (i <= 1)
                    std::cout << "   N";
                else if( i > 1 && 0 == hf_settings::get_diis_size())
                    std::cout << "   N";
                else
                    std::cout << "   Y";
                    
                std::cout << '\n';
            }  
        }
    }

    if (hf_settings::get_verbosity() > 2 && print_sad)
    {
         std::cout << "\n  ****************";
        std::cout << "\n  Final AO density\n";
        std::cout << "  ****************\n";
        HFCOUT::pretty_print_matrix<double>(d_mat + d_mat_b);
    }
    

    if (hf_settings::get_verbosity() > 1 && print_sad)
        std::cout << "\n  UHF computation for unique atom " << atom_name << " end.\n";

    return converged;
}