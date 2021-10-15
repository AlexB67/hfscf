#include "cart_to_int.hpp"
#include "../molecule/hfscf_elements.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../pretty_print/hfscf_pretty_print.hpp"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <boost/optional/optional_io.hpp>

// wrappers added to irc for hfscf
// libirc was pretty much left stock, a few minor changhes
// were made though
using HF_SETTINGS::hf_settings;
using namespace irc;
using Eigen::Index;

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void CART_INT::Cart_int::do_internal_coord_analysis()
{
    // create irc molecule object
    const Eigen::Ref<const EigenMatrix<double> > cart = m_mol->get_geom();
    const auto& zvals = m_mol->get_z_values();
    const int natoms = static_cast<int>(m_mol->get_atoms().size());

    molecule.clear(); // in case we were called before, prolly not needed
    m_bond.clear(); m_angle.clear(); m_linear_angle.clear(); 
    m_dihedral.clear(); m_out_of_plane_bend.clear();

    for(int i = 0; i < natoms; ++i)
       molecule.push_back({symbols[zvals[i]], {cart(i, 0), cart(i, 1), cart(i, 2)}});

    mat dd{distances<vec3, mat>(molecule)};
    UGraph adj{adjacency_matrix(dd, molecule)};
    mat dist{distance_matrix<mat>(adj)};

    m_bond = std::vector<Bond>{bonds(dist, molecule)};
    m_angle = std::vector<Angle>{angles(dist, molecule)};
    m_dihedral = std::vector<Dihedral>{dihedrals(dist, molecule)};
    m_linear_angle = std::vector<LinearAngle<vec3>>{linear_angles(dist, molecule)};
    m_out_of_plane_bend =  std::vector<OutOfPlaneBend>{out_of_plane_bends(dist, molecule)};
    // doesn't do anything for now
    if (is_planar() || hf_settings::get_geomopt_constrain__oop_angles())
        for (size_t i = 0; i < m_out_of_plane_bend.size(); ++i)
            m_out_of_plane_bend[i].constraint = connectivity::Constraint::constrained;
    
    if (hf_settings::get_geomopt_constrain__dihedral_angles())
        for (size_t i = 0; i < m_dihedral.size(); ++i)
            m_dihedral[i].constraint = connectivity::Constraint::constrained;
    
    if (hf_settings::get_geomopt_constrain__angles())
        for (size_t i = 0; i < m_angle.size(); ++i)
            m_angle[i].constraint = connectivity::Constraint::constrained;
    // end do nothing
    num_irc = m_bond.size() + m_angle.size() + m_dihedral.size() + m_linear_angle.size() + m_out_of_plane_bend.size();

    ircs = std::make_unique<IRC<vec3, vec, mat> >(molecule); // TODO be part of constructor

    B_mat = wilson_matrix<vec3, vec, mat>(to_cartesian<vec3, vec>(molecule), m_bond, m_angle, m_dihedral,
                                                                  m_linear_angle, m_out_of_plane_bend);
}

void CART_INT::Cart_int::print_bonding_info(bool print_projector, bool print_wilson, std::string new_title) const
{
    int verbosity_level = (hf_settings::get_geom_opt().length()) ? 2 : 3;

    if(hf_settings::get_verbosity() > verbosity_level)
    {
        if(print_wilson)
        {
            std::cout << "\n  ************************************\n";
            std::cout << "  *        Wilson B Matrix           *\n";
            std::cout << "  ************************************\n";
            HFCOUT::pretty_print_matrix<double>(B_mat);
        }

        if(print_projector)
        {
            std::cout << "  **********************************\n";
            std::cout << "  *  Projection matrix             *\n";
            std::cout << "  **********************************\n";
            mat P = wilson::projector(B_mat);
            HFCOUT::pretty_print_matrix<double>(P);
        }
    }

    if(new_title.length()) std::cout << new_title << num_irc << "\n";
    else std::cout << "\n  Number of (redundant) internal\n  coordinates: " << num_irc << "\n";

    print_bonds<vec3, vec>(to_cartesian<vec3, vec>(molecule), m_bond);
    print_angles<vec3, vec>(to_cartesian<vec3, vec>(molecule), m_angle);
    print_linear_angles<vec3, vec>(to_cartesian<vec3, vec>(molecule), m_linear_angle);
    print_dihedrals<vec3, vec>(to_cartesian<vec3, vec>(molecule), m_dihedral);
    print_out_of_plane_bends<vec3, vec>(to_cartesian<vec3, vec>(molecule), m_out_of_plane_bend);
    std::cout << "*************************************\n";
}

void CART_INT::Cart_int::get_guess_hessian(EigenMatrix<double>& hessian, const std::string& type) const
{
    hessian =  EigenMatrix<double>::Zero(num_irc, num_irc);

    if(type == "SIMPLE")
    { 
        constexpr double k_bond = 0.5;
        constexpr double k_angle = 0.2;
        constexpr double k_dihedral = 0.1;
        std::size_t offset{0};

        for (std::size_t i{0}; i < m_bond.size(); ++i) hessian(i, i) = k_bond;

        offset = m_bond.size();
        for (std::size_t i{0}; i < m_angle.size(); i++) hessian(i + offset, i + offset) = k_angle;

        offset = m_bond.size() + m_angle.size();
        for (std::size_t i{0}; i < m_dihedral.size(); i++) hessian(i + offset, i + offset) = k_dihedral;

        offset = m_bond.size() + m_angle.size() + m_dihedral.size();
        for (std::size_t i{0}; i < m_linear_angle.size(); i++) hessian(i + offset, i + offset) = k_angle;

        return;
    }
    else if(type != "SCHLEGEL")
    {
        std::cout << "\n\n  Error: Invalid request for guess hessian:  " << type << "\n";
        std::cout << "         SIMPLE and SCHLEGEL supported only.\n\n";
        exit(EXIT_FAILURE);
    }

    constexpr auto get_period_from_Z = [](size_t Z) -> size_t 
    {
        if (Z <= 2)
            return 1;
        else if (Z <= 10)
            return 2;
        else if (Z <= 18)
            return 3;
        else if (Z <= 36)
            return 4;
        else
            return 5;
    };

    vec x_c{to_cartesian<vec3, vec>(molecule)};

    for (size_t i = 0; i < m_bond.size(); ++i) 
    {
        const auto idx_i = m_bond[i].i;
        const auto idx_j = m_bond[i].j;
        vec3 p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
        vec3 p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
        auto periodZ1 = get_period_from_Z(molecule[idx_i].atomic_number.atomic_number);
        auto periodZ2 = get_period_from_Z(molecule[idx_j].atomic_number.atomic_number);
        const double r12 = connectivity::distance<vec3>(p1, p2);

        constexpr double AA = 1.734;
        double BB = 0.0;

        if (periodZ1 == 1)
        {
            switch (periodZ2)
            {
                case 1:
                    BB = -0.244;
                    break;
                case 2:
                    BB = 0.352;
                    break;
                default:
                     BB = 0.660;
                     break;
            }
        }
        else if (periodZ1 == 2)
        {
            switch (periodZ2)
            {
                case 1:
                    BB = 0.352;
                    break;
                case 2:
                    BB = 1.085;
                    break;
                default:
                     BB = 1.522;
                     break;
            }
        }
        else
        {
            switch (periodZ2)
            {
                case 1:
                    BB = 0.660;
                    break;
                case 2:
                    BB = 1.522;
                    break;
                default:
                    BB = 2.068;
                    break;
            }
        }

        Eigen::Index index = static_cast<Eigen::Index>(i);
        hessian(index, index) = AA / ((r12 - BB) * (r12 - BB) * (r12 - BB));
    }

    for (size_t i = 0; i < m_angle.size(); ++i)
    {
        const auto idx_i = m_angle[i].i;
        const auto idx_k = m_angle[i].k;
        const auto Z1 = molecule[idx_i].atomic_number.atomic_number;
        const auto Z3 = molecule[idx_k].atomic_number.atomic_number;

        Eigen::Index index = static_cast<Eigen::Index>(i + m_bond.size());

        (Z1 == 1 || Z3 == 1) ? hessian(index, index) = 0.16
                             : hessian(index, index) = 0.25;
    }

    for (size_t i = 0; i < m_dihedral.size(); ++i)
    {
        const auto idx_j = m_dihedral[i].j;
        const auto idx_k = m_dihedral[i].k;
        const auto Z2 = molecule[idx_j].atomic_number.atomic_number;
        const auto Z3 = molecule[idx_k].atomic_number.atomic_number;
        vec3 p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
        vec3 p3 = {x_c(3 * idx_k), x_c(3 * idx_k + 1), x_c(3 * idx_k + 2)};
        const double r23 = connectivity::distance<vec3>(p2, p3);
        const double R_covalent = (irc::periodic_table::covalent_radii[Z2] 
                                + irc::periodic_table::covalent_radii[Z3] );
        constexpr double a = 0.0023;
        constexpr double b = 0.07;

        Eigen::Index index = static_cast<Eigen::Index>(i + m_bond.size() + m_angle.size());
        (r23 < (R_covalent + a / b)) ? hessian(index, index) = a - (b * (r23 - R_covalent))
                                     : hessian(index, index) = a;
    }

    for (size_t i = 0; i < m_linear_angle.size(); ++i)
    {
        constexpr double k_angle = 0.2;
        Eigen::Index index = static_cast<Eigen::Index>(i + m_bond.size() + m_angle.size() + m_dihedral.size());
        hessian(index, index) = k_angle;
    }

    for (size_t i = 0; i < m_out_of_plane_bend.size(); ++i)
    {
        constexpr double oofp_angle = 0.12; //should be  0.045; behaves better with current
        Eigen::Index index = static_cast<Eigen::Index>(i + m_bond.size() + m_angle.size() 
                                                         + m_dihedral.size() + m_linear_angle.size());
        const auto idx_c =  m_out_of_plane_bend[i].c;
        const auto idx_i =  m_out_of_plane_bend[i].i;
        const auto idx_j =  m_out_of_plane_bend[i].j;
        const auto idx_k =  m_out_of_plane_bend[i].k;
        vec3 pc = {x_c(3 * idx_c), x_c(3 * idx_c + 1), x_c(3 * idx_c + 2)};
        vec3 p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
        vec3 p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
        vec3 p3 = {x_c(3 * idx_k), x_c(3 * idx_k + 1), x_c(3 * idx_k + 2)};
        vec3 r1 = p1 - pc; vec3 r2 = p2 - pc; vec3 r3 = p3 - pc;

        vec3 r23 = linalg::cross(r2, r3);
        const double d = 1.0 - r1.dot(r23) / (r1.norm() * r2.norm() * r3.norm());
        // for planar, d should always be 1, the above is as such redundant for this case
        hessian(index, index) = oofp_angle * d * d * d * d;
    } 


}

void CART_INT::Cart_int::do_bfgs_update(EigenMatrix<double>& hessian, 
                                        const Eigen::Ref<const EigenVector<double> >& del_irc_grad,
                                        const Eigen::Ref<const EigenVector<double> >& del_irc) const
{
    double dot_irc_irc_grad = del_irc.dot(del_irc_grad);
    EigenVector<double> H_del_irc = hessian * del_irc;
    double dot_del_irc_H_del_irc = del_irc.dot(H_del_irc);

    //double rho = del_irc_grad.dot(del_irc) / (del_irc_grad.norm() * del_irc.norm());

    for(Index i = 0; i < hessian.rows(); ++i)
        for(Index j = i; j < hessian.cols(); ++j)
        {
            
            hessian(i, j) += del_irc_grad(i) * del_irc_grad(j) / dot_irc_irc_grad
                           - H_del_irc(i) * H_del_irc(j) / dot_del_irc_H_del_irc;
            hessian(j, i)  = hessian(i, j);
        }

    Eigen::LLT<Eigen::MatrixXd> lltOfA(hessian);

    if (lltOfA.info() == Eigen::NumericalIssue) // hopefully this shouldn't happen too often
    {
        const std::string& hess_type = hf_settings::get_geom_opt_guess_hessian();
        std::cout << "\n\n  Positive definite test failed. BFGS Hessian step discarded.\n";
        std::cout << "  Using new " << hess_type << " guess instead at current geometry.\n";
        get_guess_hessian(hessian, hess_type);
    }
}

void CART_INT::Cart_int::rfo_step(const Eigen::Ref<const EigenMatrix<double> >& gradient_cart,
                                  EigenVector<double>& irc_coords, EigenVector<double>& irc_grad, 
                                  EigenMatrix<double>& hessian) const
{
    EigenVector<double> irc_grad_old = irc_grad;
    EigenVector<double> irc_coords_old = irc_coords;

    irc_grad = get_irc_gradient(gradient_cart, false);
    
    vec x_c{to_cartesian<vec3, vec>(molecule)};
    irc_coords = ircs->cartesian_to_irc(x_c);

    if(!irc_coords_old.size()) // compute new guess, first time visit in geom opt
    {
        get_guess_hessian(hessian, hf_settings::get_geom_opt_guess_hessian());
        
        std::cout << '\n';
        std::cout << "  ******************************************\n";
        
        if (hf_settings::get_geom_opt_guess_hessian() == "SCHLEGEL")
            std::cout << "  *  Diagonal Guess Hessian (SCHLEGEL '84) *\n";
        else
            std::cout << "  *     Diagonal Guess Hessian (SIMPLE)    *\n";

        std::cout << "  ******************************************\n";

        for (Index i = 0; i < hessian.outerSize(); ++i)
        {
            std::cout << std::setw(10) << std::right << std::setprecision(6) << hessian(i, i);
            if((i + 1) % 8 == 0) std::cout << '\n';
        }

        std::cout << '\n';
    }
    else // update from previous
    {
        EigenVector<double> del_irc_grad  = irc_grad - irc_grad_old;
        EigenVector<double> del_irc_coord = irc_coords - irc_coords_old;
            
        std::cout << '\n'
        << "**********************************************************************************************\n"
        << "   IR coordinate & gradient info:\n"
        << "   #     q_previous     q_current      delq           g_previous     g_current      delg \n"
        << "**********************************************************************************************\n";

        for(Index i = 0; i < del_irc_grad.size(); ++i)
        {
            Index offset = m_bond.size() + m_angle.size();
            // work around dihedral flip (I suspect a bug in irc library but this works for now) 
            // TODO need irc fix
            if (i >= offset && i <  offset + (Index) m_dihedral.size())
            {
                if (irc_coords_old(i) < 0 && irc_coords(i) > 0) irc_coords(i) = -irc_coords(i);
                else if (irc_coords_old(i) > 0 && irc_coords(i) < 0) irc_coords(i) = -irc_coords(i);
                
                del_irc_coord(i) = irc_coords(i) - irc_coords_old(i);
            }

            std::cout << std::setw(4) << std::right  << i + 1
            << std::setw(15) << std::right << std::setprecision(9) << irc_coords_old(i)
            << std::setw(15) << std::right << std::setprecision(9) << irc_coords(i)
            << std::setw(15) << std::right << std::setprecision(9) << del_irc_coord(i)
            << std::setw(15) << std::right << std::setprecision(9) << irc_grad_old(i)
            << std::setw(15) << std::right << std::setprecision(9) << irc_grad(i)
            << std::setw(15) << std::right << std::setprecision(9) << del_irc_grad(i) << '\n';
        }

        do_bfgs_update(hessian, del_irc_grad, del_irc_coord);

        if(hf_settings::get_verbosity() > 2)
        {
            std::cout << '\n';
            std::cout << "  **********************************\n";
            std::cout << "  *  Hessian update from BFGS step *\n";
            std::cout << "  **********************************\n";
            HFCOUT::pretty_print_matrix<double>(hessian);
        }
    }

    std::cout << "\n  Projecting redundancies now (forces and hessian):\n";

    EigenMatrix<double> rfo_mat = EigenMatrix<double>::Zero(hessian.rows() + 1, hessian.rows() + 1);
    vec irc_grad_P = get_irc_gradient(gradient_cart, true);
    mat tmp_hess = ircs->projected_hessian(hessian);

    for (Index i = 0; i < hessian.rows(); ++i)
        for (Index j = i; j < hessian.cols(); ++j) 
        {
            rfo_mat(i, j) = tmp_hess(i, j);
            if (i == j) rfo_mat(i, hessian.cols()) = rfo_mat(hessian.rows(), i) = irc_grad_P(i);
            rfo_mat(j, i) = rfo_mat(i, j);
        }
 
    if(hf_settings::get_verbosity() > 1)
    {
        std::cout << '\n';
        std::cout << "  ********************************\n";
        std::cout << "  *           RFO Matrix:        *\n";
        std::cout << "  ********************************\n";
        HFCOUT::pretty_print_matrix<double>(rfo_mat);
    }

    Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solver(rfo_mat);
    EigenMatrix<double> rfo_evecs = solver.eigenvectors();
    EigenVector<double> rfo_evals = solver.eigenvalues();

    std::cout << "\n  *************************\n";
    std::cout << "  *    RFO Eigen values   *\n";
    std::cout << "  *************************\n";

    Index pos = 0; double most_negative = std::numeric_limits<double>::max();
    for(Index i = 0; i < rfo_evals.size(); ++i)
    {
        if (rfo_evals(i) < most_negative)
        {
            most_negative = rfo_evals(i);
            pos = i;
        }

        std::cout << std::setw(16) << std::right << std::setprecision(12) << rfo_evals(i);
        if((i + 1) % 5 == 0) std::cout << '\n';
    }
    
    EigenVector<double> irc_step =  EigenVector<double>(irc_grad.size());
    for(Index i = 0; i < irc_grad.size(); ++i)
        irc_step(i) = rfo_evecs(i, pos) / rfo_evecs(irc_grad.size(), pos);

    std::cout << "\n\n**********************************************************\n";
    std::cout << "  Selected new RFO\n";
    std::cout << "  Eigenvector at column " << pos << "\n\n";
    std::cout << "   #    Eigenvector       Normalized        Step\n";
    std::cout << "**********************************************************\n";

    for(Index i = 0; i < rfo_evals.size(); ++i)
    {
        std::cout << std::setw(4) << std::right << i + 1
                  << std::setw(18) << std::right << std::setprecision(12) 
                  << rfo_evecs(i, pos) << std::setw(18) << std::right
                  << rfo_evecs(i, pos) / rfo_evecs(irc_grad.size(), pos);
        if(i < irc_grad.size())
            std::cout << std::setw(18) << std::right << std::setprecision(12) << irc_step(i);
        
        std::cout << '\n';
    }

    // project redundancies in q
    // EigenMatrix<double> G = B_mat.transpose() * B_mat;
    // EigenMatrix<double> Ginv;
    // generalised_inverse(G, Ginv);
    // EigenMatrix<double> Binv = Ginv * B_mat.transpose();
    EigenMatrix<double> P = ircs->get_P(); // = B_mat * Binv;

    vec istep = P * irc_step;

    std::cout << "\n  Projecting step redundancies:";
    std::cout << "\n**********************************************************\n";
    std::cout << "   #    dq                dq(projected)     diff\n";
     std::cout << "**********************************************************\n";
    for(Index i = 0; i < irc_step.size(); ++i)
    {
        std::cout << std::setw(4) << std::right << i + 1
                  << std::setw(18) << std::right << std::setprecision(12) 
                  << irc_step(i) << std::setw(18) << std::right
                  << istep(i) << std::setw(18) << std::right << irc_step(i) - istep(i) << "\n";
    }

    irc_to_cartesian(irc_coords, istep);
}

EigenVector<double> 
CART_INT::Cart_int::get_irc_gradient(const Eigen::Ref<const EigenMatrix<double> >& gradient_cart,
                                     bool project) const
{
    const int natoms = static_cast<int>(m_mol->get_atoms().size());
    EigenMatrix<double> cart = m_mol->get_geom_copy();

    vec grad_xc = vec::Zero(3 * natoms);
    for(int i = 0; i < natoms; ++i)
    {
        grad_xc[3 * i] = gradient_cart(i, 0);
        grad_xc[3 * i + 1] = gradient_cart(i, 1);
        grad_xc[3 * i + 2] = gradient_cart(i, 2);
    }

    if(project)
    {
        const vec irc_grad_p = ircs->grad_cartesian_to_projected_irc(grad_xc);
        return irc_grad_p;
    }
    
    const vec irc_grad = ircs->grad_cartesian_to_irc(grad_xc); //transformation::gradient_cartesian_to_irc<vec, mat>(grad_xc, B_mat);
    return irc_grad;
}

EigenVector<double> CART_INT::Cart_int::get_cartesian_to_irc() const
{
    vec x_c{to_cartesian<vec3, vec>(molecule)};
    const vec irc = ircs->cartesian_to_irc(x_c);
    return irc;
}

void CART_INT::Cart_int::irc_to_cartesian(const Eigen::Ref<const EigenVector<double> >& irc,
                                          const Eigen::Ref<const EigenVector<double> >& del_irc,
                                          bool do_geometry_analysis) const
{
    vec x_c_previous{to_cartesian<vec3, vec>(molecule)};
    const double tol = (hf_settings::get_geom_opt_algorithm() == "CGSD") ? 1E-10 : 1E-08;
    const size_t cycles = (hf_settings::get_geom_opt_algorithm() == "CGSD") ? 40U : 30U;

    const auto result = ircs->irc_to_cartesian(irc, del_irc, x_c_previous, cycles, tol);
    
    if(!result.converged)
        std::cout << "\n  Warning: IRC to cartesian back-step failed to converge after " 
                  <<  result.n_iterations << " iterations.\n";
    else
    {
        std::cout << "\n  IRC to cartesian back-step converged in " 
                  <<  result.n_iterations << " iterations.\n  New cartesian coordinates: \n";
    }
    
    const int natoms = static_cast<int>(m_mol->get_atoms().size());
    EigenMatrix<double> new_geom = EigenMatrix<double>::Zero(natoms, 3);

    for (int i = 0; i < natoms; ++i) 
    {
        new_geom(i, 0) = result.x_c[3 * i];
        new_geom(i, 1) = result.x_c[3 * i + 1];
        new_geom(i, 2) = result.x_c[3 * i + 2];
    }

    vec irc_back_transform = ircs->cartesian_to_irc(result.x_c);
    vec error = vec(irc.size());
    for (Index i = 0; i < irc.size(); ++i)
        error(i) = irc_back_transform(i) - irc(i) - del_irc(i);
    
    // same issue here with dihedral flips, 
    // but it's only for printing, no damage done, see comments in rfo_step
    std::cout << "\n*****************************************\n";
    std::cout << "          Back transform validation of q\n";
    std::cout << "   #    q_expected        error\n";
    std::cout << "*****************************************\n";
    for(Index i = 0; i < irc.size(); ++i)
    {
        std::cout << std::setw(4) << std::right << i + 1
                  << std::setw(18) << std::right << std::setprecision(12) 
                  << irc(i) + del_irc(i) << std::setw(18) << std::right
                  << error(i) << "\n";
    }

    if (hf_settings::get_use_symmetry())
    {   // Hack: Keeps things under control since we are using symmetry, 
        // but not symmetrized coordinates. TODO symmetrize coords
        // Doesn't always work, but better than without as things stand.
        // Often the current geom opt routine works best without symmetry.
        hf_settings::set_symmetrize_geom(true);
        do_geometry_analysis = true;
    }
    m_mol->update_geom(new_geom, do_geometry_analysis);

    std::cout << "\n\n***********************************************************************\n";
    std::cout << "  Atom  X / bohr          Y / bohr          Z / bohr          Mass\n";
	std::cout << "***********************************************************************\n";

    const std::vector<int>& zval = m_mol->get_z_values();
    for(int i = 0; i < natoms; ++i)
    {
        std::cout << "  " << ELEMENTDATA::atom_names[zval[i] - 1];
        std::cout << std::right << std::fixed << std::setprecision(12) 
                  << std::setw(18) << m_mol->get_geom()(i, 0)
                  << std::right << std::fixed << std::setw(18) << m_mol->get_geom()(i, 1)
                  << std::right << std::fixed << std::setw(18) << m_mol->get_geom()(i, 2)
                  << std::right << std::fixed << std::setw(13) << std::setprecision(7) 
                  << ELEMENTDATA::masses[zval[i] - 1] << '\n';
    }
    
    if (!hf_settings::get_geom_opt_write_trajectory()) return;

    std::ofstream outfile;
    const std::string& trajectory_file = hf_settings::get_geom_opt_trajectory_file();
    outfile.open(trajectory_file, std::ios_base::app);

    outfile << natoms << "\n\n";
    for(int i = 0; i < natoms; ++i)
    {
        outfile   << ELEMENTDATA::atom_names[zval[i] - 1]
                  << std::right << std::fixed << std::setprecision(12)
                  << std::setw(18) << m_mol->get_geom()(i, 0) * conversion::bohr_to_angstrom
                  << std::right << std::fixed << std::setw(18) 
                  << m_mol->get_geom()(i, 1) * conversion::bohr_to_angstrom
                  << std::right << std::fixed << std::setw(18) 
                  << m_mol->get_geom()(i, 2) * conversion::bohr_to_angstrom << "\n";
    }


    outfile.close();
}

// void CART_INT::Cart_int::generalised_inverse(const Eigen::Ref<const EigenMatrix<double> >& in_mat, 
//                                              EigenMatrix<double>& inv) const
// {
//     Eigen::SelfAdjointEigenSolver<EigenMatrix<double> > solve(in_mat);
//     EigenVector<double> evals = solve.eigenvalues();
//     EigenMatrix<double> evecs = solve.eigenvectors();
    
//     double det = 1.0;
//     for(Index i = 0; i < evals.size(); ++i) det *= evals(i);

//     EigenMatrix<double> diaginv = EigenMatrix<double>::Zero(evals.size(), evals.size());

//     for(Index i = 0; i < evals.size(); ++i)
//         if(fabs(evals(i)) > 1.E-10)
//             diaginv(i, i) = 1.0 / evals(i);
    
//     inv = evecs * diaginv * evecs.transpose();
// }

bool CART_INT::Cart_int::is_planar() const
{
    const int natoms = static_cast<int>(m_mol->get_atoms().size());
    constexpr double cutoff = 1.0E-08;

    EigenMatrix<double> geom = m_mol->get_geom();
    std::vector<bool> out_of_plane = std::vector<bool>(3, false);

    for (int cart_dir = 0; cart_dir < 3; ++cart_dir) 
    {
        for (int i = 0; i < natoms; ++i)
            if (std::fabs<double>(geom(i, cart_dir)) > cutoff)
            {
                out_of_plane[cart_dir] = true;
                break;
            }
    }

    if (!out_of_plane[0] || !out_of_plane[1] || !out_of_plane[2]) 
        return true;
    
    return false;
}
    // mat Gmat = B_mat * B_mat.transpose();
    // call gen_inverse
    // std::cout << "\n\n" << ainv << "\n\n";

    // Index natoms = (Index) m_mol->get_atoms().size();
    // vec grad_xc = vec(3 * natoms);
    // for(int i = 0; i < natoms; ++i)
    // {
    //     grad_xc[3 * i] = gradient_cart(i , 0);
    //     grad_xc[3 * i + 1] = gradient_cart(i , 1);
    //     grad_xc[3 * i + 2] = gradient_cart(i , 2);
    // }

    // vec gq = ainv * B_mat * grad_xc;

    // std::cout << "\n my method\n"<< gq << "\n\n";
    // vec ircgrad =  get_irc_gradient(gradient_cart, true);
    // std::cout << "\n irc method\n"<< ircgrad << "\n\n";

    // exit(0);

    // coord projection
    // EigenMatrix<double> G = B_mat.transpose() * B_mat;
        // EigenMatrix<double> Ginv;
        // generalised_inverse(G, Ginv);
        // EigenMatrix<double> Binv = Ginv * B_mat.transpose();
        // EigenMatrix<double> Pb = B_mat * Binv;
        // HFCOUT::pretty_print_matrix<double>(Pb);

        // std::cout << "\n del projected \n" << Pb * del_irc_coord   << "\n\n";
        // std::cout << Pb * irc_coords << "\n\n";

    // EigenMatrix<double> G = B_mat.transpose() * B_mat;
            // EigenMatrix<double> Ginv;
            // generalised_inverse(G, Ginv);
            // EigenMatrix<double> PP = G * Ginv;
            // HFCOUT::pretty_print_matrix<double>(PP);

            // EigenMatrix<double> Binv = Ginv * B_mat.transpose();
            // EigenMatrix<double> Pb = B_mat * Binv;
            // HFCOUT::pretty_print_matrix<double>(Pb);

            // vec x_c{to_cartesian<vec3, vec>(molecule)};
            // vec irc_c = ircs->cartesian_to_irc(x_c);

            // std::cout << "\n unprojected \n" << irc_c << "\n\n";

            // std::cout << "\n projected \n" << Pb * irc_c << "\n\n";

            // exit(0);

// gradient
//  EigenMatrix<double> G = B_mat * B_mat.transpose();
//         EigenMatrix<double> Ginv;
//         generalised_inverse(G, Ginv);

//         std::cout << "\n grad hfscf \n" << Ginv * B_mat * grad_xc << "\n\n";

//         const vec irc_g = transformation::gradient_cartesian_to_irc<vec, mat>(grad_xc, B_mat);
//         std::cout << "\n grad no P IRC \n" << irc_g << "\n\n";
//         exit(0);