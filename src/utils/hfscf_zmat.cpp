#include "hfscf_zmat.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <cctype>
#include <cassert>


void ZMAT::ZtoCart::add_row(const std::string& atom_name, const std::string& r_ij_num, const std::string& angle_num, 
                            const std::string& dihedral_num, const std::string& r_ij, const std::string& angle, 
                            const std::string& dihedral_angle)
{
    zparams data;

    const auto check_format_connection_num = [&](const std::string& i, const std::string& msg)
    {
        if (i.empty())
        {
            std::cout << "\n  Error: Invalid Z matrix. connectivity number not found.\n";
            exit(EXIT_FAILURE);
        }

        bool found = !(i.find_first_not_of("0123456789") == std::string::npos);
        
        if(found)
        {
            std::cout << "\n  Error: Invalid Z matrix. " + msg + ".\n";
            std::cout << "  Z matrix line containing: " 
            << atom_name << " " << r_ij_num << " "  << r_ij;
            if (angle_num.length()) std::cout << " " << angle_num << " " << angle;
            if (dihedral_num.length()) std::cout << " " << dihedral_num << " " <<  dihedral_angle;
            std::cout << " for atom number " << m_natoms + 1 << '\n';
            exit(EXIT_FAILURE);
        }
    };

    const auto check_format_real_num = [&](const std::string& i, const std::string& msg)
    {
        if (i.empty())
        {

            std::cout << "\n  Error: Invalid Z matrix. bond, angle, or torsion not found.\n";
            exit(EXIT_FAILURE);
        }

        bool found = !(i.find_first_not_of("0123456789") == std::string::npos); //  must be a number
        const auto pos = i.find_first_of(".");
        found = (pos == std::string::npos); // must contain a period
        if (pos == i.length() - 1)
            found = true; // Can't be the last . must be follwed by a number
        else if(pos == 0)
            found = true; // Can't be the first . must start with a number
    
        if(found)
        {   
            std::cout << "\n  Error: Invalid Z matrix. " + msg + ".\n";
            std::cout << "  Z matrix line containing: " 
            << atom_name << " " << r_ij_num << " "  << r_ij;
            if (angle_num.length()) std::cout << " " << angle_num << " " << angle;
            if (dihedral_num.length()) std::cout << " " << dihedral_num << " " <<  dihedral_angle;
            std::cout << " for atom number " << m_natoms + 1 << '\n';
            if (pos == i.length() - 1 || pos == 0)
                std::cout << "  Note: Real numbers must be expresed as x.y not x. or .y\n";
            exit(EXIT_FAILURE);
        }
    };

    if (std::find_if(atom_name.begin(), atom_name.end(), (int(*)(int))std::isdigit) != atom_name.end())
    {
        std::cout << "\n  Error: Invalid Z matrix. incorrect atom name.\n";
        std::cout << "  Z matrix line containing: "  << atom_name << '\n'; 
        exit(1);
    }

    if(atom_name.empty())
    {
        std::cout << "\n  Error: Invalid Z matrix. atom name not found.\n";
        exit(1);
    }

    data.atom_name = atom_name;

    //Note m_natoms starts at zero at this stage
    if(m_natoms > 0)
    {
        const std::string imsg = "Detected a bond connectivity number, but it is not an integer";
        check_format_connection_num(r_ij_num, imsg);
        data.r_ij_num = std::stoi(r_ij_num); --data.r_ij_num;

        const std::string rij_msg = "Detected a bond length, but it is not a real number";
        check_format_real_num(r_ij, rij_msg);
        data.r_ij = std::stod(r_ij);
    }

    if(m_natoms > 1)
    {
        const std::string amsg = "Detected an angle connectivity number, but it is not an integer";
        check_format_connection_num(angle_num, amsg);
        data.angle_num = std::stoi(angle_num); --data.angle_num;

        const std::string angle_msg = "Detected a bond angle, but it is not a real number";
        check_format_real_num(angle, angle_msg);
        data.angle =  std::stod(angle) * deg_to_rad;
    }

    if(m_natoms > 2)
    {
        const std::string dmsg = "Torsion connectivity number is not an integer";
        check_format_connection_num(dihedral_num, dmsg);
        data.dihedral_num = std::stoi(dihedral_num); --data.dihedral_num;

        const std::string torsion_msg = "Detected a torsion angle, but it is not a real number";
        check_format_real_num(dihedral_angle, torsion_msg);
        data.dihedral_angle =  std::stod(dihedral_angle) * deg_to_rad;
    }
    
    zarray.emplace_back(data);
    ++m_natoms; // includes dummy atoms
    if(atom_name == "X") ++m_dummies;
}

double ZMAT::ZtoCart::udot(const Eigen::Ref<const Vec3>& u1, const Eigen::Ref<const Vec3>& u2) const
{
    double udot12 = u1.dot(u2) / (u2.norm() * u1.norm());
    udot12 = std::max(std::min(udot12, 1.0), -1.0);
    
    return fabs(udot12);
}

Vec3 ZMAT::ZtoCart::ucross(const Eigen::Ref<const Vec3>& r1, const Eigen::Ref<const Vec3>& r2) const
{
    double udot12 = udot(r1, r2);
    udot12 = std::sqrt(1.0 - udot12 * udot12);
    Vec3 ucross12 = r1.cross(r2);
    return ucross12 / udot12;
}

ZMAT::cart ZMAT::ZtoCart::transform_axes(const Eigen::Ref<const Vec3>& r1, 
                                         const Eigen::Ref<const Vec3>& r2,
                                         const Eigen::Ref<const Vec3>& r3) const
{
    Vec3 u12 = (r2 - r1) / (r2 - r1).norm();
    Vec3 u23 = (r3 - r2) / (r3 - r2).norm();

    double udot1223 = udot(u12, u23);

    if(std::fabs<double>(udot1223) > 1.0)
    {
        std::cout << " Error: Z matrix error. Colinear atoms found.";
        exit(EXIT_FAILURE);
    }
    
    Vec3 ucross122312 = ucross(u12, ucross(u23, u12));
    
    cart c;
    c.z = u12;
    c.y = ucross122312;
    c.x = ucross(c.y, c.z);

    return c;
}

void ZMAT::ZtoCart::get_cartesians_from_zarray(EigenMatrix<double>& cart) const
{
    // Note we align with Z to start with
    
    if(!m_natoms)
        return;
        
    EigenMatrix<double> m_cart = EigenMatrix<double>::Zero(m_natoms, 3);

    if (m_natoms > 1) 
    {
        double r_ij = zarray[1].r_ij;
        m_cart(1, 0) = 0.0;
        m_cart(1, 1) = 0.0;
        m_cart(1, 2) = r_ij;
    }

    if (m_natoms > 2)
    {
        int r2_num = zarray[2].r_ij_num;
        double r2_ij = zarray[2].r_ij;
        double angle = zarray[2].angle;

        m_cart(2, 0) = 0.0;
        m_cart(2, 1) = r2_ij * std::sin(angle);
        m_cart(2, 2) = m_cart(r2_num, 2);

        (1 == r2_num) ? m_cart(2, 2) -= r2_ij * std::cos(angle) 
                      : m_cart(2, 2) += r2_ij * std::cos(angle);
    }

    for (int i = 3; i < m_natoms; ++i) 
    {
        const int rconnect = zarray[i].r_ij_num;
        const int aconnect = zarray[i].angle_num;
        const int dconnect = zarray[i].dihedral_num;
 
        const Vec3 c1(m_cart(rconnect, 0), m_cart(rconnect, 1), m_cart(rconnect, 2));
        const Vec3 c2(m_cart(aconnect, 0), m_cart(aconnect, 1), m_cart(aconnect, 2));
        const Vec3 c3(m_cart(dconnect, 0), m_cart(dconnect, 1), m_cart(dconnect, 2));

        const ZMAT::cart axes = transform_axes(c1, c2, c3);

        const Vec3 bond = Vec3(zarray[i].r_ij * sin(zarray[i].angle) * sin(zarray[i].dihedral_angle),
                               zarray[i].r_ij * sin(zarray[i].angle) * cos(zarray[i].dihedral_angle),
                               zarray[i].r_ij * cos(zarray[i].angle));

        const Vec3 displace = Vec3(bond[0]  * axes.x[0] +  bond[1]  * axes.y[0] + bond[2]  * axes.z[0],
                                   bond[0]  * axes.x[1] +  bond[1]  * axes.y[1] + bond[2]  * axes.z[1], 
                                   bond[0]  * axes.x[2] +  bond[1]  * axes.y[2] + bond[2]  * axes.z[2]);

        for (int j = 0; j < 3; ++j) 
            m_cart(i, j) = c1(j) + displace(j);
    }

    double unit_convert = 1.0;
    if(m_unit_type == "angstrom")
        unit_convert = angstrom_to_bohr;

    cart = EigenMatrix<double>(m_natoms - m_dummies, 3);

    int new_row = 0;
    for(int row = 0; row < m_natoms; ++row)
    {
        if("X" == zarray[row].atom_name) // Skip dummy atoms
            continue;

        cart(new_row, 0) = m_cart(row, 0) * unit_convert;
        cart(new_row, 1) = m_cart(row, 1) * unit_convert;
        cart(new_row, 2) = m_cart(row, 2) * unit_convert;
        
        ++new_row;
    }
}
