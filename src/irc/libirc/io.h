#ifndef IRC_IO_H
#define IRC_IO_H

#include "connectivity.h"
#include "conversion.h"
#include "molecule.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace irc {

namespace io {

/// Print bonds
template<typename Vector3, typename Vector>
void print_bonds(const Vector& x_c,
                 const std::vector<connectivity::Bond>& bonds,
                 std::ostream& out = std::cout) {

  const std::size_t n_bonds{bonds.size()};

  out << "*************************************\n";
  out << "  " << n_bonds << " bonds / \u212B" << '\n';
  out << "*************************************\n";

  // Atomic positions
  Vector3 p1{0., 0., 0.}, p2{0., 0., 0.};

  // Indices
  std::size_t idx_i{0}, idx_j{0};

  out.precision(4);
  out.fill(' ');

  for (std::size_t i{0}; i < n_bonds; i++) {
    idx_i = bonds[i].i;
    idx_j = bonds[i].j;

    // Print indices
    out << std::left << "  (";
    out.width(4);
    out << std::right << idx_i << ',';
    out.width(4);
    out << std::right << idx_j << ')';

    // Get positions
    p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
    p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};

    // Print distance
    out << std::setw(8) << std::fixed << " " << std::right
        << connectivity::distance(p1, p2) * tools::conversion::bohr_to_angstrom << '\n';
  }
}

/// Print angles
template<typename Vector3, typename Vector>
void print_angles(const Vector& x_c,
                  const std::vector<connectivity::Angle>& angles,
                  std::ostream& out = std::cout) {
  
  if (!angles.size()) return;
  
  const std::size_t n_angles{angles.size()};

  out << "*************************************\n";
  out << "  " << n_angles << " angles / \u00B0:" << std::endl;
  out << "*************************************\n";

  // Atomic positions
  Vector3 p1{0., 0., 0.}, p2{0., 0., 0.}, p3{0., 0., 0.};

  // Indices
  std::size_t idx_i{0}, idx_j{0}, idx_k{0};

  out.precision(2);
  out.fill(' ');

  for (std::size_t i{0}; i < n_angles; i++) {
    idx_i = angles[i].i;
    idx_j = angles[i].j;
    idx_k = angles[i].k;

    // Print indices
    out << std::left << "  (";
    out.width(4);
    out << std::right << idx_i << ',';
    out.width(4);
    out << std::right << idx_j << ',';
    out.width(4);
    out << std::right << idx_k << ')';

    // Get positions
    p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
    p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
    p3 = {x_c(3 * idx_k), x_c(3 * idx_k + 1), x_c(3 * idx_k + 2)};

    // Print distance
    out << std::setw(8) << std::fixed << std::right
        << connectivity::angle(p1, p2, p3) * tools::conversion::rad_to_deg
        << std::endl;
  }
}

/// Print dihedral angles
template<typename Vector3, typename Vector>
void print_dihedrals(const Vector& x_c,
                     const std::vector<connectivity::Dihedral>& dihedrals,
                     std::ostream& out = std::cout) {
  
  if (!dihedrals.size()) return;
  
  const std::size_t n_dihedrals{dihedrals.size()};

  out << "*************************************\n";
  out << "  " << n_dihedrals << " dihedrals / \u00B0" << std::endl;
  out << "*************************************\n";

  // Atomic positions
  Vector3 p1{0., 0., 0.}, p2{0., 0., 0.}, p3{0., 0., 0.}, p4{0., 0., 0.};

  // Indices
  std::size_t idx_i{0}, idx_j{0}, idx_k{0}, idx_l{0};

  out.precision(2);
  out.fill(' ');

  for (std::size_t i{0}; i < n_dihedrals; i++) {
    idx_i = dihedrals[i].i;
    idx_j = dihedrals[i].j;
    idx_k = dihedrals[i].k;
    idx_l = dihedrals[i].l;

    // Print indices
    out << std::left << "  (" ;
    out.width(4);
    out << std::right << idx_i << ',';
    out.width(4);
    out << std::right << idx_j << ',';
    out.width(4);
    out << std::right << idx_k << ',';
    out.width(4);
    out << std::right << idx_l << ')';

    // Get positions
    p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
    p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
    p3 = {x_c(3 * idx_k), x_c(3 * idx_k + 1), x_c(3 * idx_k + 2)};
    p4 = {x_c(3 * idx_l), x_c(3 * idx_l + 1), x_c(3 * idx_l + 2)};

    // Print distance

    out << std::setw(9) << std::fixed << std::right
        << connectivity::dihedral(p1, p2, p3, p4) *
               tools::conversion::rad_to_deg
        << std::endl;
  }
}

/// Print linear angles
template<typename Vector3, typename Vector>
void print_linear_angles(
    const Vector& x_c,
    const std::vector<connectivity::LinearAngle<Vector3>>& angles,
    std::ostream& out = std::cout) {
  
  if (!angles.size()) return;

  const std::size_t n_angles{angles.size()};

  out << "*************************************\n";
  out << "  " << n_angles << " linear angles / \u00B0" << std::endl;
  out << "*************************************\n";

  // Atomic positions
  Vector3 p1{0., 0., 0.}, p2{0., 0., 0.}, p3{0., 0., 0.};

  // Indices
  std::size_t idx_i{0}, idx_j{0}, idx_k{0};

  out.precision(2);
  out.fill(' ');

  for (std::size_t i{0}; i < n_angles; i++) {
    idx_i = angles[i].i;
    idx_j = angles[i].j;
    idx_k = angles[i].k;

    // Print indices
    out << std::left << "  (";
    out.width(4);
    out << std::right << idx_i << ',';
    out.width(4);
    out << std::right << idx_j << ',';
    out.width(4);
    out << std::right << idx_k << ')';
    out.width(7);
    out << std::left << " " + to_string(angles[i].tag);

    // Get positions
    p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
    p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
    p3 = {x_c(3 * idx_k), x_c(3 * idx_k + 1), x_c(3 * idx_k + 2)};

    // Print distance
    out << std::setw(9) << std::fixed << std::right
        << connectivity::angle(p1, p2, p3) * tools::conversion::rad_to_deg
        << std::endl;
  }
}

/// Print out of plane bends
template<typename Vector3, typename Vector>
void print_out_of_plane_bends(
    const Vector& x_c,
    const std::vector<connectivity::OutOfPlaneBend>& bends,
    std::ostream& out = std::cout) {
  
  if (!bends.size()) return;

  const std::size_t n_bends{bends.size()};

  out << "*************************************\n";
  out << "  " << n_bends << " out-of-plane bends / \u00B0" << std::endl;
  out << "*************************************\n";

  // Atomic positions
  Vector3 pc{0., 0., 0.}, p1{0., 0., 0.}, p2{0., 0., 0.}, p3{0., 0., 0.};

  // Indices
  std::size_t idx_c{0}, idx_i{0}, idx_j{0}, idx_k{0};

  out.precision(2);
  out.fill(' ');

  for (std::size_t i{0}; i < n_bends; i++) {
    idx_c = bends[i].c;
    idx_i = bends[i].i;
    idx_j = bends[i].j;
    idx_k = bends[i].k;

    // Print indices
    out << std::left << "  (";
    out.width(4);
    out << std::right << idx_c << ',';
    out.width(4);
    out << std::right << idx_i << ',';
    out.width(4);
    out << std::right << idx_j << ',';
    out.width(4);
    out << std::right << idx_k << ')';

    // Get positions
    pc = {x_c(3 * idx_c), x_c(3 * idx_c + 1), x_c(3 * idx_c + 2)};
    p1 = {x_c(3 * idx_i), x_c(3 * idx_i + 1), x_c(3 * idx_i + 2)};
    p2 = {x_c(3 * idx_j), x_c(3 * idx_j + 1), x_c(3 * idx_j + 2)};
    p3 = {x_c(3 * idx_k), x_c(3 * idx_k + 1), x_c(3 * idx_k + 2)};

    // Print distance

    out << std::setw(9) << std::fixed << std::right
        << connectivity::out_of_plane_angle(pc, p1, p2, p3) *
               tools::conversion::rad_to_deg
        << std::endl;
  }
}

} // namespace io

} // namespace irc

#endif // IRC_IO_H
