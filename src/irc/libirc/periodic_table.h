#ifndef IRC_PERIODIC_TABLE_H
#define IRC_PERIODIC_TABLE_H

#include "conversion.h"

#include <array>
#include <iostream>
#include <string>

namespace irc {

namespace periodic_table {

/// Number of atomic species in the periodic table
constexpr std::size_t pt_size{96};

// TODO: Get rid of C-style string
/// Atomic symbols
constexpr std::array<char[3], pt_size> symbols = {
    {"\0",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
     "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
     "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
     "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
     "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
     "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
     "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
     "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am"}};

/// Atomic masses
// 
/// Atomic masses
constexpr std::array<double, pt_size> masses = {{
    0.0000,   //   Ghost Atom      X       0
    1.00782503223,   //   Hydrogen        H       1
    4.00260325413,   //   Helium          He      2
    7.0160034366,    //   Lithium         Li      3
    9.012183065,   //   Beryllium       Be      4
    11.00930536,   //   Boron           B       5
    12.0000,  //   Carbon          C       6
    14.00307400443,
    15.99491461957, // O
    18.99840316273,
    19.9924401762,
    22.9897692820,
    23.985041697,
    26.98153853,
    27.97692653465,
    30.97376199842,
    31.9720711744,
    34.968852682,   // Cl
    39.9623831237,  // Ar
    38.9637064864,  // K
    39.962590863,   // Ca
    44.95590828,    // Sc
    47.94794198,    // Ti
    50.94395704,    // V
    51.94050623,    // Cr
    54.93804391,    // Mn
    55.93493633,    // Fe
    58.93319429,    // Co
    57.93534241,    // Ni
    62.92959772,    // Cu
    63.92914201,    // Zn
    68.9255735,     // Ga
    73.921177761,   // Ge
    74.92159457,    // As
    79.9165218,     // Se
    78.9183376,     // Br
    83.9114977282,  // Kr
    85.4678,  //   Rubidium        Rb      37 // not used anyway from here
    87.62,    //   Strontium       Sr      38 // but should be weight of most abundant, not average.
    88.9059,  //   Yttrium         Y       39
    91.224,   //   Zirconium       Zr      40
    92.9064,  //   Niobium         Nb      41
    95.94,    //   Molybdenum      Mo      42
    98.,      //   Technetium      Tc      43
    101.07,   //   Ruthenium       Ru      44
    102.9055, //   Rhodium         Rh      45
    106.42,   //   Palladium       Pd      46
    107.8682, //   Silver          Ag      47
    112.411,  //   Cadmium         Cd      48
    114.818,  //   Indium          In      49
    118.71,   //   Tin             Sn      50
    121.76,   //   Antimony        Sb      51
    127.6,    //   Tellurium       Te      52
    126.9045, //   Iodine          I       53
    131.293,  //   Xenon           Xe      54
    132.9055, //   Cesium          Cs      55
    137.327,  //   Barium          Ba      56
    138.9055, //   Lanthanum       La      57
    140.116,  //   Cerium          Ce      58
    140.9077, //   Praseodymium    Pr      59
    144.24,   //   Neodymium       Nd      60
    145,      //   Promethium      Pm      61
    150.36,   //   Samarium        Sm      62
    151.964,  //   Europium        Eu      63
    157.25,   //   Gadolinium      Gd      64
    158.9253, //   Terbium         Tb      65
    162.5,    //   Dysprosium      Dy      66
    164.9303, //   Holmium         Ho      67
    167.259,  //   Erbium          Er      68
    168.9342, //   Thulium         Tm      69
    173.04,   //   Ytterbium       Yb      70
    174.967,  //   Lutetium        Lu      71
    178.49,   //   Hafnium         Hf      72
    180.9479, //   Tantalum        Ta      73
    183.84,   //   Tungsten        W       74
    186.207,  //   Rhenium         Re      75
    190.23,   //   Osmium          Os      76
    192.217,  //   Iridium         Ir      77
    195.078,  //   Platinum        Pt      78
    196.9665, //   Gold            Au      79
    200.59,   //   Mercury         Hg      80
    204.3833, //   Thallium        Tl      81
    207.2,    //   Lead            Pb      82
    208.9804, //   Bismuth         Bi      83
    209.,     //   Polonium        Po      84
    210.,     //   Astatine        At      85
    222.,     //   Radon           Rn      86
    223.,     //   Francium        Fr      87
    226.,     //   Radium          Ra      88
    227.,     //   Actinium        Ac      89
    232.0381, //   Thorium         Th      90
    231.0359, //   Protactinium    Pa      91
    238.0289, //   Uranium         U       92
    237.,     //   Neptunium       Np      93
    244.,     //   Plutonium       Pu      94
    243.,     //   Americium       Am      95
}};

using tools::conversion::angstrom_to_bohr;

/// Covalent radii
///
/// J. C. Slater, "Atomic Radii in Crystals",
/// J. Phys. Chem. 41, 3199-3204 (1964)
///
/// American Elements (https://www.americanelements.com/, 30/05/2018)
// constexpr std::array<double, pt_size> covalent_radii = {{
//     0.00 * angstrom_to_bohr, //   Ghost Atom      X       0
//     0.31 * angstrom_to_bohr, // was 0.25 * angstrom_to_bohr, //   Hydrogen        H       1
//     0.00 * angstrom_to_bohr, //   Helium          He      2
//     1.45 * angstrom_to_bohr, //   Lithium         Li      3
//     1.05 * angstrom_to_bohr, //   Beryllium       Be      4
//     0.85 * angstrom_to_bohr, //   Boron           B       5
//     0.76 * angstrom_to_bohr, // was 0.70 * angstrom_to_bohr, //   Carbon          C       6
//     0.71 * angstrom_to_bohr, // was 0.65 * angstrom_to_bohr, //   Nitrogen        N       7
//     0.66 * angstrom_to_bohr, // was 0.60 * angstrom_to_bohr, //   Oxygen          O       8
//     0.50 * angstrom_to_bohr, //   Fluorine        F       9
//     0.00 * angstrom_to_bohr, //   Neon            Ne      10
//     1.80 * angstrom_to_bohr, //   Sodium          Na      11
//     1.50 * angstrom_to_bohr, //   Magnesium       Mg      12
//     1.25 * angstrom_to_bohr, //   Aluminum        Al      13
//     1.10 * angstrom_to_bohr, //   Silicon         Si      14
//     1.00 * angstrom_to_bohr, //   Phosphorus      P       15
//     1.00 * angstrom_to_bohr, //   Sulfur          S       16
//     1.00 * angstrom_to_bohr, //   Chlorine        Cl      17
//     0.00 * angstrom_to_bohr, //   Argon           Ar      18
//     2.20 * angstrom_to_bohr, //   Potassium       K       19
//     1.80 * angstrom_to_bohr, //   Calcium         Ca      20
//     1.60 * angstrom_to_bohr, //   Scandium        Sc      21
//     1.40 * angstrom_to_bohr, //   Titanium        Ti      22
//     1.35 * angstrom_to_bohr, //   Vanadium        V       23
//     1.40 * angstrom_to_bohr, //   Chromium        Cr      24
//     1.40 * angstrom_to_bohr, //   Manganese       Mn      25
//     1.40 * angstrom_to_bohr, //   Iron            Fe      26
//     1.35 * angstrom_to_bohr, //   Cobalt          Co      27
//     1.35 * angstrom_to_bohr, //   Nickel          Ni      28
//     1.35 * angstrom_to_bohr, //   Copper          Cu      29
//     1.35 * angstrom_to_bohr, //   Zinc            Zn      30
//     1.30 * angstrom_to_bohr, //   Gallium         Ga      31
//     1.25 * angstrom_to_bohr, //   Germanium       Ge      32
//     1.15 * angstrom_to_bohr, //   Arsenic         As      33
//     1.15 * angstrom_to_bohr, //   Selenium        Se      34
//     1.15 * angstrom_to_bohr, //   Bromine         Br      35
//     0.00 * angstrom_to_bohr, //   Krypton         Kr      36
//     2.35 * angstrom_to_bohr, //   Rubidium        Rb      37
//     2.00 * angstrom_to_bohr, //   Strontium       Sr      38
//     1.80 * angstrom_to_bohr, //   Yttrium         Y       39
//     1.55 * angstrom_to_bohr, //   Zirconium       Zr      40
//     1.45 * angstrom_to_bohr, //   Niobium         Nb      41
//     1.45 * angstrom_to_bohr, //   Molybdenum      Mo      42
//     1.35 * angstrom_to_bohr, //   Technetium      Tc      43
//     1.30 * angstrom_to_bohr, //   Ruthenium       Ru      44
//     1.35 * angstrom_to_bohr, //   Rhodium         Rh      45
//     1.40 * angstrom_to_bohr, //   Palladium       Pd      46
//     1.60 * angstrom_to_bohr, //   Silver          Ag      47
//     1.55 * angstrom_to_bohr, //   Cadmium         Cd      48
//     1.55 * angstrom_to_bohr, //   Indium          In      49
//     1.45 * angstrom_to_bohr, //   Tin             Sn      50
//     1.45 * angstrom_to_bohr, //   Antimony        Sb      51
//     1.40 * angstrom_to_bohr, //   Tellurium       Te      52
//     1.40 * angstrom_to_bohr, //   Iodine          I       53
//     0.00 * angstrom_to_bohr, //   Xenon           Xe      54
//     2.60 * angstrom_to_bohr, //   Cesium          Cs      55
//     2.15 * angstrom_to_bohr, //   Barium          Ba      56
//     1.95 * angstrom_to_bohr, //   Lanthanum       La      57
//     1.85 * angstrom_to_bohr, //   Cerium          Ce      58
//     1.85 * angstrom_to_bohr, //   Praseodymium    Pr      59
//     1.85 * angstrom_to_bohr, //   Neodymium       Nd      60
//     1.85 * angstrom_to_bohr, //   Promethium      Pm      61
//     1.85 * angstrom_to_bohr, //   Samarium        Sm      62
//     1.85 * angstrom_to_bohr, //   Europium        Eu      63
//     1.80 * angstrom_to_bohr, //   Gadolinium      Gd      64
//     1.75 * angstrom_to_bohr, //   Terbium         Tb      65
//     1.75 * angstrom_to_bohr, //   Dysprosium      Dy      66
//     1.75 * angstrom_to_bohr, //   Holmium         Ho      67
//     1.75 * angstrom_to_bohr, //   Erbium          Er      68
//     1.75 * angstrom_to_bohr, //   Thulium         Tm      69
//     1.75 * angstrom_to_bohr, //   Ytterbium       Yb      70
//     1.75 * angstrom_to_bohr, //   Lutetium        Lu      71
//     1.55 * angstrom_to_bohr, //   Hafnium         Hf      72
//     1.45 * angstrom_to_bohr, //   Tantalum        Ta      73
//     1.35 * angstrom_to_bohr, //   Tungsten        W       74
//     1.35 * angstrom_to_bohr, //   Rhenium         Re      75
//     1.30 * angstrom_to_bohr, //   Osmium          Os      76
//     1.35 * angstrom_to_bohr, //   Iridium         Ir      77
//     1.35 * angstrom_to_bohr, //   Platinum        Pt      78
//     1.35 * angstrom_to_bohr, //   Gold            Au      79
//     1.50 * angstrom_to_bohr, //   Mercury         Hg      80
//     1.90 * angstrom_to_bohr, //   Thallium        Tl      81
//     0.00 * angstrom_to_bohr, //   Lead            Pb      82
//     1.60 * angstrom_to_bohr, //   Bismuth         Bi      83
//     1.60 * angstrom_to_bohr, //   Polonium        Po      84
//     1.50 * angstrom_to_bohr, //   Astatine        At      85
//     0.00 * angstrom_to_bohr, //   Radon           Rn      86
//     2.42 * angstrom_to_bohr, //   Francium        Fr      87
//     2.15 * angstrom_to_bohr, //   Radium          Ra      88
//     1.95 * angstrom_to_bohr, //   Actinium        Ac      89
//     1.80 * angstrom_to_bohr, //   Thorium         Th      90
//     1.80 * angstrom_to_bohr, //   Protactinium    Pa      91
//     1.75 * angstrom_to_bohr, //   Uranium         U       92
//     1.75 * angstrom_to_bohr, //   Neptunium       Np      93
//     1.75 * angstrom_to_bohr, //   Plutonium       Pu      94
//     1.75 * angstrom_to_bohr, //   Americium       Am      95
// }};

// Use psi4 values
constexpr std::array<double, pt_size> covalent_radii = {{
0.0  * angstrom_to_bohr,  //X
0.31 * angstrom_to_bohr, //H
0.28 * angstrom_to_bohr, //He
1.28 * angstrom_to_bohr, //Li
0.96 * angstrom_to_bohr, //Be
0.84 * angstrom_to_bohr, //B
0.76 * angstrom_to_bohr, //C
0.71 * angstrom_to_bohr, //N
0.66 * angstrom_to_bohr, //O
0.57 * angstrom_to_bohr, //F
0.58 * angstrom_to_bohr, //Ne
1.66 * angstrom_to_bohr, //Na
1.41 * angstrom_to_bohr, //Mg
1.21 * angstrom_to_bohr, //Al
1.11 * angstrom_to_bohr, //Si
1.07 * angstrom_to_bohr, //P
1.05 * angstrom_to_bohr, //S
1.02 * angstrom_to_bohr, //Cl
1.06 * angstrom_to_bohr, //Ar
2.03 * angstrom_to_bohr, //K
1.76 * angstrom_to_bohr, //Ca
1.70 * angstrom_to_bohr, //Sc
1.60 * angstrom_to_bohr, //Ti
1.53 * angstrom_to_bohr, //V
1.39 * angstrom_to_bohr, //Cr
1.61 * angstrom_to_bohr, //Mn
1.52 * angstrom_to_bohr, //Fe
1.50 * angstrom_to_bohr, //Co
1.24 * angstrom_to_bohr, //Ni
1.32 * angstrom_to_bohr, //Cu
1.22 * angstrom_to_bohr, //Zn
1.22 * angstrom_to_bohr, //Ga
1.20 * angstrom_to_bohr, //Ge
1.19 * angstrom_to_bohr, //As
1.20 * angstrom_to_bohr, //Se
1.20 * angstrom_to_bohr, //Br
1.16 * angstrom_to_bohr, //Kr
2.20 * angstrom_to_bohr, //Rb
1.95 * angstrom_to_bohr, //Sr
1.90 * angstrom_to_bohr, //Y
1.75 * angstrom_to_bohr, //Zr
1.64 * angstrom_to_bohr, //Nb
1.54 * angstrom_to_bohr, //Mo
1.47 * angstrom_to_bohr, //Tc
1.46 * angstrom_to_bohr, //Ru
1.42 * angstrom_to_bohr, //Rh
1.39 * angstrom_to_bohr, //Pd
1.45 * angstrom_to_bohr, //Ag
1.44 * angstrom_to_bohr, //Cd
1.42 * angstrom_to_bohr, //In
1.39 * angstrom_to_bohr, //Sn
1.39 * angstrom_to_bohr, //Sb
1.38 * angstrom_to_bohr, //Te
1.39 * angstrom_to_bohr, //I
1.40 * angstrom_to_bohr, //Xe
2.44 * angstrom_to_bohr, //Cs
2.15 * angstrom_to_bohr, //Ba
2.07 * angstrom_to_bohr, //La
2.04 * angstrom_to_bohr, //Ce
2.03 * angstrom_to_bohr, //Pr
2.01 * angstrom_to_bohr, //Nd
1.99 * angstrom_to_bohr, //Pm
1.98 * angstrom_to_bohr, //Sm
1.98 * angstrom_to_bohr, //Eu
1.96 * angstrom_to_bohr, //Gd
1.94 * angstrom_to_bohr, //Tb
1.92 * angstrom_to_bohr, //Dy
1.92 * angstrom_to_bohr, //Ho
1.89 * angstrom_to_bohr, //Er
1.90 * angstrom_to_bohr, //Tm
1.87 * angstrom_to_bohr, //Yb
1.87 * angstrom_to_bohr, //Lu
1.75 * angstrom_to_bohr, //Hf
1.70 * angstrom_to_bohr, //Ta
1.62 * angstrom_to_bohr, //W
1.51 * angstrom_to_bohr, //Re
1.44 * angstrom_to_bohr, //Os
1.41 * angstrom_to_bohr, //Ir
1.36 * angstrom_to_bohr, //Pt
1.36 * angstrom_to_bohr, //Au
1.32 * angstrom_to_bohr, //Hg
1.45 * angstrom_to_bohr, //Tl
1.46 * angstrom_to_bohr, //Pb
1.48 * angstrom_to_bohr, //Bi
1.40 * angstrom_to_bohr, //Po
1.50 * angstrom_to_bohr, //At
1.50 * angstrom_to_bohr, //Rn
2.60 * angstrom_to_bohr, //Fr
2.21 * angstrom_to_bohr, //Ra
2.15 * angstrom_to_bohr, //Ac
2.06 * angstrom_to_bohr, //Th
2.00 * angstrom_to_bohr, //Pa
1.96 * angstrom_to_bohr, //U
1.90 * angstrom_to_bohr, //Np
1.87 * angstrom_to_bohr, //Pu
1.80 * angstrom_to_bohr, //Am
}};



/// Van der Waals radii
///
/// S. S. Batsanov, "Van der Waals Radii of Elements",
/// Inorganic Materials 37, 871-855 (2001)
///
/// American Elements (https://www.americanelements.com/, 30/05/2018)
constexpr std::array<double, pt_size> vdw_radii = {{
    0.00 * angstrom_to_bohr, //   Ghost Atom      X       0
    1.20 * angstrom_to_bohr, //   Hydrogen        H       1
    0.00 * angstrom_to_bohr, //   Helium          He      2
    2.24 * angstrom_to_bohr, //   Lithium         Li      3
    1.86 * angstrom_to_bohr, //   Beryllium       Be      4
    1.74 * angstrom_to_bohr, //   Boron           B       5
    1.70 * angstrom_to_bohr, //   Carbon          C       6
    1.55 * angstrom_to_bohr, //   Nitrogen        N       7
    1.52 * angstrom_to_bohr, //   Oxygen          O       8
    1.47 * angstrom_to_bohr, //   Fluorine        F       9
    0.00 * angstrom_to_bohr, //   Neon            Ne      10
    2.57 * angstrom_to_bohr, //   Sodium          Na      11
    2.27 * angstrom_to_bohr, //   Magnesium       Mg      12
    2.11 * angstrom_to_bohr, //   Aluminum        Al      13
    2.06 * angstrom_to_bohr, //   Silicon         Si      14
    1.80 * angstrom_to_bohr, //   Phosphorus      P       15
    1.80 * angstrom_to_bohr, //   Sulfur          S       16
    1.75 * angstrom_to_bohr, //   Chlorine        Cl      17
    0.00 * angstrom_to_bohr, //   Argon           Ar      18
    3.00 * angstrom_to_bohr, //   Potassium       K       19
    2.61 * angstrom_to_bohr, //   Calcium         Ca      20
    2.28 * angstrom_to_bohr, //   Scandium        Sc      21
    2.14 * angstrom_to_bohr, //   Titanium        Ti      22
    2.03 * angstrom_to_bohr, //   Vanadium        V       23
    1.97 * angstrom_to_bohr, //   Chromium        Cr      24
    1.96 * angstrom_to_bohr, //   Manganese       Mn      25
    1.96 * angstrom_to_bohr, //   Iron            Fe      26
    1.95 * angstrom_to_bohr, //   Cobalt          Co      27
    1.94 * angstrom_to_bohr, //   Nickel          Ni      28
    2.00 * angstrom_to_bohr, //   Copper          Cu      29
    2.02 * angstrom_to_bohr, //   Zinc            Zn      30
    2.08 * angstrom_to_bohr, //   Gallium         Ga      31
    2.13 * angstrom_to_bohr, //   Germanium       Ge      32
    2.16 * angstrom_to_bohr, //   Arsenic         As      33
    1.90 * angstrom_to_bohr, //   Selenium        Se      34
    1.85 * angstrom_to_bohr, //   Bromine         Br      35
    0.00 * angstrom_to_bohr, //   Krypton         Kr      36
    3.03 * angstrom_to_bohr, //   Rubidium        Rb      37
    2.78 * angstrom_to_bohr, //   Strontium       Sr      38
    2.45 * angstrom_to_bohr, //   Yttrium         Y       39
    2.25 * angstrom_to_bohr, //   Zirconium       Zr      40
    2.13 * angstrom_to_bohr, //   Niobium         Nb      41
    2.06 * angstrom_to_bohr, //   Molybdenum      Mo      42
    2.04 * angstrom_to_bohr, //   Technetium      Tc      43
    2.02 * angstrom_to_bohr, //   Ruthenium       Ru      44
    2.02 * angstrom_to_bohr, //   Rhodium         Rh      45
    2.05 * angstrom_to_bohr, //   Palladium       Pd      46
    2.13 * angstrom_to_bohr, //   Silver          Ag      47
    2.17 * angstrom_to_bohr, //   Cadmium         Cd      48
    2.24 * angstrom_to_bohr, //   Indium          In      49
    2.29 * angstrom_to_bohr, //   Tin             Sn      50
    2.33 * angstrom_to_bohr, //   Antimony        Sb      51
    2.06 * angstrom_to_bohr, //   Tellurium       Te      52
    1.98 * angstrom_to_bohr, //   Iodine          I       53
    0.00 * angstrom_to_bohr, //   Xenon           Xe      54
    3.43 * angstrom_to_bohr, //   Cesium          Cs      55
    2.85 * angstrom_to_bohr, //   Barium          Ba      56
    2.51 * angstrom_to_bohr, //   Lanthanum       La      57
    2.35 * angstrom_to_bohr, //   Cerium          Ce      58
    2.39 * angstrom_to_bohr, //   Praseodymium    Pr      59
    2.29 * angstrom_to_bohr, //   Neodymium       Nd      60
    2.36 * angstrom_to_bohr, //   Promethium      Pm      61
    2.29 * angstrom_to_bohr, //   Samarium        Sm      62
    2.33 * angstrom_to_bohr, //   Europium        Eu      63
    2.37 * angstrom_to_bohr, //   Gadolinium      Gd      64
    2.21 * angstrom_to_bohr, //   Terbium         Tb      65
    2.29 * angstrom_to_bohr, //   Dysprosium      Dy      66
    2.16 * angstrom_to_bohr, //   Holmium         Ho      67
    2.35 * angstrom_to_bohr, //   Erbium          Er      68
    2.27 * angstrom_to_bohr, //   Thulium         Tm      69
    2.42 * angstrom_to_bohr, //   Ytterbium       Yb      70
    2.21 * angstrom_to_bohr, //   Lutetium        Lu      71
    2.24 * angstrom_to_bohr, //   Hafnium         Hf      72
    2.00 * angstrom_to_bohr, //   Tantalum        Ta      73
    2.00 * angstrom_to_bohr, //   Tungsten        W       74
    2.00 * angstrom_to_bohr, //   Rhenium         Re      75
    2.03 * angstrom_to_bohr, //   Osmium          Os      76
    2.03 * angstrom_to_bohr, //   Iridium         Ir      77
    2.06 * angstrom_to_bohr, //   Platinum        Pt      78
    2.13 * angstrom_to_bohr, //   Gold            Au      79
    2.17 * angstrom_to_bohr, //   Mercury         Hg      80
    1.96 * angstrom_to_bohr, //   Thallium        Tl      81
    2.36 * angstrom_to_bohr, //   Lead            Pb      82
    2.07 * angstrom_to_bohr, //   Bismuth         Bi      83
    1.97 * angstrom_to_bohr, //   Polonium        Po      84
    2.02 * angstrom_to_bohr, //   Astatine        At      85
    0.00 * angstrom_to_bohr, //   Radon           Rn      86
    3.48 * angstrom_to_bohr, //   Francium        Fr      87
    2.83 * angstrom_to_bohr, //   Radium          Ra      88
    0.00 * angstrom_to_bohr, //   Actinium        Ac      89
    2.43 * angstrom_to_bohr, //   Thorium         Th      90
    2.43 * angstrom_to_bohr, //   Protactinium    Pa      91
    2.17 * angstrom_to_bohr, //   Uranium         U       92
    2.21 * angstrom_to_bohr, //   Neptunium       Np      93
    2.43 * angstrom_to_bohr, //   Plutonium       Pu      94
    2.44 * angstrom_to_bohr, //   Americium       Am      95
}};

/// Check if atomic number \param an is valid
///
/// \param an Atomic number
inline constexpr bool valid_atomic_number(std::size_t an) noexcept {
  return an > 0 && an < pt_size;
}

/// Return atomic number corresponding to symbol
///
/// \param symbol Atomic symbol
/// \return Atomic number
inline std::size_t atomic_number(const std::string& symbol) {
  std::size_t an{0};

  for (std::size_t i = 0; i < pt_size; i++) {
    if (symbol == symbols[i]) {
      an = i;
      break;
    }
  }

  if (an == 0) {
    throw std::logic_error("Invalid atomic symbol.");
  }

  return an;
}

} // namespace periodic_table

} // namespace irc

#endif // IRC_PERIODIC_TABLE_H
