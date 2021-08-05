#ifndef ELEMENT_DATA_H
#define ELEMENT_DATA_H

#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <array>
#include <tuple>
// TODO isotope support
namespace ELEMENTDATA
{
// equivalent to
// const std::map <std::string , unsigned int> in STL
// instead for fixed size containers https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html

	static const std::map<std::string, int, std::less<std::string>, 
  	Eigen::aligned_allocator<std::pair<const std::string, int> > > name_to_Z
  	{
		{"H",   1},
		{"He",  2},
		{"Li",  3},
		{"Be",  4},
		{"B",   5},
		{"C",   6},
		{"N",   7},
		{"O",   8},
		{"F",   9},  
		{"Ne", 10},
		{"Na", 11},
		{"Mg", 12},
		{"Al", 13},
		{"Si", 14},
		{"P",  15},
		{"S",  16},
		{"Cl", 17},
		{"Ar", 18},
		{"K",  19},
		{"Ca", 20},
		{"Sc", 21},
		{"Ti", 22},
		{"V",  23},
		{"Cr", 24},
		{"Mn", 25},
		{"Fe", 26},
		{"Co", 27},
		{"Ni", 28},
		{"Cu", 29},
		{"Zn", 30},
		{"Ga", 31},
		{"Ge", 32},
		{"As", 33},
		{"Se", 34},
		{"Br", 35},
		{"Kr", 36}
  	};
  	
  	static constexpr std::array<double, 36> masses =
    {
        1.00782503223,
        4.00260325413,
        7.0160034366,
        9.012183065,
        11.00930536,
        12.0, // C
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
    };
    
    inline static const std::array<std::string, 36> atom_names =
    {
        // store elements with right justification
        " H",
        "He",
        "Li",
        "Be",
        " B",
        " C",
        " N",
        " O",
        " F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        " P",
        " S",
        "Cl",
        "Ar",
        " K",
        "Ca",
        "Sc",
        "Ti",
        " V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr"
    };
}

#endif
// ELEMENT_DATA_H

