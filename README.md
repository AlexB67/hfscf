## HFSCF - Perform UHF/RHF calculations and beyond on small polyatomic molecules

Inpired by the crawdad exercises, hfcxx, 
McMurchie-Davidson, and a 30 year hiatus from physical chemistry and spectroscopy, a small *ab initio* package was born, written in C++. The original [Crawdad exercises](https://github.com/CrawfordGroup/ProgrammingProjects) hfcxx and McMurchie-Davidson may be found [here](https://github.com/ifilot/hfcxx) and [here](https://github.com/jjgoings/McMurchie-Davidson). With the exception of the DL algorithm you will find solutions to all the Crawdad exercises, and more.

### Supported features
* RHF.
* UHF.
* SOSCF.
* DIIS.
* SAD and CORE initial guess methods.
* MP2, MP3 RHF/UHF.
* CIS excitation energies and oscillator strengths.
* RPA energies
* CCSD CCSD(T) in spin and closed shell basis (RHF)
* Analytic gradients, SCF and MP2 (RHF).
* Analytic gradients, SCF (UHF).
* Numeric gradients, MP2 (UHF)
* Conjugate gradient or RFO geometry optimisation at the SCF and MP2 level in redundant internal coordinates using the IRC library.
* Shell based Cartesian Gaussian integrals up to 2nd derivative (Obara-Saika).
* Support for pure angular momentum and cartesian basis sets in Psi4 compatible format.
* Symmetry and SALC determination (RHF and UHF only) using libmsym.
* Analytic Hessian SCF and MP2 (RHF).
* Numeric Hessian SCF and MP2 (UHF).
* Vibrational Harmonic frequencies.
* IR Intensities.
* Normal modes.
* Thermochemistry analysis.
* Lowdin and Mulliken population analysis.
* Dipole moments, Quadrupole moments and static dipole polarizabilities (using direct or iterative CPHF).

### Building the software

<p>To build the software you can use either cmake or meson running on any regular Linux distributions, or the WSL subsystem for Windows.</p> 

<p>The following packages must be installed (if not present already).</p>

* Optional: git (if you you want to clone the repository)
* The make or ninja build system.
* gcc++ 9 or later. (versions must support the C++17 standard)
* Optional: OpenMP support, but highly recommended for increased performance; gcc/g++ has OpenMP support out of the box.
* meson or cmake
* Library; Eigen3 (3.3.6 or later).
* pkgconfig. required to detect tclap.
* Library: tclap.
* Library: C++ boost libraries (1.58 or later).
* libmsym: if using meson only. A cmake build includes it.

Note: while clang++ works it is unsupported at this time.

## Download the software
Either download as a zip file [from here](https://github.com/AlexB67/hfscf.git), 
or use git and issue the comand

```
git clone https://github.com/AlexB67/hfscf.git
```
## Building with cmake

<p>Assuming the project is at the top level home directory and your current path is the home directory, it will look like 

```
/home/*your_user_name*/hfscf
```

run the following commands</p>

```
cd hfscf
mkdir cm_build/
cd cm_build/
ccmake .. # or you can use cmake-gui
```

<p>Change the Release build flags for C and C++ to O3, configure and generate.</p>

```
cmake ..
make -j # or whatever build system you have elected to use, for example, for ninja it would be "ninja".
```

## Building with meson
NB: If you have already build the software with cmake you may skip this step.

Install libmsym as a static library. Generally it is not available in distribution repositories, with the exception of AUR Arch Linux  (which may well be outdated). 0.2.4 is required. It can be obtained from [here](https://github.com/mcodev31/libmsym). Follow the instructions within.

<p>Assuming the project is at the top level home directory and your current path is the home directory, it will look like 

```
/home/*your_user_name*/hfscf
```

run the following commands</p>

```
cd hfscf
mkdir m_releasebuild/
meson m_releasebuild/ --buildtype=release
ninja -C m_releasebuild/
```
<p>Note: Do not mix meson and cmake build directories. I have given the build directories unique names as described above; You can use whatever build directory names you like.</p>

## Using the software
<p>For more information how to use the software, see the examples directory or consult the manual. You can run the program from the build directory. Installation is not necessary and unsupported.</p>

```
cd *to_build_directory*
src/hfscf -i ../examples/inputfile
```
<p>For example</p>

```
src/hfscf -i ../examples/h2o_1.dat
```

or for more verbose output

```
src/hfscf -i ../examples/h2o_1.dat -v 3
```
or should you wish to write to an ouput file, you can simply redirect the ouput as follows
```
src/hfscf -i ../examples/h2o_1.dat -v 3 > /path_to_output_file/h2o_1.out
```
For aditional arguments type

```
src/hfscf --help
```
## Preparing an input file

<p>Input files are in text format. The first line is comment line. This is mandatory. Any other comment lines are optional and start with the  # symbol in column 1. Keywords are always in the form 

```
keyword = value
```
and often case sensitive. For a list of keywords see the source file

```
hfscf/src/molecule/hfscf_keywords.hpp
```
Coordinates are placed in curly braces, no keywords can be used in this section. Both Cartesian and Z matrix formats are supported, but may not be mixed. Variable names are not allowed for bond lengths, angles and dihedrals. 

**Important:** The default unit for length is in atomic units, if not specified, contrary to most packages using Angstrom as the default length unit.

A typical input Z matrix input file will look like

```
# ethene SCF frequencies 
scf_type = RHF
basis_set = 6-31+G**
gradient_type = SCF
eri_screen = false
symmetrize_geom = true
geom_opt = SCF 
geom_opt_tol = HIGH
frequencies = SCF
units = angstrom
{
	C 
	C    1  1.3358
	H    1  1.0855   2 121.051
	H    1  1.0855   2 121.051   3 180.000
	H    2  1.0855   1 121.051   3   0.000
	H    2  1.0855   1 121.051   3 180.000
}


```
dummy atoms are denoted by the symbol Z. A Cartesian input will look like

```
# Benzene HF/6-311G no dummy atoms

initial_guess = SAD

# DIIS extrapolation steps
diis_range = 8

# Basis set
basis_set = 6-311G

# turn on symmetry
use_symmetry = true

# force point group
point_group_override = D2h

# Units used. Cartesian input allows atomic numbers, or atomic symbols
# A Z matrix allows atomic symbols only.

units = angstrom
{
	6  0.000  1.396  0.000
	6  1.209  0.698  0.000
	6  1.209 -0.698  0.000
	6  0.000 -1.396  0.000
	6 -1.209 -0.698  0.000
	6 -1.209  0.698  0.000
	1  0.000  2.479  0.000
	1  2.147  1.240  0.000
	1  2.147 -1.240  0.000
	1  0.000 -2.479  0.000
	1 -2.147 -1.240  0.000
	1 -2.147  1.240  0.000
}

```
Keywords often assume a default value if not entered. If no basis set is specified, STO-3G will be used. 

The hfscf program will validate your input file. In most cases an error message will be displayed followed by a graceful exit if an error is detected, with a suggestion how to fix it.
</p>

## Basis sets
Basis sets are psi4 format compatible only, a number are already included. New basis set can be obtained from [the basis set exchange website](https://www.basissetexchange.org/). hfscf maximum supported angular momentum is L = 5 (H). Density fitted basis sets are not supported. 

Make sure that the Psi4 format option is selected when downloading. You can place a new basis in the basis set directory found at

```
hfscf/basis
```
