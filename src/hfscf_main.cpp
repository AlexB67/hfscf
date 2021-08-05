#include "settings/hfscf_settings.hpp"
#include "hfscf_main.hpp"
#include <iostream>
#include <Eigen/Core>
#include <filesystem>
#include <tclap/CmdLine.h>
#include <ctime>
#include <chrono>
#include <iomanip>
//#include </usr/local/intel/mkl/include/mkl.h>
//#include <libcpuid/libcpuid.h>
#ifdef HAS_MESON_CONFIG // meson support
#include <mesonconfig.h>
#elif  HAS_CMAKE_CONFIG // cmake support
#include <cmakeconfig.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif


Mol::hfscf::hfscf(const bool verbose, const int threads)
: m_verbose(verbose)
{
    int maxthreads = Eigen::nbThreads();
    //mkl_set_num_threads(maxthreads);

    if (threads > maxthreads)
    {
        std::cout << "  Warning: supplied threads greater than max threads. \
                        setting maximum possible threads to " << maxthreads << '\n';
        Eigen::setNbThreads(maxthreads);
    }
    else if (threads == 0) 
    {
        Eigen::setNbThreads(maxthreads);
    }
    else
    {
        #ifdef _OPENMP
        omp_set_num_threads(threads);
        #endif
        Eigen::setNbThreads(threads);
    }
}

void::Mol::hfscf::run(const std::string& geometry_file)
{

    if (false == std::filesystem::exists(geometry_file)) 
	{
		std::cout << "\n\n  Error: Input file not found at path:  " << geometry_file << '\n';
		exit(EXIT_FAILURE);
	}

    std::cout << "  ************************************************************************";
    std::cout << "\n  *   hfscf: A Small Ab Initio Electronic Structure Software package.    *";
    std::cout << "\n  *                                                                      *";
    std::cout << "\n  *                     Author:  Alexander Borro.                        *";
    std::cout << "\n  *                                                                      *";
    std::cout << "\n  *                     Third party contibutions:                        *";
    std::cout << "\n  *                                                                      *";
    std::cout << "\n  *              Rocco Meli: IRC - Internal coordinates.                 *";
    std::cout << "\n  *        Marcus Johansson: libmsym - Molecular Point groups.           *";
    std::cout << "\n  *                                                                      *";
    std::cout << "\n  *          Driven by Eigen3: A C++ linear algebra package.             *";
    std::cout << "\n  ************************************************************************\n\n";
    // struct cpu_raw_data_t raw; 
    // struct cpu_id_t data; s
    
    // cpuid_get_raw_data(&raw);
    // cpu_identify(&raw, &data);

    // auto cores = data.num_cores;
    // auto thrds = data.num_logical_cpus;
    // auto *model = data.vendor_str;
    // auto brand = data.brand_str;
    // auto code = data.cpu_codename;

    // std::cout << "**************************************************************************\n";
    // std::cout << "  Hardware information\n";
    // std::cout << "**************************************************************************\n";
    // std::cout << "  CPU Model                  = " << brand << "\n";
    // std::cout << "  CPU codename               = " << code  << "\n";
    // std::cout << "  CPU vendor id              = " << model << "\n";
    // std::cout << "  Available physical cores   = " << cores << '\n';
    // std::cout << "  Available logical  cores   = " << thrds << '\n';
    std::cout << "**************************************************************************\n";
    std::cout << "  Build information\n";
    std::cout << "**************************************************************************\n";

    std::string top_level_dir;
    std::string compiler_id;
    #if (HAS_CMAKE_CONFIG) && defined(MSVC) // For windows
    compiler_id = HFSCF_COMPILER;
    #elif defined(HAS_CMAKE_CONFIG)
    compiler_id = __VERSION__;
    #endif
    
    #ifdef HAS_MESON_CONFIG // Meson for linux
    top_level_dir = HFSCF_DIR;
    const std::string version =  VERSION_STR;
    std::cout << "  C++ Compiler Id            = " << __VERSION__ << '\n';
    std::cout << "  Build system               = meson\n";
    std::cout << "  Program version            = " << version << '\n';
    std::cout << "  Project root               = " << top_level_dir << '\n';
    #elif HAS_CMAKE_CONFIG
    top_level_dir = HFSCF_DIR;
    const std::string version =  VERSION;
    std::cout << "  C++ Compiler Id            = " << compiler_id << '\n';
    std::cout << "  Build system               = cmake\n";
    std::cout << "  Program version            = " << version << '\n';
    std::cout << "  Project root               = " << top_level_dir << '\n';
    #endif
    
    std::cout << "  libmsym version            = " << "0.2.4" << '\n'; // TODO get it from config

    #ifdef _OPENMP
    std::cout << "  OPENMP support             = " << "Yes" << '\n';
    #else
    std::cout << "  OPENMP support             = " << "No" << '\n';
    #endif

    hf_ptr = std::make_unique<Mol::scf>(m_verbose, geometry_file, top_level_dir);
    hf_ptr->hf_run();
}

int main(int argc, char **argv)
{
    std::clock_t start_time = std::clock();
    auto w_start = std::chrono::high_resolution_clock::now();

    try 
    {
        #ifdef HAS_MESON_CONFIG
        std::string version = VERSION_STR;
        #elif HAS_CMAKE_CONFIG
        std::string version = PROGRAM_VERSION;
        #else
        std::string version = "unkown"; // if for some reason we fall through .. shouldn't happen
        #endif
      

        TCLAP::CmdLine cmd("hfscf performs calculations at the Hartree Fock SCF level and beyond "\
                           "for a given molecular geometry input file.", ' ', version);

        TCLAP::ValueArg<std::string> arg_inputfile("i", "input", "Input geometry file.", true, "filename", "filename");
        cmd.add(arg_inputfile);

        TCLAP::ValueArg<std::string> arg_basisset("p", "path", "Override the default basis set directory.", 
                                                  false, "", "directory");
        cmd.add(arg_basisset);

        TCLAP::ValueArg<int> arg_threads("n", "nthreads", "The number of CPU threads to use. ", false, 0, "int");
        cmd.add(arg_threads);

        TCLAP::ValueArg<short> arg_verbose("v", "verbose", "Verbosity level 1 to 5.", false, 1, "int");
        cmd.add(arg_verbose);

        cmd.parse(argc, argv);

        std::string geom_path = arg_inputfile.getValue();
        std::string basisset_path = arg_basisset.getValue();
        int verbose = arg_verbose.getValue();
        int threads = arg_threads.getValue();

        if(verbose < 1 || verbose > 5)
        {
            std::cout << "  Error: Invalid verbosity level. Supported levels are 1 to 5.\n\n";
            exit(EXIT_FAILURE);
        }

        if(basisset_path.length()) HF_SETTINGS::hf_settings::set_basis_set_path(basisset_path);
        HF_SETTINGS::hf_settings::set_verbosity(static_cast<short>(verbose));
        std::unique_ptr<Mol::hfscf> hf = std::make_unique<Mol::hfscf>(verbose, threads);

        hf->run(geom_path);
        
    } 
    catch (TCLAP::ArgException &e) 
    {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << '\n';
        return -1;
    }

    std::clock_t end_time = std::clock();
    double milli_sec = 1000.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    double cpu_sec = milli_sec / 1000.0;
    long cpu_min = std::floor(cpu_sec / 60);
    long cpu_hr = std::floor(cpu_sec / 3600);

    auto w_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = w_end - w_start;
    auto w_sec = duration.count() / 1000;
    long w_min = std::floor(w_sec / 60);
    long w_hr = std::floor(w_sec / 3600);

    std::cout << "\n  CPU usage: " << std::setprecision(0) << 100 * cpu_sec / w_sec << "%CPU\n"; 
    std::cout << "  CPU  time: " << cpu_hr  << "h" << cpu_min  - 60 * cpu_hr << "m"
              << std::fixed << std::setprecision(3) << cpu_sec - 60.0 * cpu_min << "s\n";
    std::cout << "  Wall time: " << w_hr  << "h" << w_min - 60 * w_hr << "m"
              << std::setprecision(3) << w_sec - 60.0 * w_min << "s\n\n";
    std::cout << "  hfscf execution end.\n\n";
    return 0;
}

