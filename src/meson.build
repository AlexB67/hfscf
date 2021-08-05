conf_data = configuration_data()
conf_data.set('version', '0.7.1')
conf_data.set_quoted('PACKAGE_DATADIR', join_paths(get_option('prefix')))
conf_data.set_quoted('HFSCF_DIR', meson.source_root())
configure_file(
input:  '../hfscf_meson_config.h.in',
output: 'mesonconfig.h',
configuration: conf_data)

configuration_inc = include_directories('.')

hfscf_sources = [
  'hfscf_main.cpp',
  'hfscf/hfscf.cpp',
  'hfscf/hfscf_print.cpp',
  'hfscf/hfscf_numeric.cpp',
  'hfscf/hfscf_geomopt.cpp',
  'scfsolver/hfscf_hf_solver.cpp',
  'scfsolver/hfscf_rhf_symsolver.cpp',
  'scfsolver/hfscf_hf_hamiltonian.cpp',
  'scfsolver/hfscf_uhf_solver.cpp',
  'scfsolver/hfscf_uhf_symsolver.cpp',
  'scfsolver/hfscf_sad.cpp',
  'scfsolver/hfscf_so_rhf.cpp',
  'scfsolver/hfscf_so_uhf.cpp',
  'scfsolver/hfscf_uhf_print.cpp',
  'scfsolver/hfscf_rhf_print.cpp',
  'scfsolver/hfscf_diis.cpp',
  'gradient/hfscf_grad_ints.cpp',
  'gradient/hfscf_grad_misc.cpp',
  'gradient/hfscf_grad_rhf.cpp',
  'gradient/hfscf_grad_uhf.cpp',
  'gradient/hfscf_grad_rhf_mp2.cpp',
  'hessian/hfscf_hessian.cpp',
  'hessian/hfscf_hessian_resp.cpp',
  'hessian/hfscf_hessian_resp_mp2.cpp',
  'hessian/hfscf_hessian_ints.cpp',
  'hessian/hfscf_freq.cpp',
  'mol_properties/hfscf_properties.cpp',
  'settings/hfscf_settings.cpp',
  'postscf/hfscf_post_mp.cpp',
  'postscf/hfscf_post_cc.cpp',
  'postscf/hfscf_post_cc_rhf.cpp',
  'config_interaction/hfscf_cis.cpp',
  'integrals/hfscf_dipole.cpp',
  'integrals/hfscf_dipolederiv1.cpp',
  'integrals/hfscf_quadrupole.cpp',
  'integrals/hfscf_shell.cpp',
  'integrals/hfscf_shellpair.cpp',
  'integrals/hfscf_oseri.cpp',
  'integrals/hfscf_oseri_screen.cpp',
  'integrals/hfscf_oserideriv1.cpp',
  'integrals/hfscf_oserideriv2pq.cpp',
  'integrals/hfscf_oserideriv2qq.cpp',
  'integrals/hfscf_osnuclear.cpp',
  'integrals/hfscf_osnuclearderiv1.cpp',
  'integrals/hfscf_osnuclearderiv2.cpp',
  'integrals/hfscf_osoverlap.cpp',
  'integrals/hfscf_osoverlapderiv1.cpp',
  'integrals/hfscf_osoverlapderiv2.cpp',
  'integrals/hfscf_oskineticderiv1.cpp',
  'integrals/hfscf_oskineticderiv2.cpp',
  'integrals/hfscf_transform.cpp',
  'basis/hfscf_basis.cpp',
  'basis/hfscf_trans_basis.cpp',
  'molecule/hfscf_molecule.cpp',
  'symmetry/msym_helper.cpp',
  'molecule/hfscf_mol_print.cpp',
  'utils/hfscf_zmat.cpp',
  'irc/cart_to_int.cpp',
  'math/gamma.cpp',
  'math/solid_harmonics.cpp'
]

# add '-Wconversion' during development, it gives loads of (harmless) )warnings
# EIGEN_NO_DEBUG is to circumvent a bug eigen/meson issue not disabling asserts ... for whatever reason, 
# fine with cmake
# tmp fix for boost issue when using 1.75 BOOST_ALLOW_DEPRECATED_HEADERS
# should be fixed in 1.76

executable('hfscf', hfscf_sources,
dependencies: [hfscf_eigen_deps, hfscf_openmp_deps, hfscf_tclap_deps, 
               hfscf_boost_deps], include_directories : configuration_inc,
override_options : ['cpp_std=c++17'],
cpp_args : ['-static','-Wall', '-Wextra', '-Wpedantic', '-Wshadow',
            '-mfma', '-mavx', '-mavx2', '-msse4', '-fno-math-errno',
            '-DEIGEN_NO_DEBUG','-fopenmp','-DHAS_MESON_CONFIG', '-DBOOST_ALLOW_DEPRECATED_HEADERS'],
link_args : ['-march=native', '-msse4', '-lmsym'],
install: true,
)

