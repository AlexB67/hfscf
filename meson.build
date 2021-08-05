project('hfscf', ['cpp', 'c'], 
        version: '0.7.1',
  meson_version: '>= 0.53.0',
  default_options : ['buildtype=release', 'warning_level=3']
)

cmake = import('cmake')
pkg = import('pkgconfig')

hfscf_eigen_deps    = dependency('eigen3',  required : true, version: '>= 3.3')
hfscf_openmp_deps   = dependency('openmp',  required : false)
hfscf_tclap_deps    = dependency('tclap',   required : true)
hfscf_boost_deps    = dependency('boost',   required : true, version: '>= 1.58')
hfscf_libmsym_deps  = dependency('libmsym', method : 'cmake', static : true, required : true)

subdir('src')
