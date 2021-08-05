#ifndef MOL_HFSCF
#define MOL_HFSCF

#include "hfscf/hfscf.hpp"

namespace Mol
{
    class hfscf
    {
        public:
            explicit hfscf(bool verbose, const int threads);
            hfscf(const hfscf&) = delete;
            hfscf& operator=(const hfscf& other) = delete;
            hfscf(const hfscf&&) = delete;
            hfscf&& operator=(const hfscf&& other) = delete;
            void run(const std::string& geometry_file);


         private:
            bool m_verbose;
            std::unique_ptr<Mol::scf> hf_ptr;
    };
}

#endif
// end MOL_HFSCF