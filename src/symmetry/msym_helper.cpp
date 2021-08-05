#include "msym_helper.hpp"
#include "../settings/hfscf_settings.hpp"
#include "../molecule/hfscf_elements.hpp"
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <memory>

using HF_SETTINGS::hf_settings;
// Adapted from the libmsym examples.
void MSYM::Msym::do_symmetry_analysis(EigenMatrix<double>& geom, const std::vector<int>& zval)
{
	if (m_natoms == 1)
	{
		m_point_group = "K";
        m_sub_group = "K";
		m_sigma = 1;
		return;
	}
	
	msym_error_t ret = MSYM_SUCCESS;
    msym_element_t *elements = nullptr;
	msym_element_t *melements = nullptr;
    const msym_character_table_t *ct = nullptr;
    const char *error = nullptr;
    char point_group[6];
	double symerr = 0.0;
    int mlength = 0;
	m_sigma = 0;
	int length = m_natoms;
	double cm[3];

	elements = (msym_element_t *)  malloc(m_natoms * sizeof(msym_element_t)); // old school
    memset(elements, 0, m_natoms * sizeof(msym_element_t));

	for (int i = 0; i < m_natoms; ++i)
    {
		std::string atom_name = ELEMENTDATA::atom_names[zval[i] - 1];
		atom_name.erase(std::remove(atom_name.begin(), atom_name.end(), ' '), atom_name.end());

		std::strcpy(elements[i].name, atom_name.c_str());
        elements[i].v[0] = geom(i, 0);
        elements[i].v[1] = geom(i, 1);
        elements[i].v[2] = geom(i, 2);
    }

	msym_context ctx = msymCreateContext();

	msym_thresholds_t *thresholds = (msym_thresholds_t *) malloc(sizeof(msym_thresholds_t));
	thresholds->geometry = 1E-03;
	thresholds->eigfact = 1E-03;
	thresholds->zero = 1E-03;
	thresholds->orthogonalization = 1E-02;
	thresholds->angle = 1E-03;
	thresholds->permutation = 5.0E-03;
	// mostly users shouldn't have to alter the default
	if (hf_settings::get_point_group_equivalence_threshold() == "DEFAULT") thresholds->equivalence = 5.0E-04;
	else if (hf_settings::get_point_group_equivalence_threshold() == "RELAXED") thresholds->equivalence = 5.0E-03;
	else thresholds->equivalence = 5.0E-05; // TIGHT

    /* thresholds */
	if(MSYM_SUCCESS != (ret = msymSetThresholds(ctx, thresholds))) goto err;
    /* Set elements */
    if(MSYM_SUCCESS != (ret = msymSetElements(ctx, length, elements))) goto err;
    /* Get elements msym elements */
    if(MSYM_SUCCESS != (ret = msymGetElements(ctx, &mlength, &melements))) goto err;
    
    free(elements);  elements = nullptr;

	if(MSYM_SUCCESS != (ret = msymGetCenterOfMass(ctx, cm))) goto err;
	if(MSYM_SUCCESS != (ret = msymFindSymmetry(ctx))) goto err;
    if(MSYM_SUCCESS != (ret = msymGetPointGroupName(ctx, sizeof(char[6]), point_group))) goto err;
	
    m_point_group = point_group;

	if(hf_settings::get_symmetrize_geom() || (hf_settings::get_geom_opt().length() && hf_settings::get_use_symmetry()))
		if(MSYM_SUCCESS != (ret = msymSymmetrizeElements(ctx, &symerr))) goto err;
	
	 // Determine the symmetry number sigma (rotational symmetry number) for thermo chemistry
	if(MSYM_SUCCESS != (ret = msymGetCharacterTable(ctx, &ct))) goto err;

	if(m_point_group != "C0v" && m_point_group != "D0h")
	{
		for(int i = 0; i < ct->d; ++i) // Get all the rotations to determine sigma, everything else removed.
			if (ct->sops[i]->type == _msym_symmetry_operation::MSYM_SYMMETRY_OPERATION_TYPE_IDENTITY ||
				ct->sops[i]->type == _msym_symmetry_operation::MSYM_SYMMETRY_OPERATION_TYPE_PROPER_ROTATION)
				//std::cout << ct->classc[i] << " " << ct->sops[i]->power << "  " << ct->sops[i]->order << '\n';
				m_sigma += ct->classc[i];
	}
	else if (m_point_group == "D0h") m_sigma = 2;
	else /* if (m_point_group == "C0v" || m_point_group == "") */ m_sigma = 1;
	
	if (hf_settings::get_align_geom())
		if(MSYM_SUCCESS != (ret = msymAlignAxes(ctx))) goto err;

	if (ret == MSYM_SUCCESS)
	{
		for (int i = 0; i < m_natoms; ++i)
		{ 
			geom(i, 0) = melements[i].v[0];
			geom(i, 1) = melements[i].v[1];
			geom(i, 2) = melements[i].v[2];
		}

		m_sym_deviation = symerr;
		isaligned = true;
	}

	msymReleaseContext(ctx);
    free(elements);
	return;

err:
    free(elements);
    error = msymErrorString(ret);

	if (hf_settings::get_point_group_equivalence_threshold() != "RELAXED")
	{
		std::cout << "\n  Error: " << error << ".\n         ";
		error = msymGetErrorDetails();
		std::cout << error << ".\n";
	}
	
	msymReleaseContext(ctx);

	if (hf_settings::get_point_group_equivalence_threshold() != "RELAXED")
	{
		std::cout << "         " << "Try the \"point_group_threshold = RELAXED\" setting.\n";
		exit(EXIT_FAILURE);
	}
	else // work around for libmsym returning error: no primary axis for reorientation
	{    // which means C1                      
		//std::cout << "  Error: Point group detection failed: Assuming C1 symmetry.\n\n";
		m_point_group = "C1";
        m_sub_group = "none";
        hf_settings::set_use_symmetry(false);
		m_sigma = 1;
	}
}

int MSYM::Msym::build_salcs(const Eigen::Ref<const EigenMatrix<double>>& geom, const std::vector<int>& zval,
                              const ShellVector& shells, const std::vector<int>& centers)
{
	if (m_natoms == 1)
	{
		m_point_group = "K";
        m_sub_group = "K";
		m_sigma = 1;
		return 0;
	}

    set_shell_data(shells, centers, zval);
	
	msym_error_t ret = MSYM_SUCCESS;
    msym_element_t *elements = nullptr;
	msym_element_t *melements = nullptr;
	//const msym_character_table_t *ct = nullptr;
	msym_basis_function_t *mbfs = nullptr;
    /* these are not mutable */
    const msym_symmetry_operation_t *msops = nullptr;
    const msym_subgroup_t *msg = nullptr;
    const msym_subrepresentation_space_t *msrs = nullptr;
    const msym_character_table_t *mct = nullptr;
    const msym_equivalence_set_t *mes = nullptr;
    double *irrep = nullptr;
    msym_basis_function_t *bfs = nullptr;
    int msopsl = 0, msrsl = 0, mbfsl = 0, mesl = 0;
	int bfsl = 0, msgl = 0, nbfsl = 0;
    const char *error = nullptr;
    char point_group[6];
	double symerr = 0.0;
    int mlength = 0;
	int length = m_natoms;
	double cm[3];
	double radius = 0.0;

	elements = (msym_element_t *)  malloc(m_natoms * sizeof(msym_element_t)); // old school
    memset(elements, 0, m_natoms * sizeof(msym_element_t));

	for (int i = 0; i < m_natoms; ++i)
    {
		std::string atom_name = ELEMENTDATA::atom_names[zval[i] - 1];
		atom_name.erase(std::remove(atom_name.begin(), atom_name.end(), ' '), atom_name.end());

		std::strcpy(elements[i].name, atom_name.c_str());
        elements[i].v[0] = geom(i, 0);
        elements[i].v[1] = geom(i, 1);
        elements[i].v[2] = geom(i, 2);
    }
	
	//double (*psalcs)[bfsl] = nullptr; // SALCs in matrix form
   	double *pcmem = nullptr; // Some temporary memory
   	int *pspecies = nullptr;
   	msym_partner_function_t *ppf = nullptr;

	msym_context ctx = msymCreateContext();

	msym_thresholds_t *thresholds = (msym_thresholds_t *) malloc(sizeof(msym_thresholds_t));
	thresholds->geometry = 1E-03;
	thresholds->eigfact = 1E-03;
	thresholds->zero = 1E-03;
	thresholds->orthogonalization = 1E-02;
	thresholds->angle = 1E-03;
	thresholds->permutation = 5.0E-03;
	// mostly users shouldn't have to alter the default
	if (hf_settings::get_point_group_equivalence_threshold() == "DEFAULT") thresholds->equivalence = 5.0E-04;
	else if (hf_settings::get_point_group_equivalence_threshold() == "RELAXED") thresholds->equivalence = 5.0E-03;
	else thresholds->equivalence = 5.0E-05; // TIGHT

    /* thresholds */
	if(MSYM_SUCCESS != (ret = msymSetThresholds(ctx, thresholds))) goto err;
    
	if(MSYM_SUCCESS != (ret = msymSetElements(ctx, length, elements))) goto err;
    
    /* Get elements msym elements */
    if(MSYM_SUCCESS != (ret = msymGetElements(ctx, &mlength, &melements))) goto err;
    
    free(elements);  elements = nullptr;

    /* Some trivial information */
    if(MSYM_SUCCESS != (ret = msymGetCenterOfMass(ctx, cm))) goto err;
    if(MSYM_SUCCESS != (ret = msymGetRadius(ctx, &radius))) goto err;
 
    /* Find molecular symmetry */
    if(MSYM_SUCCESS != (ret = msymFindSymmetry(ctx))) goto err;
    
    /* Get the point group name */
    if(MSYM_SUCCESS != (ret = msymGetPointGroupName(ctx, sizeof(char[6]), point_group))) goto err;
    if(MSYM_SUCCESS != (ret = msymGetSubgroups(ctx, &msgl, &msg))) goto err;

	if(hf_settings::get_symmetrize_geom())
		if(MSYM_SUCCESS != (ret = msymSymmetrizeElements(ctx, &symerr))) goto err;

	if(MSYM_SUCCESS != (ret = msymAlignAxes(ctx))) goto err;

    m_point_group = m_sub_group = std::string(point_group);

    if (hf_settings::get_point_group().length())
    {
        if (hf_settings::get_point_group() == m_point_group)
            goto done1;
        else
        {
            int ssg = 0;
            bool found = false;

            for (int l = 0; l < msgl; l++)
                if (0 == strcmp(msg[l].name, hf_settings::get_point_group().c_str())) 
                {
                    ssg = l;
                    found = true;
                    m_sub_group = msg[l].name;
                    goto done2;
                }

            done2:
            if (found == false) 
            {
                std::cout << "\n  Error: " << hf_settings::get_point_group()
                        << " is not a valid group or subgroup for the input molecule.\n";

                exit(EXIT_FAILURE);
            }
            else
            {
                if (ssg > 0)
                {
                    //printf("Selected point group %s\n", msg[ssg].name);
                    if (MSYM_SUCCESS != (ret = msymSelectSubgroup(ctx, &msg[ssg]))) goto err;
                    if (MSYM_SUCCESS != (ret = msymGetPointGroupName(ctx, sizeof(char[6]), point_group))) goto err;
                }
            }
        }
    }

    done1:
	//printf("Selected point group %s\n",point_group);
    //   if(MSYM_SUCCESS != (ret = msymSelectSubgroup(ctx, &msg[ssg]))) exit(1);
    //if(MSYM_SUCCESS != (ret = msymGetPointGroupName(ctx, sizeof(char[6]), point_group))) exit(1);

    if (MSYM_SUCCESS != (ret = msymGetEquivalenceSets(ctx, &mesl, &mes))) goto err;

    for (int i = 0; i < mesl; ++i) 
    {
        const std::string center = mes[i].elements[0]->name;

        for (int j = 0; j < (int)unique_atom_data.size(); ++j) 
        {
            if (unique_atom_data[j].atom_id != center) continue;

            const int ele_bfsl = 2 * unique_atom_data[j].l + 1;
            const msym_equivalence_set_t *smes = &mes[i]; // &mes[unique_atom_data[j].center];
            nbfsl = smes->length * ele_bfsl;
            bfs = (msym_basis_function_t *)realloc(bfs, bfsl * sizeof(*bfs) + nbfsl * sizeof(*bfs));
            memset(&bfs[bfsl], 0, nbfsl * sizeof(*bfs));

            for (int k = 0; k < smes->length; ++k)
                for (int m = -unique_atom_data[j].l; m <= unique_atom_data[j].l; ++m) 
                {
                    bfs[bfsl].element = smes->elements[k];
                    bfs[bfsl].type = _msym_basis_function::MSYM_BASIS_TYPE_REAL_SPHERICAL_HARMONIC;
                    bfs[bfsl].f.rsh.n = unique_atom_data[j].n;  // sel_n;
                    bfs[bfsl].f.rsh.l = unique_atom_data[j].l;  // sel_l;
                    bfs[bfsl].f.rsh.m = m;
                    bfsl++;
                }

            // printf(
            //      "Add %d real spherical harmonics basis functions with n=%d l=%d m=[%d,%d] to equivalence set %d\n",
            //       nbfsl, atom_data[j].n, atom_data[j].l, -atom_data[j].l, atom_data[j].l, atom_data[j].center + 1);
        }
    }

    //std::cout << "Added " << bfsl << " basis functions\n";
    if (MSYM_SUCCESS != (ret = msymSetBasisFunctions(ctx, bfsl, bfs))) goto err;

    msym_point_group_type_t mtype;
    int mn;
    if (MSYM_SUCCESS != (ret = msymGetPointGroupType(ctx, &mtype, &mn))) goto err;

    if ((MSYM_POINT_GROUP_TYPE_Dnh == mtype || MSYM_POINT_GROUP_TYPE_Cnv == mtype) && 0 == mn) 
    {
        if (MSYM_SUCCESS != (ret = msymGetSubgroups(ctx, &msgl, &msg))) goto err;

        int ssg = 0;
        if (hf_settings::get_point_group().length())
        {
            bool found = false;

            for (int l = 0; l < msgl; l++) 
                if (0 == strcmp(msg[l].name, hf_settings::get_point_group().c_str())) 
                {
                    ssg = l;
                    found = true;
                    m_sub_group = msg[l].name;
                    goto done;
                }
            
            if(!found)
            {
                std::cout << "\n  Error: " << hf_settings::get_point_group() 
                          << " is not a valid group or subgroup for the input molecule.\n";
                
                exit(EXIT_FAILURE);
            }
        }
        else if (0 == strcmp(point_group, "D0h"))
        {
            for (int l = 0; l < msgl; l++) 
                if (0 == strcmp(msg[l].name, "D2h")) 
                {
                    ssg = l; 
                    m_sub_group = "D2h";
                    goto done;
                }
                else if (0 == strcmp(msg[l].name, "D4h")) 
                {
                    ssg = l; 
                    m_sub_group = "D4h";
                    goto done;
                }
        }
        else if (0 == strcmp(point_group, "C0v"))
        {
            for (int l = 0; l < msgl; l++)
                if (0 == strcmp(msg[l].name, "C2v")) 
                {
                    ssg = l; 
                    m_sub_group = "C2v";
                    goto done;
                }
                else if (0 == strcmp(msg[l].name, "C4v")) 
                {
                    ssg = l; 
                    m_sub_group = "C4v";
                    goto done;
                }
        }

        done:
        if (ssg > 0) 
        {
            //printf("Selected point group %s\n", msg[ssg].name);
            if (MSYM_SUCCESS != (ret = msymSelectSubgroup(ctx, &msg[ssg]))) goto err;
            if (MSYM_SUCCESS != (ret = msymGetPointGroupName(ctx, sizeof(char[6]), point_group))) goto err;
        }

        if (MSYM_SUCCESS != (ret = msymGetEquivalenceSets(ctx, &mesl, &mes))) goto err;
    }

    /* Get elements msym elements */
    if(MSYM_SUCCESS != (ret = msymGetSymmetryOperations(ctx, &msopsl, &msops))) goto err;

    if(bfsl > 0)
	{
        // double (*salcs)[bfsl] = psalcs = (double (*)[bfsl]) calloc(bfsl, sizeof(*salcs)); // SALCs in matrix form
        // std::cout << sizeof(*salcs) << "  " << sizeof(double) << "\n";
	   	std::vector<std::vector<double>> psalcs = std::vector(bfsl, std::vector<double>(sizeof(double *)));
        double *cmem = pcmem = (double *) calloc(bfsl, sizeof(*cmem)); // Some temporary memory
        int *species = pspecies = (int *) calloc(bfsl, sizeof(*species));
        msym_partner_function_t *pf = ppf = (msym_partner_function_t *) calloc(bfsl, sizeof(*pf));
        
        if(MSYM_SUCCESS != (ret = msymGetBasisFunctions(ctx, &mbfsl, &mbfs))) goto err;
        if(MSYM_SUCCESS != (ret = msymGetCharacterTable(ctx, &mct))) goto err;
        if(MSYM_SUCCESS != (ret = msymGetSubrepresentationSpaces(ctx, &msrsl, &msrs))) goto err;
        
        irrep = (double *) calloc(mct->d, sizeof(*irrep));
        
        //printf("\nGenerated SALCs from %d basis functions of %d symmetry species.\n\n", mbfsl, mct->d);

        for (int j = 0; j < msrsl; ++j)
        {
            int salcl = msrs[j].salcl;
            
            if(salcl == 0) continue;

            //printf("\n %s %d\n", mct->s[msrs[j].s].name, msrs[j].salcl);
           // sym_species.emplace_back(mct->s[msrs[j].s].name);

            for (int d = 0; d < mct->s[msrs[j].s].d; ++d)
            {
                sym_species.emplace_back(mct->s[msrs[j].s].name);
                irrep_size.emplace_back(msrs[j].salcl);
            }
            
            std::vector<EigenMatrix<double>> symblk; 
            symblk = std::vector<EigenMatrix<double>>(mct->s[msrs[j].s].d);

            for (size_t d = 0; d < symblk.size(); ++d)
                symblk[d] = EigenMatrix<double>::Zero(bfsl, salcl);

            for (int i = 0; i < salcl; ++i)
            {
                std::string type;
                const msym_salc_t& salc = msrs[j].salc[i];
                msym_basis_function_t *bf = salc.f[0];
                const msym_equivalence_set_t *salces = nullptr;
                
                if (MSYM_SUCCESS != (ret = msymGetEquivalenceSetByElement(ctx, bf->element, &salces)))
                    goto err;
                    
                if (bf->type == _msym_basis_function::MSYM_BASIS_TYPE_REAL_SPHERICAL_HARMONIC)
                    type = "real spherical harmonic ";

                //printSALC(salc, melements);
                build_symmetry_blocks(symblk, salc, melements, i);
            }

            for (size_t d = 0; d < symblk.size(); ++d)
                sblocks.emplace_back(symblk[d]);
        }
    }
    
    msymReleaseContext(ctx);
    
    //free(psalcs);
    free(pcmem);
    free(pspecies);
    free(ppf);
    free(bfs);
    free(elements);
    free(irrep);
    return ret;
err:
   // free(psalcs);
    free(pcmem);
    free(pspecies);
    free(ppf);
    free(bfs);
    free(elements);
    free(irrep);
    error = msymErrorString(ret);
    fprintf(stderr,"Error %s: ",error);
    error = msymGetErrorDetails();
    fprintf(stderr,"%s\n",error);
    msymReleaseContext(ctx);
    return ret;
}

void MSYM::printSALC(const msym_salc_t& salc, const msym_element_t * melements)
{
    // do away with VLAs from example code
    //double(*space)[salc.fl] = (double(*)[salc.fl])salc.pf;
    double *sp = (double(*))salc.pf;

    for (int d = 0; d < salc.d; d++) 
    {
        if (salc.d > 1) printf("Component %d:\n", d + 1);
        for (int line = 0; line < salc.fl; line += 6) 
        {
            for (int i = line; i < line + 6 && i < salc.fl; ++i)
            {
                msym_basis_function_t *bf = salc.f[i];
                printf(" %d%s %-8s\t", (int)(bf->element - melements) + 1, bf->element->name, bf->name);
            }

            printf("\n");

            for (int i = line; i < line + 6 && i < salc.fl; ++i)
            {
                 printf("%10.7lf\t", sp[salc.fl * d + i]);// printf("%10.7lf\t", sp[d][i]);
            }
            
            printf("\n");
        }
    }
}

void MSYM::Msym::build_symmetry_blocks(std::vector<EigenMatrix<double>>& symblock, 
                                       const msym_salc_t& salc, const msym_element_t * melements, int bfsl)
{
    // do away with VLAs from example code
    //double(*space)[salc.fl] = (double(*)[salc.fl])salc.pf;
    double *sp = (double(*))salc.pf;

    for(int d = 0; d < salc.d; ++d)
    {
        //if(salc.d > 1) printf("Component %d:\n",d + 1);
        for(int j = 0; j < salc.fl; ++j)
        {
            msym_basis_function_t *bf = salc.f[j];
            std::string at = std::to_string((int)(bf->element - melements) + 1) + bf->element->name
                        + std::to_string(bf->f.rsh.n) + std::to_string(bf->f.rsh.l)
                        + std::to_string(bf->f.rsh.m);

            int offset = bfs_to_ids[at];
            symblock[d](offset, bfsl) = sp[salc.fl * d + j]; //replace vla sp[d][j]
        }
    }
}

void MSYM::Msym::set_shell_data(const ShellVector& shells, const std::vector<int>& centers, 
                                const std::vector<int>& zval)
{
    int n = 1;
    for (size_t i = 0; i < shells.size(); ++i) 
    {
        if (i > 0)
        {
            if (shells[i - 1].L() == shells[i].L() && centers[i - 1] == centers[i]) ++n;
            else if (centers[i - 1] == centers[i] && shells[i].L() == 0) ++n;
        }

        int l = static_cast<int>(shells[i].L());

        std::string at{ELEMENTDATA::atom_names[zval[centers[i]] - 1]};
        at.erase(std::remove(at.begin(), at.end(), ' '), at.end());

        atom_data.emplace_back(atomdata(at, centers[i], n, l, shells[i].get_ids()));

       if (i > 0 && i < centers.size() - 1)
           if (centers[i] != centers[i + 1]) n = 1;
    }


    for (const auto &t : atom_data) 
    {
        bool same_center = true;
        for (const auto& s : unique_atom_data)
            if (s.atom_id == t.atom_id && s.center != t.center) same_center = false;
        
        if(same_center)
            unique_atom_data.emplace_back(t);
    }

    for (const auto& t : atom_data)
    {
        std::string bfs_id = std::to_string(t.center + 1) + t.atom_id + std::to_string(t.n) 
                           + std::to_string(t.l);

        int offset = t.idx;

        for(int m = -t.l; m <= t.l; ++m)
        {
            std::string bfs_idm = bfs_id + std::to_string(m);
            bfs_to_ids[bfs_idm] = offset;
            ++offset;
        }

    }
}
