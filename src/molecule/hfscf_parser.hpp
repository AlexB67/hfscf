#ifndef PARSE_LINE
#define PARSE_LINE

#include "hfscf_elements.hpp"
#include "hfscf_keywords.hpp"
#include <istream>
#include <sstream>
#include <iostream>
#include <string>
#include <optional>

namespace MOLPARSE
{

void inline remove_white_space(std::string& str)
{
	str.erase(std::remove(str.begin(), str.end(), '\t'), str.end());
	str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
}

std::string inline get_keyword(const std::string& line)
{
	std::string tmp;
	std::stringstream tokens(line);
	std::getline(tokens, tmp, '=');

	remove_white_space(tmp);
	return tmp;
}

bool inline check_value_is_double(const std::string& val)
{
	if (val.empty()) return false;
	else if(val.find_first_not_of("+-0123456789.Ee") != std::string::npos) return false;

	// we can fall through as the above is not robust enough
	std::stringstream is_bad_dbl(val);
	double isdbl;

	is_bad_dbl >> isdbl;

	if (is_bad_dbl.good()) return false;

	return true;
}

std::optional<std::string> inline parse_value_from_keyword(const std::string& line)
{
    std::string tmp, keyword;
	std::stringstream tokens(line);
	
	std::getline(tokens, tmp, '=');
	remove_white_space(tmp);

	const auto find_key = Keywords::lookup_keyword.find(tmp);

	if(find_key == Keywords::lookup_keyword.end())
	{
		std::cout << "\n\n  Error: Unknown keyword. Aborting.\n";
		std::cout << "  Line: \"" << line << "\"\n\n";
		exit(EXIT_FAILURE);
	}
	else
		keyword = tmp;
	
	std::getline(tokens, tmp, '=');
	remove_white_space(tmp);

	if(find_key->second == "*") // exceptions which require no further validation
	{
		std::invoke(Keywords::set_setting_str.at(get_keyword(line)), tmp);
		return std::nullopt;
	}
	else if(find_key->second == "int") // integer types
	{
		if (tmp.find_first_not_of("+-0123456789")  != std::string::npos &&
		    tmp.substr(1, tmp.length()).find_first_not_of("0123456789")  != std::string::npos)
		{
			std::cout << "\n\n" << " Error: Invalid keyword specification," 
					  << Keywords::err_mesg.at(keyword) << "\n";
			std::cout << "  Line: \"" << line << "\"\n\n";
			exit(EXIT_FAILURE);
		}
		else if(keyword != "multiplicity" && keyword != "charge") // temp fix
			std::invoke(Keywords::set_setting<int>.at(get_keyword(line)), std::stoi(tmp));
		else 
			return tmp;
	}
	else if(find_key->second == "double") // double types
	{
		if (false == check_value_is_double(tmp))
		{
			std::cout << "\n\n" << " Error: Invalid keyword specification," 
					  << Keywords::err_mesg.at(keyword) << "\n";
			std::cout << "  Line: \"" << line << "\"\n\n";
			exit(EXIT_FAILURE);
		}
		else
			std::invoke(Keywords::set_setting<double>.at(get_keyword(line)), std::stod(tmp));
	}
	else // string types and boolean from string (can have duplicate keys)
	{
		bool found = false;
		const auto start = Keywords::lookup_keyword.equal_range(keyword).first;
		const auto end = Keywords::lookup_keyword.equal_range(keyword).second;

		for(auto iter = start; iter != end; ++iter)
			if(iter->second == tmp)
			{
				found = true;
				if(tmp != "true" && tmp != "false") // string
					std::invoke(Keywords::set_setting_str.at(get_keyword(line)), tmp);
				else if (tmp == "true") // bool true
					std::invoke(Keywords::set_setting<bool>.at(get_keyword(line)), true);
				else if (tmp == "false")  // bool false
					std::invoke(Keywords::set_setting<bool>.at(get_keyword(line)), false);

				goto done;
			}
done:
		if(!found)
		{
			std::cout << "\n\n" << " Error: Invalid keyword specification," 
					<< Keywords::err_mesg.at(keyword) << "\n";
			std::cout << "  Line: \"" << line << "\"\n\n";
			exit(EXIT_FAILURE);
		}
	}

	return std::nullopt;
}

std::string inline make_file_name_from_basis_set_name(const std::string& name)
{
	std::string tmp_name = name;
	std::string trans_name;

	std::for_each(tmp_name.begin(), tmp_name.end(), [] (char& c){ c = (char)std::tolower(c); });
	
	std::for_each(tmp_name.begin(), tmp_name.end(), [&trans_name] (char& c)
	{ 
		if(c == '*')
			trans_name += "_st_";
		else
			trans_name += c;
	});

	trans_name += ".gbs";
	return trans_name;
}

int inline parse_element(const std::string& line)
{
	std::string tmpline = line;
	        
	tmpline.replace(0, 1, "");
	while (tmpline.substr(0, 1) == " ") 
		tmpline.replace(0, 1, "");
	
	int Z = 0;
	
	if(std::isdigit(static_cast<unsigned char>(tmpline.c_str()[0])))
	{
		std::istringstream data(tmpline);
		data >> Z;
		return Z;
	}
	else
	{
		std::string atom_name;
		std::istringstream data(tmpline);	
		data >> atom_name;
		if (ELEMENTDATA::name_to_Z.find(atom_name) == ELEMENTDATA::name_to_Z.end())
		{
			std::cout << "\n\n  Error: Unsupported element.\n";
			std::cout << "   Line: Unsupported element.\n\n";
			exit(EXIT_FAILURE);
		}

		Z = ELEMENTDATA::name_to_Z.find(atom_name)->second;
		return Z;
	}
	
	return Z;
}

bool inline check_for_cartesian_format(const std::string& line)
{	
	std::istringstream data(line);	
	std::string token_first;
	std::string token_next; 
	
	data >> token_first;
	data >> token_next;
	
	if(token_first.empty())
	{
		std::cout << "  Error: No molecular coordinates found.\n";
		exit(EXIT_FAILURE);
	}
	else if (token_next.empty())
		return false;
		
	return true;
}

void inline check_coord_is_double(const std::string& val, const std::string& line)
{
	if (val.empty())
	{
		std::cout << "\n\n  Error: Incomplete coordinate on line containing:\n"
					          << "  " << line << "\n\n";
		exit(EXIT_FAILURE);
	}
	else if(val.find_first_not_of("+-0123456789.Ee") != std::string::npos)
	{
		std::cout << "\n\n  Error: Bad coordinate format on line containing:\n"
					          << "  " << line << "\n\n";
		exit(EXIT_FAILURE);
	}

	// we can still fall through when the above is not robust enough
	std::stringstream isgood(val);
	double isdbl;
	isgood >> isdbl;

	if(isgood.good())
	{
		std::cout << "\n\n  Error: Bad coordinate format on line containing:\n"
					          << "  " << line << "\n\n";
		exit(EXIT_FAILURE);
	}
}
}

#endif
// End PARSE_LINE
