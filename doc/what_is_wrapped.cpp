/*

Copyright (c) 2021 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

/*

This little C++ program reads and parses files from NSIMD wrapping intrinsics
in order to build a markdown page describing in a table which operators are
just intrinsics wrapper and which one are more complicated. We only to parse
C code so no need for complicated stuff. Moreover what we doo is really simple
and a C parser is not needed.

We replace all C delimiters by spaces, then split the resulting string into
words and we get a vector of strings. Then search in it the function that we
want (say nsimd_add_sse2_f32) along with its opening curly and closing
brakets and finally:
- if there is only one token then it must be an intrinsic
- if there is a for then it must use emulation
- if there are several tokens but no for it must be a trick using other
  intrinsics

The produced markdown contains:
- E for emulation
- T for trick with other intrinsics
- NOOP for noop
- a link to the Intel/Arm documentation about the intrinsic otherwise

Well all that to say that a few hundreds of simple C++ code is more that
enough for our need and we don't need to depend on some C/C++ parser such
as Clang. Note that using a real parser will be counter productive as some
intrinsics are implemented as macros to compiler builtin which then appear
in the AST instead of the documented intrinsics.

This code is completely non-optimized and we don't care because it does not
take time to execute and it is not our purpose to optimize this code.

*/

// ----------------------------------------------------------------------------

#include <ns2.hpp>

#include <utility>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------

#define MAX_LEN (11 * 11)

typedef std::map<std::string, std::string[MAX_LEN]> table_t;

std::string type_names_str("i8,u8,i16,u16,i32,u32,i64,u64,f16,f32,f64");
std::vector<std::string> types_list(ns2::split(type_names_str, ","));

const size_t not_found = ~((size_t)0);

// ----------------------------------------------------------------------------

int nbits(std::string const &typ) {
  if (typ == "i8" || typ == "u8") {
    return 8;
  } else {
    return (10 * (typ[1] - '0')) + (typ[2] - '0');
  }
}

// ----------------------------------------------------------------------------

std::vector<std::string> get_types_names(std::string const &output) {
  std::vector<std::string> const& list = types_list;
  if (output == "same") {
    return list;
  }
  std::vector<std::string> ret;
  for (size_t i = 0; i < list.size(); i++) {
    for (size_t j = 0; j < list.size(); j++) {
      if ((output == "same_size" && nbits(list[j]) == nbits(list[i])) ||
          (output == "bigger_size" && nbits(list[j]) == 2 * nbits(list[i])) ||
          (output == "lesser_size" && 2 * nbits(list[j]) == nbits(list[i]))) {
        ret.push_back(list[j] + "_" + list[i]);
      }
    }
  }
  return ret;
}

// ----------------------------------------------------------------------------

size_t find(std::vector<std::string> const &haystack,
            std::string const &needle, size_t i0 = 0) {
  for (size_t i = i0; i < haystack.size(); i++) {
    if (haystack[i] == needle) {
      return i;
    }
  }
  return not_found;
}

// ----------------------------------------------------------------------------

size_t find_by_prefix(std::vector<std::string> const &needles,
                      std::string const &haystack) {
  for (size_t i = 0; i < needles.size(); i++) {
    if (ns2::startswith(haystack, needles[i])) {
      return i;
    }
  }
  return not_found;
}

// ----------------------------------------------------------------------------

int is_number(std::string const &s) {
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] != 'x' && s[i] != 'l' && s[i] != 'L' && s[i] != 'u' &&
        s[i] != 'U' && !(s[i] >= '0' && s[i] <= '9')) {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------------

int is_macro(std::string const &s) {
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] != '_' && !(s[i] >= 'A' && s[i] <= 'Z') &&
        !(s[i] >= 'a' && s[i] <= 'z')) {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------------

void parse_file(std::string const &input_vars, std::string const &simd_ext,
                std::vector<std::string> const &types_names,
                std::string const &op_name, std::string const &filename,
                table_t *table_) {
  table_t &table = *table_;
  std::string content(ns2::read_file(filename));

  // replace all C delimiters by spaces except {}
  for (size_t i = 0; i < content.size(); i++) {
    const char delims[] = "()[];,:+-*/%&|!%\n\t\r";
    for (size_t j = 0; j < sizeof(delims); j++) {
      if (content[i] == delims[j]) {
        content[i] = ' ';
        break;
      }
    }
  }

  // replace '{' by ' { ' and same for '}' in case there are some code
  // just before/after it
  content = ns2::replace(ns2::replace(content, "}", " } "), "{", " { ");

  // now split string on spaces and removes some tokens
  std::vector<std::string> to_be_removed(
      ns2::split("return,signed,unsigned,char,short,int,long,float,double,"
                 "const,void,__vector,__bool,bool,vector" +
                     type_names_str + "," + input_vars,
                 ','));
  std::vector<std::string> to_be_removed_by_prefix(ns2::split(
      "_mm_cast,_mm256_cast,_mm512_cast,vreinterpret,svreinterpret,svptrue_",
      ','));
  std::vector<std::string> tokens;
  { // to free tokens0 afterwards
    std::vector<std::string> tokens0 = ns2::split(content, ' ');
    for (size_t i = 0; i < tokens0.size(); i++) {
      // We also remove svptrue_* as they are everywhere for SVE and all
      // casts as they incur no opcode and are often used for intrinsics
      // not supporting certain types
      if (tokens0[i].size() == 0 || is_number(tokens0[i]) ||
          is_macro(tokens0[i]) ||
          find_by_prefix(to_be_removed_by_prefix, tokens0[i]) != not_found ||
          find(to_be_removed, tokens0[i]) != not_found) {
        continue;
      }
      tokens.push_back(tokens0[i]);
    }
  }

  // finally search for intrinsics
  for (size_t typ = 0; typ < types_names.size(); typ++) {
    std::string func_name("nsimd_" + op_name + "_" + simd_ext + "_" +
                          types_names[typ]);

    // find func_name
    size_t pos = find(tokens, func_name);
    if (pos == not_found) {
      std::cerr << "WARNING: cannot find function '" << func_name << "' in '"
                << filename << "'\n";
      table[op_name][typ] = "NA";
      continue;
    }

    // find opening {
    size_t i0 = find(tokens, "{", pos);
    if (i0 == not_found) {
      std::cerr << "WARNING: cannot find opening '{' for '" << func_name
                << "' in '" << filename << "'\n";
      table[op_name][typ] = "NA";
      continue;
    }

    // find closing }
    size_t i1 = i0;
    int nest = 0;
    for (i1 = i0; i1 < tokens.size(); i1++) {
      if (tokens[i1] == "{") {
        nest++;
      } else if (tokens[i1] == "}") {
        nest--;
      }
      if (nest == 0) {
        break;
      }
    }

    // if there is no token inside {} then it must be a noop
    // if there is only one token inside {} then it must be the intrinsic
    // if there is a for loop then it must be emulation
    // if there are several tokens but no for then it must be a trick
    if (i0 + 1 == i1) {
      table[op_name][typ] = "NOOP";
    } else if (i0 + 2 == i1 && !ns2::startswith(tokens[i0 + 1], "nsimd_")) {
      table[op_name][typ] = "[`" + tokens[i0 + 1] + "`]";
      if (simd_ext == "neon128" || simd_ext == "aarch64") {
        table[op_name][typ] +=
            "(https://developer.arm.com/architectures/instruction-sets/"
            "intrinsics/" + tokens[i0 + 1] + ")";
      } else if (ns2::startswith(simd_ext, "sve")) {
        table[op_name][typ] +=
            "(https://developer.arm.com/documentation/100987/0000)";
      } else if (simd_ext == "sse2" || simd_ext == "sse42" ||
                 simd_ext == "avx" || simd_ext == "avx2" ||
                 simd_ext == "avx512_knl" || simd_ext == "avx512_skylake") {
        table[op_name][typ] += "(https://software.intel.com/sites/landingpage/"
                               "IntrinsicsGuide/#text=" +
                               tokens[i0 + 1] + ")";
      } else if (simd_ext == "vsx" || simd_ext == "vmx") {
        table[op_name][typ] +=
            "(https://www.ibm.com/docs/en/xl-c-aix/13.1.3?topic=functions-" +
            ns2::replace(tokens[i0 + 1], "_", "-") + ")";
      }
    } else {
      if (find(std::vector<std::string>(tokens.begin() + i0,
                                        tokens.begin() + (i1 + 1)),
               "for") != not_found) {
        table[op_name][typ] = "E";
      } else {
        table[op_name][typ] = "T";
      }
    }
  }
}

// ----------------------------------------------------------------------------

std::string md_row(int nb_col, std::string const &cell_content) {
  std::string ret("|");
  for (int i = 0; i < nb_col; i++) {
    ret += cell_content + "|";
  }
  return ret;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  if ((argc % 2) != 0 || argc <= 5) {
    std::cout
        << "Usage: " << argv[0]
        << " a0,a1,a2 simd_ext output_type operator1 file1 operator2 file2 "
           "...\n"
        << "where output_type is (same|same_size|bigger_size|lesser_size)"
        << std::endl;
    return 1;
  }

  std::string input_vars(argv[1]);
  std::string simd_ext(argv[2]);
  std::string output_type(argv[3]);
  std::vector<std::string> types_names = get_types_names(output_type);
  table_t table;

  for (int i = 4; i < argc; i += 2) {
    parse_file(input_vars, simd_ext, types_names, argv[i], argv[i + 1],
               &table);
  }

  for (table_t::const_iterator it = table.begin(); it != table.end(); it++) {
    std::cout << "## " << it->first << "\n\n";
    if (output_type == "same") {
      const std::string(&row)[MAX_LEN] = it->second;
      for (size_t i = 0; i < types_list.size(); i++) {
        std::cout << "-  " << it->first << " on **" << types_list[i]
                  << "**: " << row[i] << "\n";
      }
      std::cout << "\n\n";
    } else {
      const std::string(&row)[MAX_LEN] = it->second;
      for (size_t i = 0; i < types_list.size(); i++) {
        for (size_t j = 0; j < types_list.size(); j++) {
          std::string cell_content;
          std::string typ(types_list[j] + "_" + types_list[i]);
          for (size_t k = 0; k < types_names.size(); k++) {
            if (typ == types_names[k]) {
              cell_content = row[k];
              break;
            }
          }
          if (cell_content.size() > 0) {
            std::cout << "-  " << it->first << " from **" << types_list[i]
                      << "** to **" << types_list[j] << "**: " << cell_content
                      << "\n";
          }
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  return 0;
}
