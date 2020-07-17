/*

Copyright (c) 2020 Agenium Scale

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

#include <ns2.hpp>

#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------

enum typeid_t { i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64 };

#define LEN 11

char *type_names[LEN] = {"i8",  "u8",  "i16", "u16", "i32", "u32",
                         "i64", "u64", "f16", "f32", "f64"};

typedef std::map<std::string, std::string[LEN]> table_t;

// ----------------------------------------------------------------------------

void parse_file(std::string const &simd_ext, std::string const &op_name,
                std::string const &filename, table_t *table_) {
  table_t &table = *table_;
  std::string content(ns2::read_file(filename));

  // replace all C delimiters by spaces except {}
  for (size_t i = 0; i < content.size(); i++) {
    const char delims[] = "()[];,:+-*/%&|!%";
    for (size_t j = 0; j < sizeof(delims); j++) {
      if (content[i] == delims[j]) {
        content[i] = ' ';
        break;
      }
    }
  }

  // now split string on spaces and removes 'returns', empty tokens
  std::vector<std::string> tokens = ns2::split(content, ' ');
  for (size_t i = 0; i < tokens; i++) {
  }

  for (int i = 0; i < LEN; i++) {
    std::string func_name("nsimd_" + op_name + "_" + simd_ext + "_" +
                          type_names[i]);

    size_t pos = content.find(func_name);
    if (pos == std::string::npos) {
      std::cout << "WARNING: cannot find function '" << func_name << "' in '"
                << filename << "'\n";
    }

  }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  if ((argc % 2) != 0 || argc <= 3) {
    std::cout << "Usage: " << argv[0]
              << " simd_ext operator1 file1 operator2 file2 ..." << std::endl;
    return 1;
  }

  std::string simd_ext(argv[1]);
  table_t table;

  for (int i = 2; i < argc; i += 2) {
    parse_file(simd_ext, argv[i], argv[i + 1], &table);
  }

  return 0;
}
