// Copyright (c) 2019 Agenium Scale
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <nsimd/nsimd-all.hpp>

template <typename A>
void write_pgm(std::string const &filename, std::vector<uint8_t, A> const &img,
               size_t const nb_cols, size_t const nb_rows) {
  std::ofstream file(filename, std::ios::binary);
  file << "P5\n";
  file << nb_cols << " " << nb_rows << "\n";
  file << "255\n";

  for (size_t i = 0; i < img.size(); ++i) {
    file.put(reinterpret_cast<char const &>(img[i]));
  }
}

template <typename T>
void julia_set(int const size = 1024, int const max_iteration = 1000) {
  //   for (int i = 0; i < size; ++i) {
  //     T x0 = T(i) / T(size) * x_range + x_min;
  //     for (int j = 0; j < size; ++j) {
  //       int iteration = 0;
  //       T y0 = T(j) / T(size) * y_range + y_min;
  //       T x = 0;
  //       T y = 0;
  //       T x2 = x * x;
  //       T y2 = y * y;
  //       while (x2 + y2 < 4 && iteration < max_iteration) {
  //         x2 = x * x;
  //         y2 = y * y;
  //         T x_temp = x2 - y2 + x0;
  //         y = 2 * x * y + y0;
  //         x = x_temp;
  //         ++iteration;
  //       }
  //       iterations[j + i * size] = iteration;
  //     }
  //   }
}

int main() {

  julia_set<float>();

  return 0;
}
