/*

Copyright (c) 2019 Agenium Scale

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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <nsimd/modules/fixed_point.hpp>

int main() {
  int32_t vals[8];
  memset(vals, 0, 8 * sizeof(int32_t));
  vals[0] = 1;
  
  fprintf(stdout, "Val : ");
  for (size_t i = 0; i < 8; i++) {
    fprintf(stdout, "%d ", vals[i]);
  }
  fprintf(stdout, "\n");
  nsimd::fixed_point::pack<16, 16> v_test =
      nsimd::fixed_point::loadu<16, 16>((nsimd::fixed_point::fp_t<16, 16> *)vals);

  int32_t res[8];
  nsimd::fixed_point::storeu<16, 16>((nsimd::fixed_point::fp_t<16, 16> *)res, v_test);

  fprintf(stdout, "Res : ");
  for (size_t i = 0; i < 8; i++) {
    fprintf(stdout, "%d ", res[i]);
  }
  fprintf(stdout, "\n");

  return 0;
}
