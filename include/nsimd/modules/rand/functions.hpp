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

/* This file has been auto-generated */

#ifndef NSIMD_MODULES_RAND_FUNCTIONS_HPP
#define NSIMD_MODULES_RAND_FUNCTIONS_HPP

#include <nsimd/nsimd.h>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
namespace nsimd {
namespace rand {
/* ------------------------------------------------------------------------- */

nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key);
#if 1
void mulhilo32(pack<u32> a, pack<u32> b, pack<u32> *low, pack<u32> *high) {
  nsimd::packx2<u64> a64 = nsimd::upcvt(nsimd::packx2<u64>(), a);
  nsimd::packx2<u64> b64 = nsimd::upcvt(nsimd::packx2<u64>(), b);

  nsimd::packx2<u64> product;
  product.v0 = a64.v0 * b64.v0;
  product.v1 = a64.v1 * b64.v1;

  *high =
      nsimd::downcvt(nsimd::pack<u32>(), product.v0 >> 32, product.v1 >> 32);
  *low = nsimd::downcvt(nsimd::pack<u32>(), product.v0, product.v1);
}

#else

void mulhilo32(pack<u32> a, pack<u32> b, pack<u32> *low, pack<u32> *high) {
  nsimd::pack<u32> ah = nsimd::shr(a, 16);
  nsimd::pack<u32> bh = nsimd::shr(b, 16);

  nsimd::pack<u32> al = nsimd::shr(nsimd::shl(a, 16), 16);
  nsimd::pack<u32> bl = nsimd::shr(nsimd::shl(b, 16), 16);

  nsimd::pack<u32> ahbh = ah * bh;
  nsimd::pack<u32> ahbl = ah * bl;
  nsimd::pack<u32> albh = al * bh;
  nsimd::pack<u32> albl = al * bl;

  nsimd::pack<u32> tmp1 = nsimd::shl(albh, 16);
  nsimd::pack<u32> tmp2 = nsimd::shl(ahbl, 16);

  nsimd::pack<u32> tmp3 = tmp1 + tmp2;

  nsimd::pack<u32> _1 = nsimd::set1(nsimd::pack<u32>(), 1u);
  nsimd::pack<u32> _0 = nsimd::set1(nsimd::pack<u32>(), 0u);

  nsimd::pack<u32> carry =
      nsimd::if_else1((tmp3 < tmp1) || (tmp3 < tmp2), _1, _0);

  *low = tmp3 + albl;

  carry = carry + nsimd::if_else1((*low < tmp3) || (*low < albl), _1, _0);

  *high = ahbh + nsimd::shr(albh, 16) + nsimd::shr(ahbl, 16) + carry;
}
#endif

#if 0
void mulhilo64(pack<u64> a, pack<u64> b, pack<u64> *low, pack<u64> *high) {
  u64 a_buf[8];
  u64 b_buf[8];
  u64 low_buf[8];
  u64 high_buf[8];

  nsimd::storeu(a_buf, a);
  nsimd::storeu(b_buf, b);

  for (int i = 0; i < nsimd::len(u64()); ++i) {
    __uint128_t product = ((__uint128_t)a_buf[i]) * ((__uint128_t)b_buf[i]);
    high_buf[i] = (u64)(product >> 64);
    low_buf[i] = (u64)product;
  }

  *high = nsimd::loadu(high_buf, u64());
  *low = nsimd::loadu(low_buf, u64());
}

#else

void mulhilo64(pack<u64> a, pack<u64> b, pack<u64> *low, pack<u64> *high) {
  nsimd::pack<u64> ah = nsimd::shr(a, 32);
  nsimd::pack<u64> bh = nsimd::shr(b, 32);

  nsimd::pack<u64> al = nsimd::shr(nsimd::shl(a, 32), 32);
  nsimd::pack<u64> bl = nsimd::shr(nsimd::shl(b, 32), 32);

  nsimd::pack<u64> ahbh = ah * bh;
  nsimd::pack<u64> ahbl = ah * bl;
  nsimd::pack<u64> albh = al * bh;
  nsimd::pack<u64> albl = al * bl;

  nsimd::pack<u64> tmp1 = nsimd::shl(albh, 32);
  nsimd::pack<u64> tmp2 = nsimd::shl(ahbl, 32);

  nsimd::pack<u64> tmp3 = tmp1 + tmp2;

  nsimd::pack<u64> _1 = nsimd::set1(nsimd::pack<u64>(), 1lu);
  nsimd::pack<u64> _0 = nsimd::set1(nsimd::pack<u64>(), 0lu);

  nsimd::pack<u64> carry =
      nsimd::if_else1((tmp3 < tmp1) || (tmp3 < tmp2), _1, _0);

  *low = tmp3 + albl;

  carry = carry + nsimd::if_else1((*low < tmp3) || (*low < albl), _1, _0);

  *high = ahbh + nsimd::shr(albh, 32) + nsimd::shr(ahbl, 32) + carry;
}
#endif
    nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx2<u32> out;nsimd::pack<u32> bump = nsimd::set1(nsimd::pack<u32>(), 0x9E3779B9U);
            nsimd::pack<u32> mul = nsimd::set1(nsimd::pack<u32>(), 0xD256D193U);
            nsimd::pack<u32> high, low;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo32(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx4<u32> out;
                nsimd::pack<u32> bump0 = nsimd::set1(nsimd::pack<u32>(), 0x9E3779B9U);
                nsimd::pack<u32> bump1 = nsimd::set1(nsimd::pack<u32>(), 0xBB67AE85U);
                nsimd::pack<u32> mul0 = nsimd::set1(nsimd::pack<u32>(), 0xD2511F53U);
                nsimd::pack<u32> mul1 = nsimd::set1(nsimd::pack<u32>(), 0xCD9E8D57U);
                nsimd::pack<u32> low0, high0, low1, high1;
                
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx4<u32> out;
                nsimd::pack<u32> bump0 = nsimd::set1(nsimd::pack<u32>(), 0x9E3779B9U);
                nsimd::pack<u32> bump1 = nsimd::set1(nsimd::pack<u32>(), 0xBB67AE85U);
                nsimd::pack<u32> mul0 = nsimd::set1(nsimd::pack<u32>(), 0xD2511F53U);
                nsimd::pack<u32> mul1 = nsimd::set1(nsimd::pack<u32>(), 0xCD9E8D57U);
                nsimd::pack<u32> low0, high0, low1, high1;
                
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo32(mul0, in.v0, &low0, &high0);
              mulhilo32(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx2<u64> out;nsimd::pack<u64> bump = nsimd::set1(nsimd::pack<u64>(), 0x9E3779B97F4A7C15UL);
            nsimd::pack<u64> mul = nsimd::set1(nsimd::pack<u64>(), 0xD2B74407B1CE6E93UL);
            nsimd::pack<u64> high, low;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx2<u64> out;nsimd::pack<u64> bump = nsimd::set1(nsimd::pack<u64>(), 0x9E3779B97F4A7C15UL);
            nsimd::pack<u64> mul = nsimd::set1(nsimd::pack<u64>(), 0xD2B74407B1CE6E93UL);
            nsimd::pack<u64> high, low;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            key.v0 = key.v0 + bump;
              mulhilo64(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx4<u64> out;
                nsimd::pack<u64> bump0 = nsimd::set1(nsimd::pack<u64>(), 0x9E3779B97F4A7C15UL);
                nsimd::pack<u64> bump1 = nsimd::set1(nsimd::pack<u64>(), 0xBB67AE8584CAA73BUL);
                nsimd::pack<u64> mul0 = nsimd::set1(nsimd::pack<u64>(), 0xD2E7470EE14C6C93UL);
                nsimd::pack<u64> mul1 = nsimd::set1(nsimd::pack<u64>(), 0xCA5A826395121157UL);
                nsimd::pack<u64> low0, high0, low1, high1;
                
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)}(nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{self.get_key_size(nwords)}<u{word_size}> key) { nsimd::packx4<u64> out;
                nsimd::pack<u64> bump0 = nsimd::set1(nsimd::pack<u64>(), 0x9E3779B97F4A7C15UL);
                nsimd::pack<u64> bump1 = nsimd::set1(nsimd::pack<u64>(), 0xBB67AE8584CAA73BUL);
                nsimd::pack<u64> mul0 = nsimd::set1(nsimd::pack<u64>(), 0xD2E7470EE14C6C93UL);
                nsimd::pack<u64> mul1 = nsimd::set1(nsimd::pack<u64>(), 0xCA5A826395121157UL);
                nsimd::pack<u64> low0, high0, low1, high1;
                
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            key.v0 = key.v0 + bump0;
key.v1 = key.v1 + bump1;
              mulhilo64(mul0, in.v0, &low0, &high0);
              mulhilo64(mul1, in.v2, &low1, &high1);

              in.v0 = high1 ^ key.v0 ^ in.v1;
              in.v1 = low1;
              in.v2 = high0 ^ key.v1 ^ in.v3;
              in.v3 = low0;
            
            return in;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key);
    enum enum_threefry32x2_rotations {
      Rot_32x2_0 = 13,
      Rot_32x2_1 = 15,
      Rot_32x2_2 = 26,
      Rot_32x2_3 = 6,
      Rot_32x2_4 = 17,
      Rot_32x2_5 = 29,
      Rot_32x2_6 = 16,
      Rot_32x2_7 = 24
    };

    enum enum_threefry32x4_rotations {
      Rot_32x4_0_0 = 10,
      Rot_32x4_0_2 = 26,
      Rot_32x4_1_0 = 11,
      Rot_32x4_1_2 = 21,
      Rot_32x4_2_0 = 13,
      Rot_32x4_2_2 = 27,
      Rot_32x4_3_0 = 23,
      Rot_32x4_3_2 = 5,
      Rot_32x4_4_0 = 6,
      Rot_32x4_4_2 = 20,
      Rot_32x4_5_0 = 17,
      Rot_32x4_5_2 = 11,
      Rot_32x4_6_0 = 25,
      Rot_32x4_6_2 = 10,
      Rot_32x4_7_0 = 18,
      Rot_32x4_7_2 = 20
    };

    enum enum_threefry64x2_rotations {
      Rot_64x2_0 = 16,
      Rot_64x2_1 = 42,
      Rot_64x2_2 = 12,
      Rot_64x2_3 = 31,
      Rot_64x2_4 = 16,
      Rot_64x2_5 = 32,
      Rot_64x2_6 = 24,
      Rot_64x2_7 = 21
    };
    enum enum_threefry64x4_rotations {

      Rot_64x4_0_0 = 14,
      Rot_64x4_0_2 = 16,
      Rot_64x4_1_0 = 52,
      Rot_64x4_1_2 = 57,
      Rot_64x4_2_0 = 23,
      Rot_64x4_2_2 = 40,
      Rot_64x4_3_0 = 5,
      Rot_64x4_3_2 = 37,
      Rot_64x4_4_0 = 25,
      Rot_64x4_4_2 = 33,
      Rot_64x4_5_0 = 46,
      Rot_64x4_5_2 = 12,
      Rot_64x4_6_0 = 58,
      Rot_64x4_6_2 = 22,
      Rot_64x4_7_0 = 32,
      Rot_64x4_7_2 = 32
    };
    
    #define SHIFT_MOD_32(x, N) ((x << (N & 31)) | (x >> ((32 - N) & 31)))
    #define SHIFT_MOD_64(x, N) ((x << (N & 63)) | (x >> ((64 - N) & 63)))
    nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx2<u32> out;nsimd::pack<u32> ks2 =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAu);
        nsimd::pack<u32> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks2 = ks2 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u32> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks2 = ks2 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u32>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u32>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u32>(3);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx2<u32> out;nsimd::pack<u32> ks2 =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAu);
        nsimd::pack<u32> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks2 = ks2 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u32> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks2 = ks2 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u32>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u32>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u32>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u32>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u32>(5);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx2<u32> out;nsimd::pack<u32> ks2 =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAu);
        nsimd::pack<u32> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks2 = ks2 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u32> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks2 = ks2 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u32>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u32>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u32>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u32>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u32>(5);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u32>(6);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u32>(7);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1, Rot_32x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u32>(8);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx4<u32> out;nsimd::pack<u32> ks4 =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAu);
        nsimd::pack<u32> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks4 = ks4 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u32> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks4 = ks4 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
        nsimd::pack<u32> ks2;
        ks2 = key.v2;
        out.v2 = in.v2;
        ks4 = ks4 ^ key.v2;
        out.v2 = out.v2 + key.v2;
        
        nsimd::pack<u32> ks3;
        ks3 = key.v3;
        out.v3 = in.v3;
        ks4 = ks4 ^ key.v3;
        out.v3 = out.v3 + key.v3;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u32>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u32>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u32>(3);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx4<u32> out;nsimd::pack<u32> ks4 =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAu);
        nsimd::pack<u32> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks4 = ks4 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u32> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks4 = ks4 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
        nsimd::pack<u32> ks2;
        ks2 = key.v2;
        out.v2 = in.v2;
        ks4 = ks4 ^ key.v2;
        out.v2 = out.v2 + key.v2;
        
        nsimd::pack<u32> ks3;
        ks3 = key.v3;
        out.v3 = in.v3;
        ks4 = ks4 ^ key.v3;
        out.v3 = out.v3 + key.v3;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u32>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u32>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u32>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u32>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u32>(5);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx4<u32> out;nsimd::pack<u32> ks4 =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAu);
        nsimd::pack<u32> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks4 = ks4 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u32> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks4 = ks4 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
        nsimd::pack<u32> ks2;
        ks2 = key.v2;
        out.v2 = in.v2;
        ks4 = ks4 ^ key.v2;
        out.v2 = out.v2 + key.v2;
        
        nsimd::pack<u32> ks3;
        ks3 = key.v3;
        out.v3 = in.v3;
        ks4 = ks4 ^ key.v3;
        out.v3 = out.v3 + key.v3;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u32>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u32>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u32>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u32>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u32>(5);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u32>(6);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u32>(7);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u32>(8);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u32>(9);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u32>(10);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u32>(11);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u32>(12);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u32>(13);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u32>(14);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u32>(15);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u32>(16);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u32>(17);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                           Rot_32x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                         Rot_32x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_32(out.v3,
                                           Rot_32x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_32(out.v1,
                                         Rot_32x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u32>(18);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx2<u64> out;nsimd::pack<u64> ks2 =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22u);
        nsimd::pack<u64> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks2 = ks2 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u64> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks2 = ks2 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u64>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u64>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u64>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx2<u64> out;nsimd::pack<u64> ks2 =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22u);
        nsimd::pack<u64> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks2 = ks2 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u64> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks2 = ks2 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u64>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u64>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u64>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u64>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u64>(5);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx2<u64> out;nsimd::pack<u64> ks2 =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22u);
        nsimd::pack<u64> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks2 = ks2 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u64> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks2 = ks2 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u64>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u64>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u64>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u64>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u64>(5);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v1 = out.v1 + nsimd::pack<u64>(6);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_1);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_2);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_3);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v1 = out.v1 + nsimd::pack<u64>(7);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_4);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_5);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_6);
                out.v1 = out.v1 ^ out.v0;
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1, Rot_64x2_7);
                out.v1 = out.v1 ^ out.v0;
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks0;
out.v1 = out.v1 + nsimd::pack<u64>(8);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx4<u64> out;nsimd::pack<u64> ks4 =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22u);
        nsimd::pack<u64> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks4 = ks4 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u64> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks4 = ks4 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
        nsimd::pack<u64> ks2;
        ks2 = key.v2;
        out.v2 = in.v2;
        ks4 = ks4 ^ key.v2;
        out.v2 = out.v2 + key.v2;
        
        nsimd::pack<u64> ks3;
        ks3 = key.v3;
        out.v3 = in.v3;
        ks4 = ks4 ^ key.v3;
        out.v3 = out.v3 + key.v3;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u64>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u64>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u64>(3);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx4<u64> out;nsimd::pack<u64> ks4 =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22u);
        nsimd::pack<u64> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks4 = ks4 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u64> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks4 = ks4 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
        nsimd::pack<u64> ks2;
        ks2 = key.v2;
        out.v2 = in.v2;
        ks4 = ks4 ^ key.v2;
        out.v2 = out.v2 + key.v2;
        
        nsimd::pack<u64> ks3;
        ks3 = key.v3;
        out.v3 = in.v3;
        ks4 = ks4 ^ key.v3;
        out.v3 = out.v3 + key.v3;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u64>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u64>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u64>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u64>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u64>(5);

            return out;
        }
        nsimd::packx{nwords}<u{word_size}> {self.gen_function_name(nwords, word_size, nrounds)} (nsimd::packx{nwords}<u{word_size}> in, nsimd::packx{nwords}<u{word_size}> key) { nsimd::packx4<u64> out;nsimd::pack<u64> ks4 =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22u);
        nsimd::pack<u64> ks0;
        ks0 = key.v0;
        out.v0 = in.v0;
        ks4 = ks4 ^ key.v0;
        out.v0 = out.v0 + key.v0;
        
        nsimd::pack<u64> ks1;
        ks1 = key.v1;
        out.v1 = in.v1;
        ks4 = ks4 ^ key.v1;
        out.v1 = out.v1 + key.v1;
        
        nsimd::pack<u64> ks2;
        ks2 = key.v2;
        out.v2 = in.v2;
        ks4 = ks4 ^ key.v2;
        out.v2 = out.v2 + key.v2;
        
        nsimd::pack<u64> ks3;
        ks3 = key.v3;
        out.v3 = in.v3;
        ks4 = ks4 ^ key.v3;
        out.v3 = out.v3 + key.v3;
        
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u64>(1);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u64>(2);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u64>(3);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u64>(4);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u64>(5);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u64>(6);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u64>(7);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u64>(8);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u64>(9);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u64>(10);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u64>(11);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u64>(12);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u64>(13);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks4;
out.v1 = out.v1 + ks0;
out.v2 = out.v2 + ks1;
out.v3 = out.v3 + ks2;
out.v3 = out.v3 + nsimd::pack<u64>(14);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks0;
out.v1 = out.v1 + ks1;
out.v2 = out.v2 + ks2;
out.v3 = out.v3 + ks3;
out.v3 = out.v3 + nsimd::pack<u64>(15);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks1;
out.v1 = out.v1 + ks2;
out.v2 = out.v2 + ks3;
out.v3 = out.v3 + ks4;
out.v3 = out.v3 + nsimd::pack<u64>(16);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_0_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_0_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_1_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_1_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_2_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_2_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_3_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_3_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks2;
out.v1 = out.v1 + ks3;
out.v2 = out.v2 + ks4;
out.v3 = out.v3 + ks0;
out.v3 = out.v3 + nsimd::pack<u64>(17);

                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_4_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_4_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_5_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_5_2);
                out.v1 = out.v1 ^ out.v2;
                
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                           Rot_64x4_6_0);
                out.v1 = out.v1 ^ out.v0;
                out.v2 = out.v2 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                         Rot_64x4_6_2);
                out.v3 = out.v3 ^ out.v2;
                
                out.v0 = out.v0 + out.v3;
                out.v3 = SHIFT_MOD_64(out.v3,
                                           Rot_64x4_7_0);
                out.v3 = out.v3 ^ out.v0;
                out.v2 = out.v2 + out.v1;
                out.v1 = SHIFT_MOD_64(out.v1,
                                         Rot_64x4_7_2);
                out.v1 = out.v1 ^ out.v2;
                
out.v0 = out.v0 + ks3;
out.v1 = out.v1 + ks4;
out.v2 = out.v2 + ks0;
out.v3 = out.v3 + ks1;
out.v3 = out.v3 + nsimd::pack<u64>(18);

            return out;
        }
        
    #undef SHIFT_MOD_32
    #undef SHIFT_MOD_64
    } // namespace nsimd
} // namespace rand
#endif

