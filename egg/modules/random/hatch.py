# Copyright (c) 2020 Agenium Scale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import common
import collections


# -----------------------------------------------------------------------------

rand_functions = list()


class MAddToRands(type):
    def __new__(cls, name, bases, dct):
        ret = type.__new__(cls, name, bases, dct)
        if name != 'Rand':
            rand_functions.append(ret())
        return ret

class Rand(object, metaclass=MAddToRands):
    def gen_function_name(self, nwords, word_size, nrounds):
        return '{}_{}x{}_{}'.format(self.name, nwords, word_size, nrounds)

    def gen_headers(self, opts):
        res = ''

        for word_size, nwords_nrounds in self.wordsize_nwords_nrounds.items():
            for nwords, list_nrounds in nwords_nrounds.items():
                for nrounds in list_nrounds:
                    res += self.gen_signature(nwords, word_size, nrounds)+';'

        return res

    def gen_tests(self, opts, nrounds, word_size, nwords):

        key_size = self.get_key_size(nwords)

        key_initialization = 'nsimd::packx{}<u{}> key_pack;'. \
                format(key_size, word_size)
        for i in range (0, key_size):
            key_initialization += '''
            i = {i};
            for (int j = 0; j < len; j++) {{
              key[j + i * len] = (u{word_size})(j + i * len);
            }}
            key_pack.v{i} = nsimd::loadu(&key[i*len], u{word_size}());
            '''.format(i=i, word_size=word_size)

        input_initilization = \
                'memset(in, 0, sizeof(u{}) * {} * ulen);\n'. \
                format(word_size, nwords)
        for i in range (0, nwords):
            input_initilization += 'in_pack.v{} = nsimd::pack<u{}>(0);'. \
                    format(i, word_size)

        compare = ''
        for i in range (0, nwords):
            compare += '''
                if (i=={i}) {{
                    nsimd::storeu(out_nsimd, out_pack.v{i});
                }}
                '''.format(i=i)

        l = 'll' if word_size == 64 else ''

        res = '''
        #include <nsimd/modules/random/functions.hpp>
        #include "reference.hpp"
        #include <iostream>

        #ifdef NSIMD_LONGLONG_IS_EXTENSION
          #if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
            #pragma GCC diagnostic ignored "-Wformat"
          #endif
        #endif

        int main() {{
          int res = EXIT_SUCCESS;
          printf("Test of {function_name} ...\\n");

          nsimd::packx{nwords}<u{word_size}> in_pack;
          nsimd::packx{nwords}<u{word_size}> out_pack;

          const int len = nsimd::len(u{word_size}());
          const unsigned int ulen = (unsigned int)len;

          u{word_size} *key = (u{word_size}*)malloc(ulen *
                                sizeof(u{word_size}) * {key_size});
          u{word_size} *in = (u{word_size}*)malloc(ulen *
                               sizeof(u{word_size}) * {nwords});
          u{word_size} *out = (u{word_size}*)malloc(ulen *
                                sizeof(u{word_size}) * {nwords});
          u{word_size} *out_nsimd = (u{word_size}*)malloc(ulen *
                                      sizeof(u{word_size}));

          tab{word_size}x{nwords}_t in_ref;
          tab{word_size}x{key_size}_t key_ref;
          tab{word_size}x{nwords}_t out_ref;

          int i;

          // Keys
          {key_initialization}

          {input_initilization}

          for (int cpt=0; cpt < 100000; ++cpt) {{
            out_pack = nsimd::random::{function_name}(in_pack, key_pack);

            for (int i=0; i<len; ++i) {{
              for (int j=0; j<{nwords}; ++j) {{
                  in_ref.v[j] = in[i + j * len];
              }}

              for (int j=0; j<{key_size}; ++j) {{
                  key_ref.v[j] = key[i + j*len];
              }}

              out_ref = branson_{name}{nwords}x{word_size}_R({nrounds},
                          in_ref, key_ref);

              for (int j=0; j<{nwords}; ++j) {{
                  out[i + j * len] = out_ref.v[j];
              }}
            }}

            for (int i=0; i<{nwords}; ++i) {{
              {compare}

              if (memcmp(out_nsimd, &out[i * len],
                         ulen * sizeof(u{word_size}))) {{
                printf ("%i\\n", i);
                for (int j=0; j<len; ++j) {{
                  printf ("%{l}u\\t(0x%{l}x)\\t\\t%{l}u\\t(0x%{l}x)\\n",
                          out[j+i*len], out[j+i*len],
                          out_nsimd[j], out_nsimd[j]);
                }}

                res = EXIT_FAILURE;
                printf("... FAILED\\n");
                goto cleanup;
              }}
            }}

            in_pack = out_pack;
            memcpy(in, out, sizeof(u{word_size}) * {nwords} * ulen);
          }}

          fprintf(stdout, "... OK\\n");

        cleanup:
          free(key);
          free(in);
          free(out);
          free(out_nsimd);

          return res;
        }}
        '''.format(function_name=self.gen_function_name(nwords, word_size,
                   nrounds), word_size=word_size, key_size=key_size,
                   nwords=nwords, key_initialization=key_initialization,
                   nrounds=nrounds, input_initilization=input_initilization,
                   compare=compare, l=l, name = self.name)

        # Write file
        return res

class Philox(Rand):
    name = 'philox'

    wordsize_nwords_nrounds = {32: {2: [10],
                                    4: [7, 10]},
                               64: {2: [6, 10],
                                    4: [7, 10]}}

    mullohi='''
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

  nsimd::pack<u64> _1 = nsimd::set1(nsimd::pack<u64>(), (u64)1);
  nsimd::pack<u64> _0 = nsimd::set1(nsimd::pack<u64>(), (u64)0);

  nsimd::pack<u64> carry =
      nsimd::if_else1((tmp3 < tmp1) || (tmp3 < tmp2), _1, _0);

  *low = tmp3 + albl;

  carry = carry + nsimd::if_else1((*low < tmp3) || (*low < albl), _1, _0);

  *high = ahbh + nsimd::shr(albh, 32) + nsimd::shr(ahbl, 32) + carry;
}
#endif
    '''

    def gen_signature(self, nwords, word_size, nrounds):
        return ('nsimd::packx{nwords}<u{word_size}> {fun_name}' \
                '(nsimd::packx{nwords}<u{word_size}> in, ' \
                'nsimd::packx{key_size}<u{word_size}> key)'). \
                format(nwords = nwords, word_size = word_size,
                       fun_name = self.gen_function_name(nwords, word_size,
                                                         nrounds),
                       key_size = self.get_key_size(nwords))

    def get_key_size(self, nwords):
        return int(nwords/2)

    def gen_func(self, opts, nrounds, word_size, nwords):
        if nwords == 2:
            bump_keys_init = \
            'nsimd::pack<u{word_size}> bump = ' \
            'nsimd::set1(nsimd::pack<u{word_size}>(), {bump});'.\
            format(word_size=word_size, bump = '0x9E3779B97F4A7C15ULL' \
                                        if word_size == 64 else '0x9E3779B9U')
            bump_keys = 'key.v0 = key.v0 + bump;'

            round_init = '''
            nsimd::pack<u{word_size}> mul =
                nsimd::set1(nsimd::pack<u{word_size}>(), {mul});
            nsimd::pack<u{word_size}> high, low;'''. \
            format(word_size=word_size, mul='0xD2B74407B1CE6E93ULL' \
                   if word_size == 64 else '0xD256D193U')

            round='''
              mulhilo{word_size}(mul, in.v0, &low, &high);

              in.v0 = high ^ key.v0 ^ in.v1;
              in.v1 = low;
            '''.format(word_size=word_size)

        elif nwords == 4:
            bump_keys_init = '''
            nsimd::pack<u{word_size}> bump0 =
                nsimd::set1(nsimd::pack<u{word_size}>(), {bump0});
            nsimd::pack<u{word_size}> bump1 =
                nsimd::set1(nsimd::pack<u{word_size}>(), {bump1});'''.\
                format(word_size=word_size,
                       bump0 = '0x9E3779B97F4A7C15ULL' \
                       if word_size == 64 else '0x9E3779B9U',
                       bump1 = '0xBB67AE8584CAA73BULL' \
                       if word_size == 64 else '0xBB67AE85U')
            bump_keys = 'key.v0 = key.v0 + bump0;\nkey.v1 = key.v1 + bump1;'

            round_init = '''
            nsimd::pack<u{word_size}> mul0 =
                nsimd::set1(nsimd::pack<u{word_size}>(), {mul0});
            nsimd::pack<u{word_size}> mul1 =
                nsimd::set1(nsimd::pack<u{word_size}>(), {mul1});
            nsimd::pack<u{word_size}> low0, high0, low1, high1;
            '''.format(word_size=word_size,
                       mul0='0xD2E7470EE14C6C93ULL' \
                       if word_size == 64 else '0xD2511F53U',
                       mul1='0xCA5A826395121157ULL' \
                       if word_size == 64 else '0xCD9E8D57U')

            round='''
            mulhilo{word_size}(mul0, in.v0, &low0, &high0);
            mulhilo{word_size}(mul1, in.v2, &low1, &high1);

            in.v0 = high1 ^ key.v0 ^ in.v1;
            in.v1 = low1;
            in.v2 = high0 ^ key.v1 ^ in.v3;
            in.v3 = low0;'''.format(word_size=word_size)


        res = self.gen_signature (nwords, word_size, nrounds)
        res += ' {{ nsimd::packx{}<u{}> out;'.format(nwords, word_size)
        res += bump_keys_init
        res += round_init

        # Round 0:
        res += round;

        for i in range(1, nrounds):
            res += bump_keys
            res += round

        res+='''
            return in;
        }
        '''

        return res

    def generate(self, opts):
        res = self.mullohi

        for word_size, nwords_nrounds in self.wordsize_nwords_nrounds.items():
            for nwords, list_nrounds in nwords_nrounds.items():
                for nrounds in list_nrounds:
                    res += self.gen_func(opts, nrounds, word_size, nwords)

        return res



class ThreeFry(Rand):
    name = 'threefry'

    enums='''
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
    '''

    # Following macros should not be changed to function : gcc can't inline them
    rotations='''
    #define SHIFT_MOD_32(x, N) ((x << (N & 31)) | (x >> ((32 - N) & 31)))
    #define SHIFT_MOD_64(x, N) ((x << (N & 63)) | (x >> ((64 - N) & 63)))
    '''

    undef_macro='''
    #undef SHIFT_MOD_32
    #undef SHIFT_MOD_64
    '''

    wordsize_nwords_nrounds = {32: {2: [12, 20, 32],
                                    4: [12, 20, 72]},
                               64: {2: [13, 20, 32],
                                    4: [12, 20, 72]}}

    def gen_signature(self, nwords, word_size, nrounds):
        return '''nsimd::packx{nwords}<u{word_size}> \
            {fun_name} \
            (nsimd::packx{nwords}<u{word_size}> in, \
            nsimd::packx{nwords}<u{word_size}> key)'''. \
            format(nwords=nwords, word_size = word_size,
                   fun_name=self.gen_function_name(nwords, word_size, nrounds))

    def get_key_size(self, nwords):
        return nwords

    def gen_body(self, opts, nrounds, word_size, nwords):
        if word_size == 32:
            initialize_keys = '''nsimd::pack<u32> ks{nwords} =
                nsimd::set1(nsimd::pack<u32>(), 0x1BD11BDAU);'''. \
                        format(nwords=nwords)
        elif word_size == 64:
            initialize_keys = '''nsimd::pack<u64> ks{nwords} =
                nsimd::set1(nsimd::pack<u64>(), 0x1BD11BDAA9FC1A22ULL);'''. \
                format(nwords=nwords)

        res = self.gen_signature(nwords, word_size, nrounds)

        res += ' {{ nsimd::packx{}<u{}> out;'.format(nwords, word_size)

        res += initialize_keys

        initialisation_keys = '''
        nsimd::pack<u{word_size}> ks{i};
        ks{i} = key.v{i};
        out.v{i} = in.v{i};
        ks{nwords} = ks{nwords} ^ key.v{i};
        out.v{i} = out.v{i} + key.v{i};
        '''

        for i in range(0,nwords):
            res += initialisation_keys.format(i=i, nwords=nwords,
                                              word_size=word_size)

        for i in range(0, nrounds):
            if nwords == 4:
                indexes= [1 if i%2==0 else 3, 1 if i%2==1 else 3]

                res += '''
                out.v0 = out.v0 + out.v{index0};
                out.v{index0} = SHIFT_MOD_{word_size}(out.v{index0},
                                           Rot_{word_size}x{nwords}_{i_mod}_0);
                out.v{index0} = out.v{index0} ^ out.v0;
                out.v2 = out.v2 + out.v{index1};
                out.v{index1} = SHIFT_MOD_{word_size}(out.v{index1},
                                         Rot_{word_size}x{nwords}_{i_mod}_2);
                out.v{index1} = out.v{index1} ^ out.v2;
                '''.format(index0=indexes[0], index1=indexes[1], i_mod=i%8,
                        word_size=word_size, nwords=nwords)
            elif nwords == 2:
                res += '''
                out.v0 = out.v0 + out.v1;
                out.v1 = SHIFT_MOD_{word_size}(out.v1,
                             Rot_{word_size}x{nwords}_{i_mod});
                out.v1 = out.v1 ^ out.v0;'''. \
                format(i_mod=i % 8, word_size=word_size, nwords=nwords)

            #if (i % nwords) == nwords - 1:
            if (i % 4) == 3:
                d = int(i / 4 + 1)
                res += '\n'
                for j in range(0, nwords):
                    res += 'out.v{j} = out.v{j} + ks{calc};\n'. \
                            format(j=j, calc=str(int((d+j)%(nwords+1))))

                res += 'out.v{n} = out.v{n} + ' \
                       'nsimd::pack<u{word_size}>({d});\n'. \
                       format(d=d, n=nwords-1, word_size=word_size)

        res+='''
            return out;
        }
        '''

        return res

    def generate(self, opts):
        res = ''
        res += self.enums
        res += self.rotations

        for word_size, nwords_nrounds in self.wordsize_nwords_nrounds.items():
            for nwords, list_nrounds in nwords_nrounds.items():
                for nrounds in list_nrounds:
                    res += self.gen_body(opts, nrounds, word_size, nwords)

        res += self.undef_macro

        return res

def gen_functions(opts):
    ## Write source files
    #dirname = os.path.join(opts.include_dir, 'modules', 'random')
    #common.mkdir_p(dirname)
    #filename = os.path.join(dirname, 'functions.cpp')
    #print(filename)
    #with common.open_utf8(opts, filename) as out:
    #    out.write('#include "functions.hpp"\n')
    #    out.write('{}\n\n'.format(common.hbar))
    #    out.write(gen(opts))
    #    out.write('#endif\n')
    #common.clang_format(opts, filename)

    # Write headers
    dirname = os.path.join(opts.include_dir, 'modules', 'random')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, 'functions.hpp')
    with common.open_utf8(opts, filename) as out:
        out.write('#ifndef NSIMD_MODULES_RANDOM_FUNCTIONS_HPP\n')
        out.write('#define NSIMD_MODULES_RANDOM_FUNCTIONS_HPP\n\n')

        out.write('#include <nsimd/nsimd.h>\n')
        out.write('#include <nsimd/cxx_adv_api.hpp>\n')
        out.write('#include <nsimd/cxx_adv_api_functions.hpp>\n')

        out.write('namespace nsimd {\n')
        out.write('namespace random {\n')

        out.write('{}\n\n'.format(common.hbar))
        for func in rand_functions:
            out.write(func.gen_headers(opts))
            out.write(func.generate(opts))

        out.write('} // namespace nsimd\n')
        out.write('} // namespace random\n')
        out.write('#endif\n')
    common.clang_format(opts, filename)

def gen_tests(opts):
    for func in rand_functions:
        for word_size, nwords_nrounds in func.wordsize_nwords_nrounds.items():
            for nwords, list_nrounds in nwords_nrounds.items():
                for nrounds in list_nrounds:
                    # Write headers
                    dirname = os.path.join(opts.tests_dir, 'modules', 'random')
                    common.mkdir_p(dirname)
                    filename = os.path.join(dirname, '{}.cpp'. \
                               format(func.gen_function_name(nwords, word_size,
                                                             nrounds)))
                    with common.open_utf8(opts, filename) as out:
                        out.write(func.gen_tests(opts, nrounds, word_size,
                                  nwords))

                    common.clang_format(opts, filename)


# -----------------------------------------------------------------------------

def name():
    return 'Random number generators'

def desc():
    return \
    'This module define functions that generate pseudorandom numbers using' \
    'algorithms described in Parallel Random Numbers: As Easy as 1,2,3, by' \
    'John K. Salmon, Mark A. Moraes, Ron O. Dror and David E.Shaw.'

def gen_doc(opts):
    api =  ''
    for func in rand_functions:
        for word_size, nwords_nrounds in func.wordsize_nwords_nrounds.items():
            for nwords, list_nrounds in nwords_nrounds.items():
                for nrounds in list_nrounds:
                    api += '- `' + func.gen_signature(nwords, word_size,
                                                      nrounds) + '`;  \n'
                    api += '  Returns a random number using the ' \
                           '{func_name} generator\n\n'. \
                           format(func_name=func.name)

    res = '''
# NSIMD Random module overview

{desc}

Two different algorithms are proposed : threefry and philox. Both should give
high quality random number.
Threefry is quicker on CPU, while philox is best used on GPU.

Both algorithms are counter based pseudorandom number generator, meaning that
they need two parameters:
- a key, each key will generate an unique sequence,
- a counter, which will give the different numbers in the sequence.

# NSIMD Random API reference

{api}
'''.format(desc = desc(), api=api)


    filename = common.get_markdown_file(opts, 'overview', 'random')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write(res)

def doc_menu():
    return dict()

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating module random')

    if opts.library:
        gen_functions(opts)
    if opts.tests:
        gen_tests(opts)
    if opts.doc:
        gen_doc(opts)
