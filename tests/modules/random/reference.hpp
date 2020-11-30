/*
# License (BSD)

Copyright (c) 2017, Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for
 Los Alamos National Laboratory (LANL), which is operated by Triad National
Security, LLC for the U.S. Department of Energy/National Nuclear Security
Administration.

All rights in the program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to
reproduce, prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.

This is open source software; you can redistribute it and/or modify it under
the terms of the BSD-3 Clause License. If software is modified to produce
derivative works, such modified software should be clearly marked, so as not to
confuse it with the version available from LANL. Full text of the BSD-3 License
can be found in the License file in the main development branch of the
repository.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

- Neither the name of Triad National Security, LLC, Los Alamos
  National Laboratory, LANL, the U.S. Government, nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY TRIAD NATIONAL SECURITY, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL TRIAD NATIONAL SECURITY, LLC OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <nsimd/nsimd.h>
#include <assert.h>
#include <stdint.h>

/* ------------------------------------------------------------------------- */
/* Structs */
/* ------------------------------------------------------------------------- */

typedef struct {
  u64 v[4];
} tab64x4_t;

typedef struct {
  u64 v[2];
} tab64x2_t;

typedef struct {
  u64 v[1];
} tab64x1_t;

typedef struct {
  u32 v[4];
} tab32x4_t;

typedef struct {
  u32 v[2];
} tab32x2_t;

typedef struct {
  u32 v[1];
} tab32x1_t;


/* ------------------------------------------------------------------------- */
/* Philox 64 bits */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* Helpers */

NSIMD_INLINE u64 mulhilo64(u64 a, u64 b, u64 *hip) {
  /*
                           a1      a0
                        x  b1      b0
                        -------------
                     a1b0+r00    a0b0
          +  a1b1        b1a0
          ---------------------------

  */
  u64 a0 = a & 0xFFFFFFFF;
  u64 a1 = a >> 32;
  u64 b0 = b & 0xFFFFFFFF;
  u64 b1 = b >> 32;
  *hip = (a1 * b1) + ((b0 * a1 + a0 * b1 + ((a0 * b0) >> 32)) >> 32);
  return a * b;
}

/* ------------------------------------------------------------------------- */
/* Philox 64x4 */

NSIMD_INLINE tab64x2_t _philox4x64bumpkey(tab64x2_t key) {
  key.v[0] += 0x9E3779B97F4A7C15ULL;
  key.v[1] += 0xBB67AE8584CAA73BULL;
  return key;
}

NSIMD_INLINE tab64x4_t _philox4x64round(tab64x4_t ctr, tab64x2_t key) {
  u64 hi0;
  u64 hi1;
  u64 lo0 = mulhilo64(0xD2E7470EE14C6C93ULL, ctr.v[0], &hi0);
  u64 lo1 = mulhilo64(0xCA5A826395121157ULL, ctr.v[2], &hi1);
  tab64x4_t out = {
      {hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
  return out;
}

tab64x4_t branson_philox4x64_R(unsigned int R, tab64x4_t ctr, tab64x2_t key) {

  if (R > 0) {
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 1) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 2) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 3) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 4) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 5) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 6) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 7) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 8) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 9) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 10) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 11) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 12) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 13) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 14) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 15) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  return ctr;
}

/* ------------------------------------------------------------------------- */
/* Philox 64x2 */

NSIMD_INLINE tab64x1_t _philox2x64bumpkey(tab64x1_t key) {
  key.v[0] += 0x9E3779B97F4A7C15ULL;
  return key;
}

NSIMD_INLINE tab64x2_t _philox2x64round(tab64x2_t ctr, tab64x1_t key) {
  uint64_t hi;
  uint64_t lo = mulhilo64(0xD2B74407B1CE6E93ULL, ctr.v[0], &hi);
  tab64x2_t out = {{hi ^ key.v[0] ^ ctr.v[1], lo}};
  return out;
}

tab64x2_t branson_philox2x64_R(unsigned int R, tab64x2_t ctr, tab64x1_t key) {
  if (R > 0) {
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 1) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 2) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 3) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 4) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 5) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 6) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 7) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 8) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 9) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 10) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 11) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 12) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 13) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 14) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  if (R > 15) {
    key = _philox2x64bumpkey(key);
    ctr = _philox2x64round(ctr, key);
  }
  return ctr;
}

/* ------------------------------------------------------------------------- */
/* Philox 32 bits */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* Helpers */

NSIMD_INLINE u32 mulhilo32(u32 a, u32 b, u32 *hip) {
  u64 product = ((u64)a) * ((u64)b);
  *hip = (u32)(product >> 32);
  return (u32)product;
}

/* ------------------------------------------------------------------------- */
/* Philox 32x4 */

NSIMD_INLINE tab32x2_t _philox4x32bumpkey(tab32x2_t key) {
  key.v[0] += ((uint32_t)0x9E3779B9);
  key.v[1] += ((uint32_t)0xBB67AE85);
  return key;
}

NSIMD_INLINE tab32x4_t _philox4x32round(tab32x4_t ctr, tab32x2_t key) {
  uint32_t hi0;
  uint32_t hi1;
  uint32_t lo0 = mulhilo32(((uint32_t)0xD2511F53), ctr.v[0], &hi0);
  uint32_t lo1 = mulhilo32(((uint32_t)0xCD9E8D57), ctr.v[2], &hi1);
  tab32x4_t out = {
      {hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
  return out;
}

tab32x4_t branson_philox4x32_R(unsigned int R, tab32x4_t ctr, tab32x2_t key) {
  assert(R <= 16);
  if (R > 0) {
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 1) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 2) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 3) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 4) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 5) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 6) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 7) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 8) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 9) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 10) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 11) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 12) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 13) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 14) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 15) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  return ctr;
}

/* ------------------------------------------------------------------------- */
/* Philox 32x2 */

NSIMD_INLINE tab32x1_t _philox2x32bumpkey(tab32x1_t key) {
  key.v[0] += ((uint32_t)0x9E3779B9);
  return key;
}

NSIMD_INLINE tab32x2_t _philox2x32round(tab32x2_t ctr, tab32x1_t key) {
  uint32_t hi;
  uint32_t lo = mulhilo32(((uint32_t)0xd256d193), ctr.v[0], &hi);
  tab32x2_t out = {{hi ^ key.v[0] ^ ctr.v[1], lo}};
  return out;
}

tab32x2_t branson_philox2x32_R(unsigned int R, tab32x2_t ctr, tab32x1_t key) {
  assert(R <= 16);
  if (R > 0) {
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 1) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 2) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 3) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 4) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 5) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 6) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 7) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 8) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 9) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 10) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 11) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 12) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 13) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 14) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  if (R > 15) {
    key = _philox2x32bumpkey(key);
    ctr = _philox2x32round(ctr, key);
  }
  return ctr;
}

/* ------------------------------------------------------------------------- */
/* Threefry 64 bits */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* Helpers */

#define RotL_64(x, N) ((x << (N & 63)) | (x >> ((64 - N) & 63)))

/* ------------------------------------------------------------------------- */
/* Threefry 64x4 */
enum r123_enum_threefry64x4 {
  R_64x4_0_0 = 14,
  R_64x4_0_1 = 16,
  R_64x4_1_0 = 52,
  R_64x4_1_1 = 57,
  R_64x4_2_0 = 23,
  R_64x4_2_1 = 40,
  R_64x4_3_0 = 5,
  R_64x4_3_1 = 37,
  R_64x4_4_0 = 25,
  R_64x4_4_1 = 33,
  R_64x4_5_0 = 46,
  R_64x4_5_1 = 12,
  R_64x4_6_0 = 58,
  R_64x4_6_1 = 22,
  R_64x4_7_0 = 32,
  R_64x4_7_1 = 32
};

tab64x4_t branson_threefry4x64_R(unsigned int Nrounds,
                                 tab64x4_t in,
                                 tab64x4_t k) {
  tab64x4_t X;
  uint64_t ks[4 + 1];
  int i;
  ks[4] = ((0xA9FC1A22) + (((uint64_t)(0x1BD11BDA)) << 32));
  for (i = 0; i < 4; i++) {
    ks[i] = k.v[i];
    X.v[i] = in.v[i];
    ks[4] ^= k.v[i];
  }
  X.v[0] += ks[0];
  X.v[1] += ks[1];
  X.v[2] += ks[2];
  X.v[3] += ks[3];
  if (Nrounds > 0) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 1) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 2) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 1;
  }
  if (Nrounds > 4) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 5) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 6) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 2;
  }
  if (Nrounds > 8) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 9) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 10) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 3;
  }
  if (Nrounds > 12) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 13) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 14) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 4;
  }
  if (Nrounds > 16) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 17) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 18) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 5;
  }
  if (Nrounds > 20) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 21) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 22) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 23) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 23) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 6;
  }
  if (Nrounds > 24) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 25) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 26) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 27) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 27) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 7;
  }
  if (Nrounds > 28) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 29) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 30) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 31) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 31) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 8;
  }
  if (Nrounds > 32) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 33) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 34) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 35) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 35) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 9;
  }
  if (Nrounds > 36) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 37) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 38) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 39) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 39) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 10;
  }
  if (Nrounds > 40) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 41) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 42) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 43) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 43) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 11;
  }
  if (Nrounds > 44) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 45) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 46) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 47) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 47) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 12;
  }
  if (Nrounds > 48) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 49) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 50) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 51) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 51) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 13;
  }
  if (Nrounds > 52) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 53) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 54) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 55) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 55) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 14;
  }
  if (Nrounds > 56) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 57) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 58) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 59) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 59) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 15;
  }
  if (Nrounds > 60) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 61) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 62) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 63) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 63) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 16;
  }
  if (Nrounds > 64) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 65) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 66) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 67) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 67) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 17;
  }
  if (Nrounds > 68) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 69) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 70) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 71) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 71) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 18;
  }
  return X;
}

/* ------------------------------------------------------------------------- */
/* Threefry 64x2 */
enum r123_enum_threefry64x2 {

  R_64x2_0_0 = 16,
  R_64x2_1_0 = 42,
  R_64x2_2_0 = 12,
  R_64x2_3_0 = 31,
  R_64x2_4_0 = 16,
  R_64x2_5_0 = 32,
  R_64x2_6_0 = 24,
  R_64x2_7_0 = 21
};

tab64x2_t branson_threefry2x64_R(unsigned int Nrounds,
                                 tab64x2_t in,
                                 tab64x2_t k) {
  tab64x2_t X;
  uint64_t ks[2 + 1];
  int i;
  ks[2] = ((0xA9FC1A22) + (((uint64_t)(0x1BD11BDA)) << 32));
  for (i = 0; i < 2; i++) {
    ks[i] = k.v[i];
    X.v[i] = in.v[i];
    ks[2] ^= k.v[i];
  }
  X.v[0] += ks[0];
  X.v[1] += ks[1];
  if (Nrounds > 0) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 1) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 2) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 3) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 3) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[1] += 1;
  }
  if (Nrounds > 4) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 5) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 6) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 7) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 7) {
    X.v[0] += ks[2];
    X.v[1] += ks[0];
    X.v[1] += 2;
  }
  if (Nrounds > 8) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 9) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 10) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 11) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 11) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[1] += 3;
  }
  if (Nrounds > 12) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 13) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 14) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 15) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 15) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[1] += 4;
  }
  if (Nrounds > 16) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 17) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 18) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 19) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 19) {
    X.v[0] += ks[2];
    X.v[1] += ks[0];
    X.v[1] += 5;
  }
  if (Nrounds > 20) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 21) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 22) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 23) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 23) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[1] += 6;
  }
  if (Nrounds > 24) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 25) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 26) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 27) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 27) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[1] += 7;
  }
  if (Nrounds > 28) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 29) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 30) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 31) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 31) {
    X.v[0] += ks[2];
    X.v[1] += ks[0];
    X.v[1] += 8;
  }
  return X;
}

/* ------------------------------------------------------------------------- */
/* Threefry 32 bits */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* helper */
#define RotL_32(x, N) ((x << (N & 31)) | (x >> ((32 - N) & 31)))

/* ------------------------------------------------------------------------- */
/* Threefry 32x4 */
enum r123_enum_threefry32x4 {
  R_32x4_0_0 = 10,
  R_32x4_0_1 = 26,
  R_32x4_1_0 = 11,
  R_32x4_1_1 = 21,
  R_32x4_2_0 = 13,
  R_32x4_2_1 = 27,
  R_32x4_3_0 = 23,
  R_32x4_3_1 = 5,
  R_32x4_4_0 = 6,
  R_32x4_4_1 = 20,
  R_32x4_5_0 = 17,
  R_32x4_5_1 = 11,
  R_32x4_6_0 = 25,
  R_32x4_6_1 = 10,
  R_32x4_7_0 = 18,
  R_32x4_7_1 = 20
};

tab32x4_t branson_threefry4x32_R(unsigned int Nrounds,
                                 tab32x4_t in,
                                 tab32x4_t k) {
  tab32x4_t X;
  uint32_t ks[4 + 1];
  int i;

  ks[4] = 0x1BD11BDA;
  for (i = 0; i < 4; i++) {
    ks[i] = k.v[i];
    X.v[i] = in.v[i];
    ks[4] ^= k.v[i];
  }
  X.v[0] += ks[0];
  X.v[1] += ks[1];
  X.v[2] += ks[2];
  X.v[3] += ks[3];
  if (Nrounds > 0) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 1) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 2) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 1;
  }
  if (Nrounds > 4) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 5) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 6) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 2;
  }
  if (Nrounds > 8) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 9) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 10) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 3;
  }
  if (Nrounds > 12) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 13) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 14) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 4;
  }
  if (Nrounds > 16) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 17) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 18) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 5;
  }
  if (Nrounds > 20) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 21) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 22) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 23) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 23) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 6;
  }
  if (Nrounds > 24) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 25) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 26) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 27) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 27) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 7;
  }
  if (Nrounds > 28) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 29) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 30) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 31) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 31) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 8;
  }
  if (Nrounds > 32) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 33) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 34) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 35) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 35) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 9;
  }
  if (Nrounds > 36) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 37) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 38) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 39) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 39) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 10;
  }
  if (Nrounds > 40) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 41) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 42) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 43) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 43) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 11;
  }
  if (Nrounds > 44) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 45) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 46) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 47) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 47) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 12;
  }
  if (Nrounds > 48) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 49) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 50) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 51) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 51) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 13;
  }
  if (Nrounds > 52) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 53) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 54) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 55) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 55) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 14;
  }
  if (Nrounds > 56) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 57) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 58) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 59) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 59) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 15;
  }
  if (Nrounds > 60) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 61) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 62) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 63) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 63) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 16;
  }
  if (Nrounds > 64) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 65) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 66) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 67) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 67) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 17;
  }
  if (Nrounds > 68) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 69) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 70) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 71) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 71) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 18;
  }
  return X;
}

/* ------------------------------------------------------------------------- */
/* Threefry 32x2 */
enum r123_enum_threefry32x2 {
  R_32x2_0_0 = 13,
  R_32x2_1_0 = 15,
  R_32x2_2_0 = 26,
  R_32x2_3_0 = 6,
  R_32x2_4_0 = 17,
  R_32x2_5_0 = 29,
  R_32x2_6_0 = 16,
  R_32x2_7_0 = 24
};

tab32x2_t branson_threefry2x32_R(unsigned int Nrounds,
                                 tab32x2_t in,
                                 tab32x2_t k) {
  tab32x2_t X;
  uint32_t ks[2 + 1];
  int i;
  ks[2] = 0x1BD11BDA;
  for (i = 0; i < 2; i++) {
    ks[i] = k.v[i];
    X.v[i] = in.v[i];
    ks[2] ^= k.v[i];
  }
  X.v[0] += ks[0];
  X.v[1] += ks[1];
  if (Nrounds > 0) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 1) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 2) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 3) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 3) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[1] += 1;
  }
  if (Nrounds > 4) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 5) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 6) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 7) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 7) {
    X.v[0] += ks[2];
    X.v[1] += ks[0];
    X.v[1] += 2;
  }
  if (Nrounds > 8) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 9) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 10) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 11) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 11) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[1] += 3;
  }
  if (Nrounds > 12) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 13) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 14) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 15) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 15) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[1] += 4;
  }
  if (Nrounds > 16) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 17) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 18) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 19) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 19) {
    X.v[0] += ks[2];
    X.v[1] += ks[0];
    X.v[1] += 5;
  }
  if (Nrounds > 20) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 21) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 22) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 23) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 23) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[1] += 6;
  }
  if (Nrounds > 24) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_0_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 25) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_1_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 26) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_2_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 27) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_3_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 27) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[1] += 7;
  }
  if (Nrounds > 28) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_4_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 29) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_5_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 30) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_6_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 31) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x2_7_0);
    X.v[1] ^= X.v[0];
  }
  if (Nrounds > 31) {
    X.v[0] += ks[2];
    X.v[1] += ks[0];
    X.v[1] += 8;
  }
  return X;
}
