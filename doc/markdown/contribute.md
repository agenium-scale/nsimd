<!--

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

-->

## How to Contribute to `nsimd`?

You are welcome to contribute to `nsimd`. This document gives some details on
how to add/wrap new intrinsics. When you have finished fixing some bugs or
adding some new features, please make a pull request. One of our repository
maintainer will then merge or comment the pull request.


##  Prerequisites

- Respect the philosophy of the library (see [index](index.md).)
- Basic knowledge of Python 3.
- Good knowledge of C.
- Good knowledge of C++.
- Good knowledge of SIMD programming.


## How Do I Add Support for a New Intrinsic?

### Introduction

`nsimd` currently supports the following architectures:
- `CPU`:
  + `CPU` called `CPU` in source code. This "extension" is not really one as it
    is only present so that code written with `nsimd` can compile and run on
    targets not supported by `nsimd` or with no SIMD.
- Intel:
  + `SSE2` called `SSE2` in source code.
  + `SSE4.2` called `SSE42` in source code.
  + `AVX` called `AVX` in source code.
  + `AVX2` called `AVX2` in source code.
  + `AVX-512` as found on KNLs called `AVX512_KNL` in source code.
  + `AVX-512` as found on Xeon Skylake CPUs called `AVX512_SKYLAKE` in source code.
- Arm
  + `NEON` 128 bits as found on ARMv7 CPUs called `NEON128` in source code.
  + `NEON` 128 bits as found on Aarch64 CPUs called `AARCH64` in source code.
  + `SVE` called `SVE` in source code.

`nsimd` currently supports the following types:
- `i8`: signed integers over 8 bits (usually `signed char`),
- `u8`: unsigned integers over 8 bits (usually `unsigned char`),
- `i16`: signed integers over 16 bits (usually `short`),
- `u16`: unsigned integers over 16 bits (usually `unsigned short`),
- `i32`: signed integers over 32 bits (usually `int`),
- `u32`: unsigned integers over 32 bits (usually `unsigned int`),
- `i64`: signed integers over 64 bits (usually `long`),
- `u64`: unsigned integers over 64 bits (usually `unsigned long`),
- `f16`: floating point numbers over 16 bits in IEEE format called `float16`
  in the rest of this document
  (<https://en.wikipedia.org/wiki/Half-precision_floating-point_format>),
- `f32`: floating point numbers over 32 bits (usually `float`)
- `f64`: floating point numbers over 64 bits (usually `double`),

As C and C++ do not support `float16`, `nsimd` provides its own types to handle
them. Therefore special care has to be taken when implementing
intrinsics/operators on architecures that do not natively supports them.

We will make the following misuse of language in the rest of this document.
The type taken by intrinsics is of course a SIMD vector and more precisely a
SIMD vector of chars or a SIMD vector of `short`s or a SIMD vector of `int`s…
Therefore when we will talk about an intrinsic, we will say that it takes
type `T` as arguments when it takes in fact a SIMD vector of `T`.

### Our imaginary intrinsic

We will add support to the library for the following imaginary intrinsic: given
a SIMD vector, suppose that this intrisic called `foo` takes each element `x`
of the vector and compute `1 / (1 - x) + 1 / (1 - x)^2`. Moreover suppose that
hardware vendors all propose this intrisic only for floatting point numbers as
follows:
- CPU (no intrinsics is given of course in standard C and C++)
- Intel (no intrinsics is given for `float16`s)
  + `SSE2`: no intrinsics is provided.
  + `SSE42`: `_mm_foo_ps` for `float`s and `_mm_foo_pd` for `double`s.
  + `AVX`: no intrinsics is provided.
  + `AVX2`: `_mm256_foo_ps` for `float`s and `_mm256_foo_pd` for `double`s.
  + `AVX512_KNL`: no intrinsics is provided.
  + `AVX512_SKYLAKE`: `_mm512_foo_ps` for `float`s and `_mm512_foo_pd` for
    `double`s.
- ARM
  + `NEON128`: `vfooq_f16` for `float16`s, `vfooq_f32` for `float`s and no
    intrinsics for `double`s.
  + `AARCH64`: same as `NEON128` but `vfooq_f64` for doubles.
  + `SVE`: `svfoo_f16`, `svfoo_f32` and `svfoo_f64` for respectively `float16`s,
    `float`s and `double`s.

First thing to do is to declare this new intrinsic to the generation system.
A lot of work is done by the generation system such as generating all functions
signatures for C and C++ APIs, tests, benchmarks and documentation. Of course
the default documentation does not say much but you can add a better
description.

### Registering the intrinsic (or operator)

A function or an intrinsic is called an operator in the generation system.
Go at the bottom of `egg/operators.py` and add the following just after
the `Rsqrt11` class.

```python
class Foo(Operator):
    full_name = 'foo'
    signature = 'v foo v'
    types = common.ftypes
    domain = Domain('R\{1}')
    categories = [DocBasicArithmetic]
```

This little class will be processed by the generation system so that operator
`foo` will be available for the end-user of the library in both C and C++ APIs.
Each member of this class controls how the generation is be done:
- `full_name` is a string containing the human readable name of the operator.
  If not given, the class name will be taken for it.
- `signature` is a string describing what kind of arguments and how many takes
  the operator. This member is mandatory and must respect the following syntax:
  `return_type name_of_operator arg1_type arg2_type ...` where `return_type`
  and the `arg*_type` can be taken from the following list:
  + `v   ` SIMD vector parameter
  + `vx2 ` Structure of 2 SIMD vector parameters
  + `vx3 ` Structure of 3 SIMD vector parameters
  + `vx4 ` Structure of 4 SIMD vector parameters
  + `l   ` SIMD vector of logicals parameter
  + `s   ` Scalar parameter
  + `*   ` Pointer to scalar parameter
  + `c*  ` Pointer to const scalar parameter
  + `_   ` void (only for return type)
  + `p   ` Parameter (integer)

In our case `v foo v` means that `foo` takes one SIMD vector as argument and
returns a SIMD vector as output. Several signatures will be generated for this
intrinsic according to the types it can supports. In our case the intrinsic
only support floatting point types.
- `types` is a Python list indicating which types are supported by the
  intrinsic. If not given, the intrinsic is supposed to support all types.
  Some Python lists are predefined to help the programmer:
  + `ftypes = ['f64', 'f32', 'f16']       ` All floatting point types
  + `ftypes_no_f16 = ['f64', 'f32']       `
  + `itypes = ['i64', 'i32', 'i16', 'i8'] ` All signed integer types
  + `utypes = ['u64', 'u32', 'u16', 'u8'] ` All unsigned integer types
  + `iutypes = itypes + utypes`
  + `types = ftypes + iutypes`
- `domain` is a string indicating the mathematical domain of definition of the
  operator. This helps for benchmarks and tests for generating random numbers
  as inputs in the correct interval. In our case `R\{1}` means all real numbers
  (of course all floating point numbers) expect `-1` for which the operator
  cannot be computed. For examples see how other operators are defined in
  `egg/operators.py`.
- `categories` is a list of Python classes that indicates the generation
  system to which categories `foo` belongs. The list of available categories
  is as follow:
  + `DocShuffle          ` for Shuffle functions
  + `DocTrigo            ` for Trigonometric functions
  + `DocHyper            ` for Hyperbolic functions
  + `DocExpLog           ` for Exponential and logarithmic functions
  + `DocBasicArithmetic  ` for Basic arithmetic operators
  + `DocBitsOperators    ` for Bits manipulation operators
  + `DocLogicalOperators ` for Logicals operators
  + `DocMisc             ` for Miscellaneous
  + `DocLoadStore        ` for Loads & stores
  + `DocComparison       ` for Comparison operators
  + `DocRounding         ` for Rounding functions
  + `DocConversion       ` for Conversion operators
  If no category corresponds to the operator you want to add to `nsimd` then feel
  free to create a new category (see the bottom of this document)

Many other members are supported by the generation system. We describe them
quickly here and will give more details in a later version of this document.
Default values are given in square brakets:
- `cxx_operator [= None]` in case the operator has a corresponding C++ operator.
- `autogen_cxx_adv [= True]` in case the C++ advanced API signatures for this
  operator must not be auto-generated.
- `output_to [= common.OUTPUT_TO_SAME_TYPE]` in case the operator output type
  differs from its input type. Possible values are:
  + `OUTPUT_TO_SAME_TYPE`: output is of same type as input.
  + `OUTPUT_TO_SAME_SIZE_TYPES`: output can be any type of same bit size.
  + `OUTPUT_TO_UP_TYPES`: output can be any type of bit size twice the bit
    bit size of the input. In this case the input type will never be a 64-bits
    type.
  + `OUTPUT_TO_DOWN_TYPES`: output can be any type of bit size half the bit
    bit size of the input. In this case the input type will never be a 8-bits
    type.
- `src [= False]` in case the code must be compiled in the library.
- `load_store [= False]` in case the operator loads/store data from/to
  memory.
- `do_bench [= True]` in case benchmarks for the operator must not be
  auto-generated.
- `desc [= '']` description (in Markdown format) that will appear in the
  documentation for the operator.
- `bench_auto_against_cpu [= True]` for auto-generation of benchmark against
  `nsimd` CPU implementation.
- `bench_auto_against_mipp [= False]` for auto-generation of benchmark against
  the MIPP library.
- `bench_auto_against_sleef [= False]` for auto-generation of benchmark against
  the Sleef library.
- `bench_auto_against_std [= False]` for auto-generation of benchmark against
  the standard library.
- `tests_mpfr [= False]` in case the operator has an MPFR counterpart for
  comparison, then test the correctness of the operator against it.
- `tests_ulps [= False]` in case the auto-generated tests has to compare ULPs
  (<https://en.wikipedia.org/wiki/Unit_in_the_last_place>).

### Implementing the operator

Now that the operator is registered, all signatures will be generated but
the implemenatations will be missing. Type
```sh
python3 egg/hatch.py -Af
```

and the following files (among many other) should appear:
- `include/nsimd/cpu/cpu/foo.h`
- `include/nsimd/x86/sse2/foo.h`
- `include/nsimd/x86/sse42/foo.h`
- `include/nsimd/x86/avx/foo.h`
- `include/nsimd/x86/avx2/foo.h`
- `include/nsimd/x86/avx512_knl/foo.h`
- `include/nsimd/x86/avx512_skylake/foo.h`
- `include/nsimd/arm/neon128/foo.h`
- `include/nsimd/arm/aarch64/foo.h`
- `include/nsimd/arm/sve/foo.h`

They each correspond to the implementations of the operator for each supported
architectures. When openening one of these files the implementations in plain
C and then in C++ (falling back to the C function) should be there but all the
C implementations are reduced to `abort();`. This is the default when none is
provided. Note that the "cpu" architecture is just a fallback involving no
SIMD at all. This is used on architectures not supported by `nsimd` or when the
architectures does not offer any SIMD.

Providing implementations for `foo` is done by completing the following Python
files:
- `egg/platform_cpu.py`
- `egg/platform_x86.py`
- `egg/platform_arm.py`

The idea is to produce plain C (not C++) code using Python string format. Each
`platform_*.py` file provides some helper functions to ease as much as
possible the programmer's job. But every `platform_*.py` file provides the
same "global" variables available in every functions and is designed in the
same way:
1. At the bottom of the file is the `get_impl` function taking the following
   arguments:
   + `func     ` the name of the operator the system is currently
     auto-generating.
   + `simd_ext ` the SIMD extension for which the system wants the
     implemetation.
   + `from_typ ` the input type of the argument that will be passed to the
     operator.
   + `to_typ   ` the output type produced by the operator.
2. Inside this function lies a Python dictionary that provides functions
   implementing each operator. The string containing the C code for the
   implementations can be put here directly but usually the string is
   returned by a Python function that is written above in the same file.
3. At the top of the file lies helper functions that helps generating code.
   This is specific to each architecture. Do not hesitate to look at it.

Let's begin by the `cpu` implementations. It turns out that there is no SIMD
extension in this case, and by convention, `simd_ext == 'cpu'` and this
argument can therefore be ignored. So we first add an entry to the `impls`
Python dictionary of the `get_impl` function:
```python
    impls = {

        ...

        'reverse': reverse1(from_typ),
        'addv': addv(from_typ),
        'foo': foo1(from_typ) # Added at the bottom of the dictionary
    }
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

    ...
```

Then, above in the file we write the Python function `foo1` that will provide
the C implementation of operator `foo`:
```python
def foo1(typ):
    return func_body(
           '''ret.v{{i}} = ({typ})1 / (({typ})1 - {in0}.v{{i}}) +
                           ({typ})1 / ((({typ})1 - {in0}.v{{i}}) *
                                       (({typ})1 - {in0}.v{{i}}));'''. \
                                       format(**fmtspec), typ)
```

First note that the arguments names passed to the operator in its C
implementation are not known in the Python side. Several other parameters
are not known or are cumbersome to find out. Therefore each function has access
to the `fmtspec` Python dictionary that hold some of these values:
- `in0`: name of the first parameter for the C implementation.
- `in1`: name of the second parameter for the C implementation.
- `in2`: name of the third parameter for the C implementation.
- `simd_ext`: name of the SIMD extension (for the cpu architecture, this is
  equal to `"cpu"`).
- `from_typ`: type of the input.
- `to_typ`: type of the output.
- `typ`: equals `from_typ`, shorter to write as usually `from_typ == to_typ`.
- `utyp`: bitfield type of the same size of `typ`.
- `typnbits`: number of bits in `typ`.

The CPU extension can emulate 64-bits or 128-bits wide SIMD vectors. Each type
is a struct containing as much members as necessary so that `sizeof(T) *
(number of members) == 64 or 128`. In order to avoid the developper to write
two cases (64-bits wide and 128-bits wide) the `func_body` function is provided
as a helper. Note that the index `{{i}}` is in double curly brackets to go
through two Python string formats:
1. The first pass is done within the `foo1` Python function and replaces
   `{typ}` and `{in0}`. In this pass `{{i}}` is formatted into `{i}`.
2. The second pass is done by the `func_body` function which unrolls the string
   to the necessary number and replace `{i}` by the corresponding number. The
   produced C code will look like one would written the same statement for each
   members of the input struct.

Then note that as plain C (and C++) does not support native 16-bits wide
floating point types `nsimd` emulates it with a C struct containing 4 floats
(32-bits swide floatting point numbers). In some cases extra care has to be
taken to handle this type.

For each SIMD extension one can find a `types.h` file (for `cpu` the files can
be found in `include/nsimd/cpu/cpu/types.h`) that declares all SIMD types. If
you have any doubt on a given type do not hesitate to take a look at this file.
Note also that this file is auto-generated and is therefore readable only after
a successfull first `python3 egg/hatch -Af`.

Now that the `cpu` implementation is written, you should be able to write the
implementation of `foo` for other architectures. Each architecture has its
particularities. We will cover them now by providing directly the Python
implementations and explaining in less details.

Finally note that `clang-format` is called by the generation system to
autoformat produced C/C++ code. Therefore prefer indenting C code strings within
the Python according to Python indentations, do not write C code beginning at
column 0 in Python files.

### For Intel

```python
def foo1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v1 = {pre}foo_ps({in0}.v1);
                  ret.v2 = {pre}foo_ps({in0}.v2);
                  return ret;'''.format(**fmtspec)
    if simd_ext == 'sse2':
        return emulate_op1('foo', 'sse2', typ)
    if simd_ext in ['avx', 'avx512_knl']:
        return split_opn('foo', simd_ext, typ, 1)
    return 'return {pre}foo{suf}({in0});'.format(**fmtspec)
```

Here are some notes concerning the Intel implementation:
1. `float16`s are emulated with two SIMD vectors of `float`s.
2. When the intrinsic is provided by Intel one can access it easily by
   constructing it with `{pre}` and `{suf}`. Indeed all Intel intrinsics
   names follow a pattern with a prefix indicating the SIMD extension and a
   suffix indicating the type of data. As for `{in0}`,  `{pre}` and
   `{suf}` are provided and contain the correct values with respect to
   `simd_ext` and `typ`, you do not need to compute them yourself.
3. When the intrinsic is not provided by Intel then one has to use tricks.
   + For `SSE2` one can use complete emulation, that is, putting the content of
     the SIMD vector into a C-array, working on it with a simple for loop and
     loading back the result into the resulting SIMD vector. As said before a
     lot of helper functions are provided and the `emulate_op1` Python function
     avoid writing by hand this for-loop emulation.
   + For `AVX` and `AVX512_KNL`, one can fallback to the "lower" SIMD extension
     (`SSE42` for `AVX` and `AVX2` for `AVX512_KNL`) by splitting the input
     vector into two smaller vectors belonging to the "lower" SIMD extension. In
     this case again the tedious and cumbersome work is done by the `split_opn`
     Python function.
4. Do not forget to add the `foo` entry to the `impls` dictionary in the `get_impl`
   Python function.

### For ARM

```python
def foo1(simd_ext, typ):
    ret = f16f64(simd_ext, typ, 'foo', 'foo', 1)
    if ret != '':
        return ret
    if simd_ext in neon:
        return 'return vfooq_{suf}({in0});'.format(**fmtspec)
    else:
        return 'return svfoo_{suf}_z({svtrue}, {in0});'.format(**fmtspec)
```

Here are some notes concerning the ARM implementation:
1. `float16`s can be natively supported but this is not mandatory.
2. On 32-bits ARM chips, intrinsics on `double` almost never exist.
3. The Python helper function `f16f64` hides a lot of details concerning the
   above two points. If the function returns a non empty string then it means
   that the returned string contains C code to handle the case given by the
   pair `(simd_ext, typ)`. We advise you to look at the generated C code. You
   will see the `nsimd_FP16` macro used. When defined it indicates that `nsimd`
   is compiled with native `float16` support. This also affect SIMD types (see
   `nsimd/include/arm/*/types.h`.)
4. Do not forget to add the `foo` entry to the `impls` dictionary in the
   `get_impl` Python function.

### Implementing the test for the operator

Now that we have written the implementations for the `foo` operator we must
write the corresponding tests. For tests all generations are done by
`egg/gen_tests.py`. Writing tests is more simple. The intrinsic that we just
implemented can be tested by an already-written test pattern code, namely by
the `gen_test` Python function.

Here is how the `egg/gen_tests.py` is organized:
1. The entry point is the `doit` function located at the bottom of the file.
2. In the `doit` function a dispatching is done according to the operator that
   is to be tested. All operators cannot be tested by the same C/C++ code. The
   reading of all different kind of tests is rather easy and we are not going
   through all the code in this document.
3. All Python functions generating test code begins with the following:
   ```python
       filename = get_filename(opts, op, typ, lang)
       if filename == None:
           return
   ```
   This must be the case for newly created function. The `get_filename` function
   ensures that the file must be created with respect to the command line
   options given to the `egg/hatch.py` script. Then note that to output to a
   file the Python function `open_utf8` must be used to handle Windows and to
   automatically put the MIT license at the beginning of generated files.
4. Tests must be written for C base API, the C++ base API and the C++ advanced
   API.

If you need to create a new kind of tests then the best way is to copy-paste
the Python function that produces the test that resembles the most to the test
you want. Then modify the newly function to suit your needs. Here is a quick
overview of Python functions present in the `egg/gen_test.py` file:
- `gen_nbtrue`, `gen_adv`, `gen_all_any` generate tests for reduction operators.
- `gen_reinterpret_convert` generates tests for non closed operators.
- `gen_load_store` generates tests for load/store operators.
- `gen_reverse` generates tests for one type of shuffle but can be extended
  for other kind of shuffles.
- `gen_test` generates tests for "standard" operators, typically those who do
  some computations. This is the kind of tests that can handle our `foo`
  operator and therefore nothing has to be done on our part.

### Conclusion

At first sight the implementation of `foo` seems complicated because intrinsics
for all types and all architectures are not provided by vendors. But `nsimd`
provides a lot of helper functions and tries to put away details so that
wrapping intrinsics is quickly done and easy, the goal is that the programmer
concentrate on the implementation itself. But be aware that more complicated
tricks can be implemented. Browse through a `platform_*.py` file to see what
kind of tricks are used and how they are implemented.


## How do I add a new category?

Adding a category is way much simplier than an operator. It suffices to add
a class with only one member named `title` as follows:
```python
class DocMyCategoryName(DocCategory):
    title = 'My category name functions'
```

The class must inherit from the `DocCategory` class and its name must begin
with `Doc`. The system will then take it into account, generate the entry
in the documentation and so on.
