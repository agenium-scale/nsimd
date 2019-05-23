How to contribute?
==================

You are welcome to contribute to NSIMD. This document gives some details on how
to add/wrap new intrinsics. When you have finished fixing some bugs or adding
some new features, please make a pull request. One of our repository maintainer
will then merge of comment hte pull request.

Prerequisites
-------------

- Respect the philosophy of the library (cf. [README.md](README.md).)
- Basic knowledge of Python 3.
- Good knowledge of C.
- Good knowledge of C++.
- Good knowledge of SIMD programming.

How do I add support for a new intrinsics?
==========================================

Introduction
------------

We will add support to the library for the following imaginary intrinsic: given
a SIMD vector, suppose that this intrisic called `foo` takes each element `x`
and compute `1 / (1 - x) + 1 / (1 - x)^2` and suppose that hardware vendors all
propose this intrisic only for floatting point numbers as follows:

- Intel (no intrinsics are given for float16)
  + SSE2: no intrinsics is provided.
  + SSE42: `_mm_foo_ps` for floats and `_mm_foo_pd` for doubles.
  + AVX: no intrinsics is provided.
  + AVX2: `_mm256_foo_ps` for floats and `_mm256_foo_pd` for doubles.
  + AVX512\_KNL: no intrinsics is provided.
  + AVX512\_SKYLAKE: `_mm512_foo_ps` for floats and `_mm512_foo_pd` for doubles.
- ARM
  + NEON128: `vfooq_f16` for float16, `vfooq_f32` for floats and no intrinsics
    for doubles.
  + AARCH64: same as NEON128 and `vfooq_f64` for doubles.
  + SVE: `svfoo_f16`, `svfoo_f32` and `svfoo_f64` for respectively float16,
    floats and doubles.

First thing to do is to declare this new intrinsic to the generation system.
A lot of work is done by the generation system such as generating all functions
signatures for C and C++ APIs, tests, benchmarks and documentation. Of course
the default documentation does not say much but you can of course add a better
description.

Registering the intrinsic (or operator)
---------------------------------------

A function, or an intrinsic is called an operator in the generation system.
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

This little class will be processed by the generation system to be available
for the end-user of the library in both C and C++ APIs. Each member of this
class controls how the generation will be done:

- `full_name` is a string containing the human readable name of the operator.
  If not given the class name will be taken for it.
- `signature` is a string describing what kind of arguments and how many takes
  the operator. This member is mandatory and must respect the following:
  `return_type name_of_operator arg1_type arg2_type ...` where `return_type`
  and `arg*_type` can be taken from the following:
  + `v`   = SIMD vector parameter
  + `vx2` = struct of 2 SIMD vector parameters
  + `vx3` = struct of 3 SIMD vector parameters
  + `vx4` = struct of 4 SIMD vector parameters
  + `l`   = SIMD vector of logicals parameter
  + `s`   = Scalar parameter
  + `*`   = Pointer to scalar parameter
  + `c*`  = Pointer to const scalar parameter
  + `_`   = void (only for return type)
  + `p`   = Parameter (int)

In our case `v foo v` means that `v` takes a SIMD vector and returns a SIMD
vector. Several signatures will be generated for the intrinsic according to
the types it can supports. In our case the intrinsic only support floatting
point types.

- `types` is a Python list indicating which types are supported by the
  intrinsic. If not given the intrinsic is supposed to support all types.
  Some Python lists are predefined to help the programmer:
  + `ftypes = ['f64', 'f32', 'f16']` = all floatting point types
  + `ftypes_no_f16 = ['f64', 'f32']`
  + `itypes = ['i64', 'i32', 'i16', 'i8']` = all signed integer types
  + `utypes = ['u64', 'u32', 'u16', 'u8']` = all unsigned integer types
  + `iutypes = itypes + utypes`
  + `types = ftypes + iutypes`
- `domain` is a string indicating the mathematical domain of definition of the
  operator. This helps for benchmarks and tests for generating random input
  numbers in the correct interval. In our case `R\{1}` means all real numbers
  (of course all floating point numbers) expect -1 for which the operator
  cannot be computed. For examples see how other operators are defined in
  `egg/operators.py`.
- `categories` is a list of Python classes that indicated the generation
  system to which categories it belongs. The list of available categories
  is as follow:
  + `DocShuffle` for Shuffle functions
  + `DocTrigo` for Trigonometric functions
  + `DocHyper` for Hyperbolic functions
  + `DocExpLog` for Exponential and logarithmic functions
  + `DocBasicArithmetic` for Basic arithmetic operators
  + `DocBitsOperators` for Bits manipulation operators
  + `DocLogicalOperators` for Logicals operators
  + `DocMisc` for Miscellaneous
  + `DocLoadStore` for Loads & stores
  + `DocComparison` for Comparison operators
  + `DocRounding` for Rounding functions
  + `DocConversion` for Conversion operators
  If no category correspond to the operator you want to add to NSIMD then feel
  free to create a new category (cf. at the bottom of this document)

Many other members are supported by the generation system. We describe them
quickly here and will give more details in a later version of this document.
Default values are given after the equal sign:

- `cxx_operator [= None]` in case the operator has a corresponding C++ operator.
- `autogen_cxx_adv [= True]` in case the C++ advanced API signatures for this
  operator must not be auto-generated.
- `closed [= True]` in case the operator output type differs from its input
  type.
- `src [= False]` in case the code source must be compiled and therefore not
  auto-generated.
- `load_store [= False]` in case the operator loads/store data from/to
  memory.
- `do_bench [= True]` incase benchmarks for the operator must not be
  auto-generated.
- `desc [= '']` description that will appear in the documentation for the
  operator.
- `bench_auto_against_cpu [= True]` for auto-generation of benchmark against
  NSIMD CPU implementation.
- `bench_auto_against_mipp [= False]` for auto-generation of benchmark against
  the MIPP library.
- `bench_auto_against_sleef [= False]` for auto-generation of benchmark against
  the Sleef library.
- `bench_auto_against_std [= False]` for auto-generation of benchmark against
  the standard library.
- `tests_mpfr [= False]` in case the operator has an MPFR counterpart for
  comparison.
- `tests_ulps [= False]` in case the auto-generated tests has to compare ulps.

Implementing the operator
-------------------------

Now that the operator is registered, all signatures will be generated but
the implemenatations will be missing. Type

```sh
python3 egg/hatch.py -Af
```

and the following files (among many) should appear:

- `include/nsimd/x86/cpu/foo.h`
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
SIMD at all. This is used on architectures not supported by NSIMD or when the
architectures does not offer any SIMD.

Providing implementations for `foo` is done by completing the following Python
files:

- `egg/platform_cpu.py`
- `egg/platform_x86.py`
- `egg/platform_arm.py`

The idea is to produce plain C (not C++) code using Python string format. Each
`platform_*.py` file provides somes helper functions to ease as much as
possible the programmer's job. But every `platform_*.py` file provides the
same "global" variables available in every functions and is designed in the
same way:

1. At the bottom of the file is the `get_impl` function taking the
   following arguments:
   + `func`: the name of the operator the system is auto-generating.
   + `simd_ext`: the SIMD extension for which the system wants the implemetation.
   + `from_typ`: the input type of the argument that will be passed to the
     operator.
   + `to_typ`: the output type produced by the operator
2. Inside this function lies a Python dict that provides functions implementing
   each operator. The string containing the C code for the implementations can
   be put here directly but usually the string is returned by a Python function
   that is written above in the same file.
3. At the top of the file lies helper functions that generates code. This is
   specific to each architecture. Do not hesitate to look at it.

Let's begin by the `cpu` implementations. It turns out that there is no SIMD
extension in this case, and by convention, `simd_ext == 'cpu'` and this
argument can be discarded. So we first add an entry to the `impls` Python dict
of the `get_impl` function:

```python
    impls = {

        ...

        'reverse': reverse1(from_typ),
        'addv': addv(from_typ),
        'foo': foo1(from_typ) # Added at the bottom of the dict
    }
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

    ...
```

Then above in the file we write the Python function `foo1` that will provide
the C implementation of the operator `foo`:

```python
def foo1(typ):
    if typ == 'f16':
        return '''return nsimd_f32_to_f16(
                             1 / (1 - {in0}.f) +
                             1 / ((1 - {in0}.f) * (1 - {in0}.f))
                         );'''.format(**fmtspec)
    else:
        return 'return 1 / (1 - {in0}) + 1 / ((1 - {in0}) * (1 - {in0}));'. \
               format(**fmtspec)
```

First note that the arguments names passed to the operator in its C
implementation are not known in the Python side. Several other parameters
are not known or are cumbersome to find out. Therefore each function has access
to the `fmtspec` Python dict that hold all these values:

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

Then note that as plain C (and C++) does not support native 16-bits wide
floating point types NSIMD emulates it with a C struct containing a single
float whose name is `f`. So when `typ == 'f16'` our Python `foo1` function
returns C code that does the computation with `{in0}.f` where `{in0}` is still
the argument name, which is a struct, exposing its `.f` member which contains
the actual value.

For each SIMD extension one can find a `types.h` file (for `cpu` the files can
be found in `include/nsimd/cpu/cpu/types.h`) that declares all SIMD
types. Therfore if you have any doubt on a given type do not hesitate to take
a look at this file. Note also that this file is auto-generated and is
therefore readable only after a successfull first `python3 egg/hatch -Af`.

Now that the `cpu` implementation is written, you should be able to write the
implementation of `foo` for other architectures. Each architecture has its
particularities. We will cover them now by providing directly the Python
implementations and explaining in less details.

Finally note that `clang-format` is called by the generation system to
autoformat produced C/C++ code. Therefore prefer indenting C code string within
the Python according to Python indentations, do not write C code beginning
at column 0.

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

1. Float16 are emulated with two SIMD vectors of floats.
2. When the intrinsic is provided by Intel one can access it easily by
   constructing it with `{pre}` and `{suf}`. Indeed all Intel intrinsics
   names follow a pattern with a prefix indicating the SIMD extension and a
   suffix indicating the type of data. As for `{in0}` and other `{pre}` and
   `{suf}` are provided and contain the correct values with respect to
   `simd_ext` and `typ`.
3. When the intrinsic is not provided by Intel then one has to use tricks.
   + For SSE2 one can use the complete emulation, that is, putting the content
     of the SIMD vector into a C-array, working on it with a simple for loop and
     loading the result into the resulting SIMD vector. As said before a lot of
     helper functions are provided and the `emulate_op1` Python function avoid
     writing by hand this for-loop emulation.
   + For AVX and AVX512\_KNL, one can fallback to the "lower" SIMD extension
     (SSE42 for AVX amd AVX2 for AVX512\_KNL) by splitting the input vector
     into two smaller vectors belonging to the "lower" SIMD extension. In this
     case again the tedious and cumbersome work is done by the `split_opn`
     Python function.
4. Do not forget to add the `foo` entry to the `impls` dict in the `get_impl`
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

1. Float16 can be natively supported but this is not mandatory.
2. On 32-bits ARM chips, intrisics on double almost never exist.
3. The Python helper function `f16f64` hides a lot of details concerning the
   above two points. If the function returns a non empty string then it means
   that it must handle the case given by the pair `(simd_ext, typ)`. We
   advise you to look at the generated C code. You will see the `NSIMD_FP16`
   macro used. When defined it indicates that NSIMD is compiled with native
   float16 support. This also affect SIMD types (cf. `types.h`.)
4. Do not forget to add the `foo` entry to the `impls` dict in the `get_impl`
   Python function.

Implementing the test for the operator
--------------------------------------

Now that we have written the implementations for the `foo` operator we must
write the corresponding tests. For testing all generatipm is done by the
`egg/gen_tests.py`. Writing tests is more simple. The intrinsic that we
just implemented can be tested by an already-written test code, namely in the
`gen_test` Python function.

Here is how the `egg/gen_tests.py` is organized:

1. The entry point if the `doit` function located art the bottom of the file.
2. In the `doit` function a dispatching is done according to the operator that
   is to be tested. All operators cannot be tested by the same C code. The
   reading of all different kind of tests is rather easy and we are not going
   through all the code in this document.
3. All Python functions generating test code begins with the following:
   ```python
       filename = get_filename(opts, op, typ, lang)
       if filename == None:
           return
   ```
   This must be the case for newly created function. The `get_filename`
   function ensures that the file must be created with respect to the command
   line options given to the `egg/hatch.py` script. Then note that to output
   to a file the Python function `open_utf8` must be used to handle Windows
   and to automatically put the MIT license at the beginning of generated
   files.

If you need to create a new kind of tests then the best way is to copy-paste
the Python function that produces tests that resembles the most to the tests
you want. Then modify the newly function to suits your needs. Here is a
quick overview of Python functions present in the `egg/gen_test.py` file:

- `gen_nbtrue`, `gen_addv`, `gen_all_any` generate tests for reduction
  operators.
- `gen_reinterpret_convert` generates tests for non closed operators.
- `gen_load_store` generates tests for load/store operators.
- `gen_reverse` gemnerates tests for one type of shuffle but can be extended
  for other kind of shuffles.
- `gen_test` generates tests for "standard" operators, typically those who
  do some computations. This is the kind of tests that can handle our `foo`
  operator and therefore nothing has to be done on our part.

Conclusion
----------

At first sight the implementation of `foo` seems complicated because intrinsics
for all types and all architectures are not provided by vendors. But NSIMD
provides a lot of helper functions and tries to put away details so that
wrapping intrinsics is quickly done and easy, the goal is that the programmer
concentrate on the implementation itself. But be aware that more complicated
tricks can be implemented. Browse through a `platform_*.py` file to see what
kind tricks are used and how they are implemented.

How do I add a new category?
============================

Adding a category is way much simplier than an operator. It suffices to add
a class with only one member named `title` as follows:

```python
class DocMyCategoryName(DocCategory):
    title = 'My category name functions'
```

The class must inherit from the `DocCategory` class and its name must begin
with `Doc`. The system will then take it into account, generate the entry
in the documentation and so on.
