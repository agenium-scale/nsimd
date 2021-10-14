# C++20 concepts

As of C++20, concepts are available. We quote <en.cppreference.com> to
introduce concepts.

*Class templates, function templates, and non-template functions (typically
members of class templates) may be associated with a constraint, which
specifies the requirements on template arguments, which can be used to select
the most appropriate function overloads and template specializations.*

*Named sets of such requirements are called concepts. Each concept is a
predicate, evaluated at compile time, and becomes a part of the interface of a
template where it is used as a constraint*

## Concepts provided by NSIMD

All concepts provided by NSIMD comes in two forms:
- The native C++20 form in the `nsimd` namespace
- As a macro for keeping the compatibility with older versions of C++

The following tables list all concepts and is exhaustive. Native concepts are
accessible through the `nsimd` namespace. They take only one argument. Their
macro counterparts take no argument as they are meant to be used as
constraint placeholder types. When compiling for older C++ versions NSIMD
concepts macros are simply read as `typename` by the compiler.

Table for base C and C++ APIs:

| Native concept              | Macro                              | Description                                    |
|:----------------------------|:-----------------------------------|:-----------------------------------------------|
| `simd_ext_c`                | `NSIMD_CONCEPT_SIMD_EXT`           | Valid SIMD extension                           |
| `simd_value_type_c`         | `NSIMD_CONCEPT_VALUE_TYPE`         | Valid NSIMD underlying value type              |
| `simd_value_type_or_bool_c` | `NSIMD_CONCEPT_VALUE_TYPE_OR_BOOL` | Valid NSIMD underlying value type or `bool`    |
| `alignment_c`               | `NSIMD_CONCEPT_ALIGNMENT`          | Valid NSIMD alignment `aligned` or `unaligned` |

Table for advanced C++ API:

| Native concept | Macro                    | Description                                    |
|:---------------|:-------------------------|:----------------------|
| `is_pack_c`    | `NSIMD_CONCEPT_PACK`     | Valid NSIMD pack      |
| `is_packl_c`   | `NSIMD_CONCEPT_PACKL`    | Valid NSIMD packl     |
| `is_packx1_c`  | `NSIMD_CONCEPT_PACKX1`   | Valid NSIMD packx1    |
| `is_packx2_c`  | `NSIMD_CONCEPT_PACKX2`   | Valid NSIMD packx2    |
| `is_packx3_c`  | `NSIMD_CONCEPT_PACKX3`   | Valid NSIMD packx3    |
| `is_packx4_c`  | `NSIMD_CONCEPT_PACKX4`   | Valid NSIMD packx4    |
| `any_pack_c`   | `NSIMD_CONCEPT_ANY_PACK` | Any of the above pack |

## Expressing C++20 constraints

Expressing constraints can of course be done with the `requires` keyword. But
for compatibility with older C++ versions NSIMD provides `NSIMD_REQUIRES`
which take as onyl argument the constraints.

```c++
template <typename T, typename S>
NSIMD_REQUIRES(sizeof(T) == sizeof(S))
void foo(T, S);
```

It is advised to use doubled parenthesis as coma in the constraints expression
can be interpreted as argument separators for the macro itself.

```c++
template <typename T, typename S>
NSIMD_REQUIRES((std::is_same<T, S>))
void foo(T, S);
```

Note that when expressing constraints using `nsimd::sizeof_v`'s prefer the
NSIMD definition of sizeof for the following reason: when dealing with
float16's one cannot know the underlying representation of such a type as it
is non-portable and non-standard, but NSIMD provides helper functions to
transparently deal with float16's as if they were 16-bits wide. Therefore
expressing sizeof equality should be done with `nsimd::sizeof_v`.

```c++
template <typename T, typename S>
NSIMD_REQUIRES((nsimd::sizeof_v<T> == nsimd::sizeof_v<S>))
void foo(T, S);
```
