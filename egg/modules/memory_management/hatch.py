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

import common

# -----------------------------------------------------------------------------

def name():
    return 'Memory management'

def desc():
    return '''This module provides C-style memory managmenent functions:
malloc, calloc, free, copy to/from devices, etc... Its purpose is to facilitate
the use of data buffers in a portable way for systems with CPUs only and
for systems with CPUs and GPUs.'''

def doc_menu():
    return dict()

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating module memory_management')
    if not opts.doc:
        return
    filename = common.get_markdown_file(opts, 'overview', 'memory_management')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# Overview

This module provides C-style memory managmenent functions. Its purpose is not
to become a fully feature container library. It is to provide portable
malloc, memcpy and free functions with a little helpers to copy data from and
to the devices.

# API reference

## Equivalents of malloc, calloc, memcpy and free for devices

Note that the below functions simply wraps the corresponding C functions
when targeting a CPU.

- `template <typename T> T *device_malloc(size_t sz)`{br}
  Allocates `sz * sizeof(T)` bytes of memory on the device.
  On error NULL is returned.

- `template <typename T> T *device_calloc(size_t sz)`{br}
  Allocates `sz * sizeof(T)` bytes of memory on the device and set the
  allocated memory to zero.
  On error NULL is returned.

- `template <typename T> void device_free(T *ptr)`{br}
  Free the memory pointed to by the given pointer.

- `template <typename T> void copy_to_device(T *device_ptr, T *host_ptr,
  size_t sz)`{br}
  Copy data to from host to device.

- `template <typename T> void copy_to_host(T *host_ptr, T *device_ptr,
  size_t sz)`{br}
  Copy data to from device to host.

- `#define nsimd_fill_dev_mem_func(func_name, expr)`{br}
  Create a device function that will fill data with `expr`. To call the created
  function one simply does `func_name(ptr, sz)`. The `expr` argument represents
  some simple C++ expression that can depend only on `i` the i-th element in
  the vector as shown in the example below.

  ```c++
  nsimd_fill_dev_mem_func(prng, ((i * 1103515245 + 12345) / 65536) % 32768)

  int main() {{
    prng(ptr, 1000);
    return 0;
  }}
  ```

## Pairs of pointers

It is often useful to allocate a pair of data buffers: one on the host and
one on the devices to perform data transfers. The below functions provides
quick ways to malloc, calloc, free and memcpy pointers on host and devices at
once. Note that when targeting CPUs the pair of pointers is reduced to one
pointer that ponit the a single data buffer in which case memcpy's are not
performed. Note also that there is no implicit synchronization of data
between both data buffers. It is up to the programmer to triggers memcpy's.

```c++
template <typename T>
struct paired_pointers_t {{
  T *device_ptr, *host_ptr;
  size_t sz;
}};
```

Members of the above structure are not to be modified but can be passed as
arguments for reading/writing data from/to memory they point to.

- `template <typename T> paired_pointers_t<T> pair_malloc(size_t sz)`{br}
  Allocate `sz * sizeof(T)` bytes of memory on the host and on the device.
  If an error occurs both pointers are NULL.

- `template <typename T> paired_pointers_t<T> pair_malloc_or_exit(size_t
  sz)`{br}
  Allocate `sz * sizeof(T)` bytes of memory on the host and on the device.
  If an error occurs, prints an error message on stderr and exit(3).

- `template <typename T> paired_pointers_t<T> pair_calloc(size_t sz)`{br}
  Allocate `sz * sizeof(T)` bytes of memory on the host and on the device.
  Write both data buffers with zeros.
  If an error occurs both pointers are NULL.

- `template <typename T> paired_pointers_t<T> pair_calloc_or_exit(size_t
  sz)`{br}
  Allocate `sz * sizeof(T)` bytes of memory on the host and on the device.
  Write both data buffers with zeros.
  If an error occurs, prints an error message on stderr and exit(3).

- `template <typename T> void pair_free(paired_pointers_t<T> p)`{br}
  Free data buffers on the host and the device.

- `template <typename T> void copy_to_device(paired_pointers_t<T> p)`{br}
  Copy data from the host buffer to its corresponding device buffer.

- `template <typename T> void copy_to_host(paired_pointers_t<T> p)`{br}
  Copy data from the device buffer to its corresponding host buffer.
'''.format(br='  '))

