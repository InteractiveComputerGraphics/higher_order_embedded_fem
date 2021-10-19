/*!

# mkl-sys

Auto-generated bindings to Intel MKL.

Currently only tested on Linux, and should be **considered experimental and unstable**
(in an API sense).

This crate relies on Intel MKL having been installed on the target system,
and that the environment is set up for use with MKL.
It uses `pkg-config` to determine library paths. The easiest way to make it work is to run the provided
`mklvars.sh` setup script that is bundled with MKL.

The library can generate bindings for only selected modules of MKL, or for the entire library.
By default, no modules are selected, and compilation will fail with an error message. To use
this library, enable the features corresponding to the desired MKL modules, or enable the
"all" feature if you want to generate code for all of MKL. At the moment, the currently available
features corresponding to MKL modules are:

- `all`: Create bindings for all modules.
- `dss`: The Direct Sparse Solver (DSS) interface.
- `sparse-matrix-checker`: Routines for checking validity of sparse matrix storage.
- `extended-eigensolver`: Routines for the Extended Eigensolver functionality.
- `inspector-executor`: The Inspector-Executor API for sparse matrices.

It is strongly recommended to only enable the modules that you need, otherwise the effects
on compilation time may be severe. See "Known issues" below.

By default, the sequential version of MKL is used. To enable OpenMP support, enable the
`openmp` feature.

By default, 32-bit integers are used for indexing. This corresponds to the `lp64` configuration
of MKL. To use 64-bit integers with the `ilp64` configuration, enable the `ilp64` feature.

Please refer to the Intel MKL documentation for how to use the functions exposed by this crate.

## Contributions
Contributions are very welcome. I am generally only adding features to this library as I need them.
If you require something that is not yet available, please file an issue on GitHub
and consider contributing the changes yourself through a pull request.

## Known issues
- `bindgen` does not handle many preprocessor macros used by MKL, such as e.g. `dss_create`.
It also does not generate type aliases for #define-based type aliases, such as e.g. `MKL_INT`.
Some of these types are manually added to this library, but they do not appear in the
function arguments.
- Generating bindings for the entire MKL library takes a lot of time. This is a significant issue for debug
builds, as we currently have no way of forcing optimizations for bindgen when dependent projects are
built without optimizations. To circumvent this, you should use features to enable binding generation
only for the parts of the library that you will need. For example, the `dss` feature generates bindings for the
Direct Sparse Solver (DSS) interface.

*/

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));