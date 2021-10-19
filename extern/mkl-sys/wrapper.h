#include <mkl_types.h>

// For some types, MKL uses preprocessor macros as a preprocessor alternative to
// typedefs. Unfortunately, this makes it just about impossible for `bindgen`
// to understand that it's actually a typedef. To remedy the situation,
// we replace the preprocessor macros with actual, proper typedefs.

/// Underlying MKL_INT type. This is an intermediate type alias introduced by `mkl-sys`,
/// and should never be directly referenced.
typedef MKL_INT ____MKL_SYS_UNDERLYING_MKL_INT;

/// Underlying MKL_UINT type. This is an intermediate type alias introduced by `mkl-sys`,
/// and should never be directly referenced.
typedef MKL_UINT ____MKL_SYS_UNDERLYING_MKL_UINT;

#undef MKL_INT
#undef MKL_UINT
typedef ____MKL_SYS_UNDERLYING_MKL_INT MKL_INT;
typedef ____MKL_SYS_UNDERLYING_MKL_UINT MKL_UINT;

#include <mkl.h>