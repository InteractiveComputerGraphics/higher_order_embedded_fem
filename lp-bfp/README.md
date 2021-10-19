lp-bfp
======

A tiny Rust library written as a wrapper around a small C++ project that relies on an external LP solver to find a basic feasible point of a Linear Program.

To use, you need:

- To install Google OR-Tools. Get it here: https://developers.google.com/optimization.
- To set the environment variable `ORTOOLS_ROOT` to the root directory of your OR-Tools installation.
The root directory is the folder containing `lib`, `bin` etc.
- To make sure that the `lib` directory in `ORTOOLS_ROOT` is available in your library search path.
For Linux platforms, this can be accomplished by e.g. including `$ORTOOLS_ROOT/lib` in the `$LD_LIBRARY_PATH` environment
variable. For Windows, this can be accomplished by including the lib directory in `PATH` (TODO: Is this correct?). 
