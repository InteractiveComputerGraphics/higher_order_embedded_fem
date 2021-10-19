# mkl-sys

[![Build Status](https://github.com/Andlon/mkl-sys/workflows/Build%20and%20run%20tests/badge.svg)](https://github.com/Andlon/mkl-sys/actions)

Auto-generated bindings to Intel MKL. Currently only supports Linux and Windows, and not considered stable/ready for production use. Only tested with Intel MKL 2019 and 2020.

This crate relies on Intel MKL having been installed on the target system,
and that the environment is set up for use with MKL.
The easiest way to make it work is to run the provided `mklvars.sh` setup script that is bundled with MKL.
This sets up the environment for use with MKL. This crate then detects the correct Intel MKL installation
by inspecting the value of the `MKLROOT` environment variable.

Note that we used to support `pkg-config`, but as of Intel MKL 2020, Intel is shipping broken
configurations. Therefore we instead directly rely on the value of `MKLROOT`.

## Windows support

### Compile time requirements
To compiled this create on Windows, the following requirements have to be met:
1. To run `bindgen` a Clang (`libclang`) installation is required. According to the `bindgen` [documentation](https://rust-lang.github.io/rust-bindgen/requirements.html#clang) version 3.9 should suffice. A recent pre-built version of Clang can be downloaded on the [LLVM release page](https://releases.llvm.org/download.html). To ensure that `bindgen` can find Clang, the environment variable `LIBCLANG_PATH` used by `bindgen` has to be set to point to the `bin` folder of the Clang installation. 
2. On Windows, Clang uses the MSVC standard library. Therefore, the build process should be started from a Visual Studio or Build Tools command prompt. The command prompt can be started from a start menu shortcut created by the Visual Studio installer or by running a `vcvars` script (e.g. `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat`) in an open command prompt. An IDE such as Clion with a configured MSVC toolchain should already provide this configuration for targets inside of the IDE.
3. The environment variable `MKLROOT` has to be configured properly to point to the path containing the `bin`, `lib`, `include`, etc. folders of MKL (e.g. `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl`). This can also be done by running the `mklvars.bat` script in the `bin` folder of MKL.

A script to build the library and run all tests on Windows might then look like this:
```
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\bin\mklvars.bat intel64"
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cargo test --release --features "all"
```

### Run time requirements
During runtime the corresponding redistributable DLLs of MKL (e.g. located in `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\mkl`) have to be in `PATH`.

## Known issues
- `bindgen` does not seem to be able to properly handle many preprocessor macros, such as e.g. `dss_create`.
This appears to be related to [this issue](https://github.com/rust-lang/rust-bindgen/issues/753).
- Generating bindings for the entire MKL library might take a lot of time. To circumvent this, you should use features
to enable binding generation only for the parts of the library that you will need. For example, the `dss` feature
generates bindings for the Direct Sparse Solver (DSS) interface.

A second approach that alleviates long build times due to `bindgen` is to use the following profile override
in your application's TOML file:

```toml
[profile.dev.package.bindgen]
opt-level = 2
```

This ensures that bindgen is compiled with optimizations on, significantly improving its runtime when
invoked by the build script in `mkl-sys`.

## License
Intel MKL is provided by Intel and licensed separately.

This crate is licensed under the MIT license. See `LICENSE` for details.

