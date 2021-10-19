use bindgen::callbacks::{IntKind, ParseCallbacks};
use bindgen::EnumVariation;
use std::env;
use std::path::PathBuf;

/// Paths required for linking to MKL from MKLROOT folder
struct MklDirectories {
    lib_dir: String,
    omp_lib_dir: String,
    include_dir: String,
}

impl MklDirectories {
    /// Constructs paths required for linking MKL from the specified root folder. Checks if paths exist.
    fn try_new(mkl_root: &str) -> Result<Self, String> {
        let os = if cfg!(target_os = "windows") {
            "win"
        } else if cfg!(target_os = "linux") {
            "lin"
        } else {
            return Err("Target OS not supported".into());
        };

        let arch = if cfg!(target_arch = "x86_64") {
            "64"
        } else {
            return Err("Target architecture not supported".into());
        };

        let mkl_root: String = mkl_root.into();
        let prefix: String = mkl_root.clone();
        let exec_prefix: String = prefix.clone();
        let lib_dir = format!(
            "{exec_prefix}/lib/intel{arch}_{os}",
            exec_prefix = exec_prefix,
            arch = arch,
            os = os
        );
        let omp_lib_dir = format!(
            "{exec_prefix}/../compiler/lib/intel{arch}_{os}",
            exec_prefix = exec_prefix,
            arch = arch,
            os = os
        );
        let include_dir = format!("{prefix}/include", prefix = prefix);

        let mkl_root_path = PathBuf::from(mkl_root);
        let lib_dir_path = PathBuf::from(lib_dir);
        let omp_lib_dir_path = PathBuf::from(omp_lib_dir);
        let include_dir_path = PathBuf::from(include_dir);

        let mkl_root_str = mkl_root_path
            .to_str()
            .ok_or("Unable to convert 'mkl_root' to string")?;
        let lib_dir_str = lib_dir_path
            .to_str()
            .ok_or("Unable to convert 'lib_dir_path' to string")?;
        let omp_lib_dir_str = omp_lib_dir_path
            .to_str()
            .ok_or("Unable to convert 'omp_lib_dir_path' to string")?;
        let include_dir_str = include_dir_path
            .to_str()
            .ok_or("Unable to convert 'include_dir_path' to string")?;

        // Check if paths exist

        if !mkl_root_path.exists() {
            println!(
                "cargo:warning=The 'mkl_root' folder with path '{}' does not exist.",
                mkl_root_str
            );
        }

        if !lib_dir_path.exists() {
            println!(
                "cargo:warning=The 'lib_dir_path' folder with path '{}' does not exist.",
                lib_dir_str
            );
        }

        if cfg!(feature = "openmp") && !omp_lib_dir_path.exists() {
            println!(
                "cargo:warning=The 'omp_lib_dir_path' folder with path '{}' does not exist.",
                omp_lib_dir_str
            );
        }

        if !include_dir_path.exists() {
            println!(
                "cargo:warning=The 'include_dir_path' folder with path '{}' does not exist.",
                include_dir_str
            );
        }

        Ok(MklDirectories {
            lib_dir: lib_dir_str.into(),
            omp_lib_dir: omp_lib_dir_str.into(),
            include_dir: include_dir_str.into(),
        })
    }
}

fn get_lib_dirs(mkl_dirs: &MklDirectories) -> Vec<String> {
    if cfg!(feature = "openmp") {
        vec![mkl_dirs.lib_dir.clone(), mkl_dirs.omp_lib_dir.clone()]
    } else {
        vec![mkl_dirs.lib_dir.clone()]
    }
}

fn get_link_libs_windows() -> Vec<String> {
    // Note: The order of the libraries is very important
    let mut libs = Vec::new();

    if cfg!(feature = "ilp64") {
        libs.push("mkl_intel_ilp64_dll");
    } else {
        libs.push("mkl_intel_lp64_dll");
    };

    if cfg!(feature = "openmp") {
        libs.push("mkl_intel_thread_dll");
    } else {
        libs.push("mkl_sequential_dll");
    };

    libs.push("mkl_core_dll");

    if cfg!(feature = "openmp") {
        libs.push("libiomp5md");
    }

    libs.into_iter().map(|s| s.into()).collect()
}

fn get_link_libs_linux() -> Vec<String> {
    // Note: The order of the libraries is very important
    let mut libs = Vec::new();

    if cfg!(feature = "ilp64") {
        libs.push("mkl_intel_ilp64");
    } else {
        libs.push("mkl_intel_lp64");
    };

    if cfg!(feature = "openmp") {
        libs.push("mkl_intel_thread");
    } else {
        libs.push("mkl_sequential");
    };

    libs.push("mkl_core");

    if cfg!(feature = "openmp") {
        libs.push("iomp5");
    }
    libs.extend(vec!["pthread", "m", "dl"]);

    libs.into_iter().map(|s| s.into()).collect()
}

fn get_link_libs() -> Vec<String> {
    if cfg!(target_os = "windows") {
        get_link_libs_windows()
    } else if cfg!(target_os = "linux") {
        get_link_libs_linux()
    } else {
        panic!("Target OS not supported");
    }
}

fn get_cflags_windows(mkl_dirs: &MklDirectories) -> Vec<String> {
    let mut cflags = Vec::new();

    if cfg!(feature = "ilp64") {
        cflags.push("-DMKL_ILP64".into());
    }

    cflags.push("--include-directory".into());
    cflags.push(format!("{}", mkl_dirs.include_dir));
    cflags
}

fn get_cflags_linux(mkl_dirs: &MklDirectories) -> Vec<String> {
    let mut cflags = Vec::new();

    if cfg!(feature = "ilp64") {
        cflags.push("-DMKL_ILP64".into());
    }

    cflags.push("-I".into());
    cflags.push(format!("{}", mkl_dirs.include_dir));
    cflags
}

fn get_cflags(mkl_dirs: &MklDirectories) -> Vec<String> {
    if cfg!(target_os = "windows") {
        get_cflags_windows(mkl_dirs)
    } else if cfg!(target_os = "linux") {
        get_cflags_linux(mkl_dirs)
    } else {
        panic!("Target OS not supported");
    }
}

#[derive(Debug)]
pub struct Callbacks;

impl ParseCallbacks for Callbacks {
    fn int_macro(&self, name: &str, _value: i64) -> Option<IntKind> {
        // This forces all MKL constants to be signed. Otherwise `bindgen` might
        // give different types to different constants, which is inconvenient.
        // MKL expects these constants to be compatible with MKL_INT.
        if &name[..4] == "MKL_" {
            // Important: this should be the same as MKL_INT
            if cfg!(feature = "ilp64") {
                Some(IntKind::I64)
            } else {
                Some(IntKind::I32)
            }
        } else {
            None
        }
    }
}

fn main() {
    if cfg!(not(any(
        feature = "all",
        feature = "dss",
        feature = "sparse-matrix-checker",
        feature = "extended-eigensolver",
        feature = "inspector-executor"
    ))) {
        panic!(
            "No MKL modules selected.
To use this library, please select the features corresponding \
to MKL modules that you would like to use, or enable the `all` feature if you would \
like to generate symbols for all modules."
        );
    }

    // Link with the proper MKL libraries and simultaneously set up arguments for bindgen.
    // Otherwise we don't get e.g. the correct MKL preprocessor definitions).
    let clang_args = {
        let mklroot = match env::var("MKLROOT") {
            Ok(mklroot) => mklroot,
            Err(_) => panic!(
"Environment variable 'MKLROOT' is not defined. Remember to run the mklvars script bundled
with MKL in order to set up the required environment variables."),
        };

        let mkl_dirs = MklDirectories::try_new(&mklroot).unwrap();

        for lib_dir in get_lib_dirs(&mkl_dirs) {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }

        for lib in get_link_libs() {
            println!("cargo:rustc-link-lib={}", lib);
        }

        let args = get_cflags(&mkl_dirs);
        args
    };

    #[allow(unused_mut)]
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(Callbacks))
        .default_enum_style(EnumVariation::ModuleConsts)
        .impl_debug(true)
        .derive_debug(true)
        .clang_args(clang_args);

    // If only part of MKL is needed, we use features to construct whitelists of
    // the needed functionality. These can be overridden with the "all" feature, which
    // avoids whitelisting and instead encompasses everything.
    #[cfg(not(feature = "all"))]
    {
        #[cfg(feature = "dss")]
        {
            let dss_regex = "(dss_.*)|(DSS_.*)|(MKL_DSS.*)";
            builder = builder
                .whitelist_function(dss_regex)
                .whitelist_type(dss_regex)
                .whitelist_var(dss_regex);
        }

        #[cfg(feature = "sparse-matrix-checker")]
        {
            builder = builder
                .whitelist_function("sparse_matrix_checker*")
                .whitelist_function("sparse_matrix_checker_init*");
        }

        #[cfg(feature = "extended-eigensolver")]
        {
            builder = builder
                .whitelist_function(".*feast.*")
                .whitelist_function("mkl_sparse_ee_init")
                .whitelist_function("mkl_sparse_._svd")
                .whitelist_function("mkl_sparse_._ev")
                .whitelist_function("mkl_sparse_._gv");
        }

        #[cfg(feature = "inspector-executor")]
        {
            builder = builder.whitelist_function("mkl_sparse_.*");
        }
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
