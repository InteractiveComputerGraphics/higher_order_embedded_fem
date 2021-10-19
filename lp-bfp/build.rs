use cmake;
use std::env::vars;

fn main() {
    let cpp = cmake::build("cpp");

    let or_tools_root = vars()
        .find(|(var, _)| var == "ORTOOLS_ROOT")
        .map(|(_, value)| value)
        .expect("Could not find ORTOOLS_ROOT");

    let or_tools_lib_path = format!("{}/lib", or_tools_root);

    // Link static shim library
    println!("cargo:rustc-link-search=native={}/lib", cpp.display());
    println!("cargo:rustc-link-lib=static=lp-bfp");

    // Link OR-tools
    println!("cargo:rustc-link-search=native={}", &or_tools_lib_path);
    println!("cargo:rustc-link-lib=dylib=ortools");

    // TODO: Come up with something more general?
    if cfg!(target_env = "gnu") {
        // GCC-specific, link to C++ stdlib
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
