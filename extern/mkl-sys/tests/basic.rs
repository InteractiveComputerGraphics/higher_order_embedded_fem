use mkl_sys::{
    _MKL_DSS_HANDLE_t, dss_create_, dss_delete_, MKL_DSS_DEFAULTS, MKL_DSS_ZERO_BASED_INDEXING,
};
use std::ptr::null_mut;

#[test]
/// Calls some arbitrary MKL functions to ensure that linking and running an executable works
fn does_link_and_run() {
    let create_opts = MKL_DSS_DEFAULTS + MKL_DSS_ZERO_BASED_INDEXING;
    let mut handle: _MKL_DSS_HANDLE_t = null_mut();
    unsafe {
        let error = dss_create_(&mut handle, &create_opts);
        if error != 0 {
            panic!("dss_create error: {}", error);
        }
    }

    let delete_opts = MKL_DSS_DEFAULTS;
    unsafe {
        let error = dss_delete_(&mut handle, &delete_opts);
        if error != 0 {
            panic!("dss_delete error: {}", error);
        }
    }

    assert!(true);
}
