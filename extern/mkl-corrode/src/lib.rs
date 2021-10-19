pub extern crate mkl_sys;

pub mod dss;
pub mod extended_eigensolver;
pub mod sparse;

mod util;

mod internal {
    pub trait InternalScalar {
        fn zero_element() -> Self;
        fn try_as_f64(&self) -> Option<f64>;
    }
}

/// Marker trait for supported scalar types.
///
/// Can not be implemented by dependent crates.
pub unsafe trait SupportedScalar: 'static + Copy + internal::InternalScalar {}

// TODO: To support f32 we need to pass appropriate options during handle creation
// Can have the sealed trait provide us with the appropriate option for this!
//impl private::Sealed for f32 {}
impl internal::InternalScalar for f64 {
    fn zero_element() -> Self {
        0.0
    }

    fn try_as_f64(&self) -> Option<f64> {
        Some(*self)
    }
}
//unsafe impl SupportedScalar for f32 {}
unsafe impl SupportedScalar for f64 {}
