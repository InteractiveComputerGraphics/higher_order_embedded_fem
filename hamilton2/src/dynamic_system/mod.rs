pub(crate) mod mutable;
pub(crate) mod stateless;
pub(crate) mod views;

pub use mutable::MutableDifferentiableDynamicSystem as DifferentiableDynamicSystem;
pub use mutable::MutableDynamicSystem as DynamicSystem;
pub use stateless::{StatelessDifferentiableDynamicSystem, StatelessDynamicSystem};
pub use views::IntoStateful;

// This is old code from exploring an operator based approach to the DynamicSystem trait
/*
mod stateless;
mod stateful;
mod views;

pub use stateful::{
    DifferentiableDynamicSystemSnapshot, DynamicSystemSnapshot, StatefulDynamicSystem,
};
pub use stateless::{StatelessDifferentiableDynamicSystem, StatelessDynamicSystem};
//pub use views::{AsStateful, AsStateless, StatelessView};
pub use views::StatefulView;

use nalgebra::{DVectorSlice, DVectorSliceMut, Scalar};
use std::error::Error;

pub trait Operator<T: Scalar> {
    /// Applies this operator to `x` and stores or accumulates the result into `y`.
    ///
    /// It depends on the concrete implementation if the result overwrites the content of `y`
    /// or is added to it.
    fn apply(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>>;
}
*/
