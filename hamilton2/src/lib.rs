#![allow(clippy::excessive_precision)]
#![allow(clippy::too_many_arguments)]

/// Traits to model dynamic systems that can be integrated by this crate's integrators.
pub mod dynamic_system;
/// Implementations of various integration schemes for dynamic systems.
pub mod integrators;

/// Calculus helper traits and numerical differentiation
pub mod calculus;
/// Implementations of the Newton method with different line search strategies
pub mod newton;
