use nalgebra::{DVectorSlice, DVectorSliceMut, Scalar};
use std::error::Error;

/// An abstract dynamic system represented by a second-order ODE.
///
/// A dynamic system is represented by the second-order system of ODEs
///
/// ```ignore
///    M dv/dt = f(t, u, v),
///      du/dt = v.
/// ```
///
/// where M is constant.
pub trait StatelessDynamicSystem<T: Scalar> {
    /// Apply the mass matrix `M` to the vector `x` and accumulate the result in `y`,
    /// yielding `y = y + Mx`.
    fn apply_mass_matrix(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>);

    fn apply_inverse_mass_matrix(&self, sol: DVectorSliceMut<T>, rhs: DVectorSlice<T>) -> Result<(), Box<dyn Error>>;

    fn eval_f(&self, f: DVectorSliceMut<T>, t: T, u: DVectorSlice<T>, v: DVectorSlice<T>);
}

/// An abstract dynamic system that allows for differentiation of the functions involved.
///
/// This enables implicit integrators to work with an abstract representation of the dynamic
/// system.
pub trait StatelessDifferentiableDynamicSystem<T: Scalar>: StatelessDynamicSystem<T> {
    /// Apply a linear combination of Jacobians to a vector.
    ///
    /// Computes `y = y + (alpha * df/du + beta * df/dv + gamma * M) * x`,
    /// with the Jacobians evaluated at time `t` and the provided `u` and `v` variables.
    fn apply_jacobian_combination(
        &self,
        y: DVectorSliceMut<T>,
        x: DVectorSlice<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
        alpha: Option<T>,
        beta: Option<T>,
        gamma: Option<T>,
    ) -> Result<(), Box<dyn Error>>;

    /// Solve a system consisting of linear combinations of Jacobians.
    ///
    /// Specifically, solve a linear system
    ///
    /// ```ignore
    /// H x = rhs,
    /// ```
    ///
    /// where `H = M + alpha * df/du + beta * df/dv`,
    /// with `H` evaluated at time `t` and the provided `u` and `v` variables.
    fn solve_jacobian_combination(
        &self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
        alpha: Option<T>,
        beta: Option<T>,
    ) -> Result<(), Box<dyn Error>>;
}
