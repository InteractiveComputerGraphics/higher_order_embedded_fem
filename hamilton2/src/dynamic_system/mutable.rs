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
pub trait MutableDynamicSystem<T: Scalar> {
    /// Apply the mass matrix `M` to the vector `x` and accumulate the result in `y`,
    /// yielding `y = y + Mx`.
    fn apply_mass_matrix(&mut self, y: DVectorSliceMut<T>, x: DVectorSlice<T>);

    fn apply_inverse_mass_matrix(
        &mut self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>>;

    fn eval_f(
        &mut self,
        f: DVectorSliceMut<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>>;
}

/// An abstract dynamic system that allows for differentiation of the functions involved.
///
/// This enables implicit integrators to work with an abstract representation of the dynamic
/// system.
pub trait MutableDifferentiableDynamicSystem<T: Scalar>: MutableDynamicSystem<T> {
    /// Internally stores the state, pre-computes stiffness matrix, etc. for calls to Jacobian combination functions
    fn set_state(&mut self, t: T, u: DVectorSlice<T>, v: DVectorSlice<T>) -> Result<(), Box<dyn Error>>;

    /// Pre-computes the Jacobian linear combination to apply it to any vector using `apply_jacobian_combination`
    fn init_apply_jacobian_combination(
        &mut self,
        alpha: Option<T>,
        beta: Option<T>,
        gamma: Option<T>,
    ) -> Result<(), Box<dyn Error>>;

    /// Performs factorization for subsequent calls to `solve_jacobian_combination`
    fn init_solve_jacobian_combination(&mut self, alpha: Option<T>, beta: Option<T>) -> Result<(), Box<dyn Error>>;

    /// Apply a linear combination of the system Jacobians to a vector.
    ///
    /// Computes `y = y + (alpha * df/du + beta * df/dv + gamma * M) * x`,
    /// with the Jacobians evaluated at time `t` and the provided `u` and `v` variables
    /// that were set using `set_state`.
    fn apply_jacobian_combination(&mut self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>>;

    /// Solve a system consisting of linear combinations of Jacobians.
    ///
    /// Specifically, solve a linear system
    ///
    /// ```ignore
    /// H x = rhs,
    /// ```
    ///
    /// where `H = M + alpha * df/du + beta * df/dv`,
    /// with `H` evaluated at time `t` and the provided `u` and `v` variables that were set
    /// using `set_state`.
    fn solve_jacobian_combination(
        &mut self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>>;
}
