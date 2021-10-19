/*
use nalgebra::{DVectorSlice, DVectorSliceMut, Scalar};
use std::error::Error;

use crate::dynamic_system::Operator;

pub trait StatefulDynamicSystem<'snap, T: Scalar> {
    type Snapshot: DynamicSystemSnapshot<T> + 'snap;

    fn apply_mass_matrix(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>);

    fn apply_inverse_mass_matrix(
        &self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>>;

    fn with_state(
        &self,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
    ) -> Result<Self::Snapshot, Box<dyn Error>>;
}

pub trait DynamicSystemSnapshot<T: Scalar> {
    fn eval_f(&self, f: DVectorSliceMut<T>);
}

pub trait DifferentiableDynamicSystemSnapshot<'op, T: Scalar>: DynamicSystemSnapshot<T> {
    type ApplyJacobianCombinationOperator: Operator<T> + 'op;
    type SolveJacobianCombinationOperator: Operator<T> + 'op;

    fn build_apply_jacobian_combination_operator(
        &'op self,
        alpha: T,
        beta: T,
        gamma: T,
    ) -> Result<Self::ApplyJacobianCombinationOperator, Box<dyn Error>>;

    fn build_solve_jacobian_combination_operator(
        &'op self,
        alpha: T,
        beta: T,
    ) -> Result<Self::SolveJacobianCombinationOperator, Box<dyn Error>>;
}
*/
