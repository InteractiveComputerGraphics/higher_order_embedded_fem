use nalgebra::{DVector, DVectorSlice, DVectorSliceMut, Scalar};
use std::error::Error;

use crate::dynamic_system::mutable::{MutableDifferentiableDynamicSystem, MutableDynamicSystem};
use crate::dynamic_system::stateless::{StatelessDifferentiableDynamicSystem, StatelessDynamicSystem};

/// Wrapper for a `StatelessDynamicSystem` to use it as a `DynamicSystem`, use the `IntoStateful` trait to construct.
pub struct StatefulWrapper<T: Scalar, S> {
    system: S,
    state: Option<State<T>>,
    jacobian_apply_params: Option<(Option<T>, Option<T>, Option<T>)>,
    jacobian_solve_params: Option<(Option<T>, Option<T>)>,
}

/// Allows to view a stateless dynamic system as a stateful system for a simplified implementation.
pub trait IntoStateful<T: Scalar, S> {
    fn into_stateful(self) -> StatefulWrapper<T, S>;
}

impl<T, S> IntoStateful<T, S> for S
where
    T: Scalar,
    S: StatelessDynamicSystem<T>,
{
    fn into_stateful(self) -> StatefulWrapper<T, S> {
        StatefulWrapper {
            system: self,
            state: None,
            jacobian_apply_params: None,
            jacobian_solve_params: None,
        }
    }
}

struct State<T: Scalar> {
    t: T,
    u: DVector<T>,
    v: DVector<T>,
}

impl<T, S> MutableDynamicSystem<T> for StatefulWrapper<T, S>
where
    T: Scalar,
    S: StatelessDynamicSystem<T>,
{
    fn apply_mass_matrix(&mut self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) {
        self.system.apply_mass_matrix(y, x)
    }

    fn apply_inverse_mass_matrix(
        &mut self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        self.system.apply_inverse_mass_matrix(sol, rhs)
    }

    fn eval_f(
        &mut self,
        f: DVectorSliceMut<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        self.system.eval_f(f, t, u, v);
        Ok(())
    }
}

impl<T, S> MutableDifferentiableDynamicSystem<T> for StatefulWrapper<T, S>
where
    T: Scalar,
    S: StatelessDifferentiableDynamicSystem<T>,
{
    fn set_state(&mut self, t: T, u: DVectorSlice<T>, v: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        self.state = Some(State {
            t,
            u: u.clone_owned(),
            v: v.clone_owned(),
        });

        Ok(())
    }

    fn init_apply_jacobian_combination(
        &mut self,
        alpha: Option<T>,
        beta: Option<T>,
        gamma: Option<T>,
    ) -> Result<(), Box<dyn Error>> {
        self.jacobian_apply_params = Some((alpha, beta, gamma));
        Ok(())
    }

    fn init_solve_jacobian_combination(&mut self, alpha: Option<T>, beta: Option<T>) -> Result<(), Box<dyn Error>> {
        self.jacobian_solve_params = Some((alpha, beta));
        Ok(())
    }

    fn apply_jacobian_combination(&mut self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        match (self.state.as_ref(), self.jacobian_apply_params.as_ref()) {
            (Some(state), Some((alpha, beta, gamma))) => self.system.apply_jacobian_combination(
                y,
                x,
                state.t.clone(),
                DVectorSlice::from(&state.u),
                DVectorSlice::from(&state.v),
                alpha.clone(),
                beta.clone(),
                gamma.clone(),
            ),
            (None, Some(_)) => Err(Box::<dyn Error>::from("no state set")),
            (Some(_), None) => Err(Box::<dyn Error>::from("no jacobian apply params set")),
            (None, None) => Err(Box::<dyn Error>::from("no state and no jacobian apply params set")),
        }
    }

    fn solve_jacobian_combination(
        &mut self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        match (self.state.as_ref(), self.jacobian_solve_params.as_ref()) {
            (Some(state), Some((alpha, beta))) => self.system.solve_jacobian_combination(
                sol,
                rhs,
                state.t.clone(),
                DVectorSlice::from(&state.u),
                DVectorSlice::from(&state.v),
                alpha.clone(),
                beta.clone(),
            ),
            (None, Some(_)) => Err(Box::<dyn Error>::from("no state set")),
            (Some(_), None) => Err(Box::<dyn Error>::from("no jacobian solve params set")),
            (None, None) => Err(Box::<dyn Error>::from("no state and no jacobian solve params set")),
        }
    }
}

/*
use std::marker::PhantomData;

use crate::dynamic_system::DifferentiableDynamicSystemSnapshot;
use crate::dynamic_system::{
    DynamicSystemSnapshot, Operator, StatefulDynamicSystem, StatelessDifferentiableDynamicSystem,
    StatelessDynamicSystem,
};

/// Allows to view a stateful dynamic system as a stateless system for a simplified interface.
pub struct StatelessView<'a, T, O, S> {
    system: &'a S,
    _phantom: PhantomData<(T, O)>,
}

pub trait AsStateless<T, O, S> {
    fn as_stateless(&self) -> StatelessView<T, O, S>;
}

impl<T, O, S> AsStateless<T, O, S> for S
where
    T: Scalar,
    O: DynamicSystemSnapshot<T>,
    S: StatefulDynamicSystem<T, Snapshot = O>,
{
    fn as_stateless(&self) -> StatelessView<T, O, S> {
        StatelessView {
            system: self,
            _phantom: PhantomData,
        }
    }
}

/// Allows to view a stateless dynamic system as a stateful system for a simplified implementation.
pub struct StatefulView<'view, T, S: 'view> {
    system: &'view S,
    _phantom: PhantomData<T>,
}

pub trait AsStateful<'view, 'sys: 'view, T, S: 'view> {
    fn as_stateful(&'sys self) -> StatefulView<'view, T, S>;
}

impl<'view, 'sys: 'view, T, S> AsStateful<'view, 'sys, T, S> for S
where
    T: Scalar,
    S: StatelessDynamicSystem<T> + 'view,
{
    fn as_stateful(&'sys self) -> StatefulView<'view, T, S> {
        StatefulView {
            system: self,
            _phantom: PhantomData,
        }
    }
}

// FIXME: Ideally, u and v would be slices
pub struct NaiveSnapshot<'snap, T: Scalar, S: 'snap> {
    system: &'snap S,
    t: T,
    u: DVector<T>,
    v: DVector<T>,
}

impl<'snap, T, S> DynamicSystemSnapshot<T> for NaiveSnapshot<'snap, T, S>
where
    T: Scalar,
    S: StatelessDynamicSystem<T> + 'snap,
{
    fn eval_f(&self, f: DVectorSliceMut<T>) {
        self.system.eval_f(
            f,
            self.t.clone(),
            DVectorSlice::from(&self.u),
            DVectorSlice::from(&self.v),
        );
    }
}

pub struct NaiveApplyJacobianOperator<'op, 'snap: 'op, T: Scalar, S> {
    snapshot: &'op NaiveSnapshot<'snap, T, S>,
    alpha: T,
    beta: T,
    gamma: T,
}

impl<'op, 'snap: 'op, T, S> Operator<T> for NaiveApplyJacobianOperator<'op, 'snap, T, S>
where
    T: Scalar,
    S: StatelessDifferentiableDynamicSystem<T> + 'snap,
{
    fn apply(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        self.snapshot.system.apply_jacobian_combination(
            y,
            x,
            self.snapshot.t.clone(),
            DVectorSlice::from(&self.snapshot.u),
            DVectorSlice::from(&self.snapshot.v),
            self.alpha.clone(),
            self.beta.clone(),
            self.gamma.clone(),
        )
    }
}

pub struct NaiveSolveJacobianOperator<'op, 'snap: 'op, T: Scalar, S> {
    snapshot: &'op NaiveSnapshot<'snap, T, S>,
    alpha: T,
    beta: T,
}

impl<'op, 'snap: 'op, T, S> Operator<T> for NaiveSolveJacobianOperator<'op, 'snap, T, S>
where
    T: Scalar,
    S: StatelessDifferentiableDynamicSystem<T> + 'snap,
{
    fn apply(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        self.snapshot.system.solve_jacobian_combination(
            y,
            x,
            self.snapshot.t.clone(),
            DVectorSlice::from(&self.snapshot.u),
            DVectorSlice::from(&self.snapshot.v),
            self.alpha.clone(),
            self.beta.clone(),
        )
    }
}

impl<'op, 'snap: 'op, T, S> DifferentiableDynamicSystemSnapshot<'op, T>
    for NaiveSnapshot<'snap, T, S>
where
    T: Scalar,
    S: StatelessDifferentiableDynamicSystem<T> + 'snap,
{
    type ApplyJacobianCombinationOperator = NaiveApplyJacobianOperator<'op, 'snap, T, S>;
    type SolveJacobianCombinationOperator = NaiveSolveJacobianOperator<'op, 'snap, T, S>;

    fn build_apply_jacobian_combination_operator(
        &'op self,
        alpha: T,
        beta: T,
        gamma: T,
    ) -> Result<Self::ApplyJacobianCombinationOperator, Box<dyn Error>> {
        Ok(NaiveApplyJacobianOperator {
            snapshot: self,
            alpha: alpha.clone(),
            beta: beta.clone(),
            gamma: gamma.clone(),
        })
    }

    fn build_solve_jacobian_combination_operator(
        &'op self,
        alpha: T,
        beta: T,
    ) -> Result<Self::SolveJacobianCombinationOperator, Box<dyn Error>> {
        Ok(NaiveSolveJacobianOperator {
            snapshot: self,
            alpha: alpha.clone(),
            beta: beta.clone(),
        })
    }
}

impl<'snap, 'view: 'snap, T, S> StatefulDynamicSystem<'snap, T> for StatefulView<'view, T, S>
where
    T: Scalar,
    S: StatelessDynamicSystem<T> + 'view,
{
    type Snapshot = NaiveSnapshot<'snap, T, S>;

    fn apply_mass_matrix(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) {
        self.system.apply_mass_matrix(y, x);
    }

    fn apply_inverse_mass_matrix(
        &self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        self.system.apply_inverse_mass_matrix(sol, rhs)
    }

    fn with_state(
        &self,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
    ) -> Result<Self::Snapshot, Box<dyn Error>> {
        Ok(NaiveSnapshot {
            system: self.system,
            t: t.clone(),
            u: u.clone_owned(),
            v: v.clone_owned(),
        })
    }
}
*/

/*
impl<'a, T, S, O> StatelessDynamicSystem<T> for StatelessView<'a, T, O, S>
where
    T: Scalar,
    S: StatefulDynamicSystem<T, Snapshot = O>,
    O: DynamicSystemSnapshot<T>,
{
    fn apply_mass_matrix(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) {
        self.system.apply_mass_matrix(y, x);
    }

    fn apply_inverse_mass_matrix(
        &self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        self.system.apply_inverse_mass_matrix(sol, rhs)
    }

    fn eval_f(
        &self,
        f: DVectorSliceMut<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        self.system.with_state(t, u, v)?.eval_f(f);
        Ok(())
    }
}

impl<'a, T, S, O> StatelessDifferentiableDynamicSystem<T> for StatelessView<'a, T, O, S>
where
    T: Scalar,
    S: StatefulDynamicSystem<T, Snapshot = O>,
    O: DifferentiableDynamicSystemSnapshot<T>,
{
    fn apply_jacobian_combination(
        &self,
        y: DVectorSliceMut<T>,
        x: DVectorSlice<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
        alpha: T,
        beta: T,
        gamma: T,
    ) -> Result<(), Box<dyn Error>> {
        self.system
            .with_state(t, u, v)?
            .build_apply_jacobian_combination_operator(alpha, beta, gamma)?
            .apply(y, x)
    }

    fn solve_jacobian_combination(
        &self,
        sol: DVectorSliceMut<T>,
        rhs: DVectorSlice<T>,
        t: T,
        u: DVectorSlice<T>,
        v: DVectorSlice<T>,
        alpha: T,
        beta: T,
    ) -> Result<(), Box<dyn Error>> {
        self.system
            .with_state(t, u, v)?
            .build_solve_jacobian_combination_operator(alpha, beta)?
            .apply(sol, rhs)
    }
}
*/
