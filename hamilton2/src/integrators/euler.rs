use nalgebra::{DVector, DVectorSlice, DVectorSliceMut, RealField, Scalar};
use numeric_literals::replace_float_literals;
use std::error::Error;

use crate::dynamic_system::{DifferentiableDynamicSystem, DynamicSystem};

use crate::calculus::{DifferentiableVectorFunction, VectorFunction};
use crate::newton::{newton_line_search, BacktrackingLineSearch, NewtonSettings};

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn symplectic_euler_step_<T>(
    system: &mut impl DynamicSystem<T>,
    mut u: DVectorSliceMut<T>,
    mut v: DVectorSliceMut<T>,
    mut f: DVectorSliceMut<T>,
    mut dv: DVectorSliceMut<T>,
    t0: T,
    dt: T,
) -> Result<(), Box<dyn Error>>
where
    T: RealField,
{
    // f <- f(t^n, u^n, v^n)
    system.eval_f(
        DVectorSliceMut::from(&mut f),
        t0,
        DVectorSlice::from(&u),
        DVectorSlice::from(&v),
    )?;
    // dv <- M^{-1} * f
    system.apply_inverse_mass_matrix(DVectorSliceMut::from(&mut dv), DVectorSlice::from(f))?;
    // v += dt * dv
    v.axpy(dt, &dv, 1.0);
    // u += dt * v
    u.axpy(dt, &v, 1.0);
    Ok(())
}

pub fn symplectic_euler_step<'a, 'b, 'c, 'd, T>(
    system: &mut impl DynamicSystem<T>,
    u: impl Into<DVectorSliceMut<'a, T>>,
    v: impl Into<DVectorSliceMut<'b, T>>,
    f: impl Into<DVectorSliceMut<'c, T>>,
    dv: impl Into<DVectorSliceMut<'d, T>>,
    t0: T,
    dt: T,
) -> Result<(), Box<dyn Error>>
where
    T: RealField,
{
    symplectic_euler_step_(system, u.into(), v.into(), f.into(), dv.into(), t0, dt)
}

/// Implicit function for Backward-Euler.
///
/// The Backward-Euler discretization is given by
///  M dv - dt * f(t^{n + 1}, u^{n + 1}, v^{n + 1}) = 0
///                                              du = dt * v^{n + 1}
/// with dv = v^{n + 1} - v^n, du = u^{n + 1} - u^n
///
/// Writing w = dv and
///  u^p = u^n + dt * v^n
///  v^{n + 1} = v^n + dv = v^n + w
///  u^{n + 1} = u^n + dt * v^{n + 1} = u^n + dt * v^n + dt * w
///            = u^p + dt * w
///  g(w) = f(t^{n + 1}, u^p + dt * w, v^n + w)
/// We get the non-linear equation system
///  F(w) := M w - dt * g(w) = 0,
/// which we wish to solve with Newton's method. To that end,
/// we need the Jacobian of F. We first note that
///  dg/dw = df/du * du^{n + 1}/dw + df/dv * dv^{n + 1}/dw
///        = df/du * dt * I + df/dv * I
///        = dt * df/du + df/dv
///  dF/dw = M - dt * dg/dw
///        = M - dt * df/dv - (dt)^2 * df/du
struct BackwardEulerImplicitFunction<'a, T, S>
where
    T: Scalar,
{
    dynamic_system: &'a mut S,
    u_p: DVector<T>,
    v_n: DVector<T>,
    t_next: T,
    dt: T,
}

impl<'a, T, S> VectorFunction<T> for BackwardEulerImplicitFunction<'a, T, S>
where
    T: RealField,
    S: DynamicSystem<T>,
{
    fn dimension(&self) -> usize {
        self.u_p.len()
    }

    #[allow(non_snake_case)]
    fn eval_into(&mut self, F: &mut DVectorSliceMut<T>, w: &DVectorSlice<T>) {
        let u = &self.u_p + w * self.dt;
        let v = &self.v_n + w;

        // F <- - dt * f(t^{n+1}, u, v)
        self.dynamic_system
            .eval_f(
                DVectorSliceMut::from(&mut *F),
                self.t_next,
                DVectorSlice::from(&u),
                DVectorSlice::from(&v),
            )
            .unwrap();
        *F *= self.dt * T::from_f64(-1.0).unwrap();
        // F <- F + M * w
        self.dynamic_system
            .apply_mass_matrix(DVectorSliceMut::from(F), DVectorSlice::from(w));
    }
}

impl<'a, T, S> DifferentiableVectorFunction<T> for BackwardEulerImplicitFunction<'a, T, S>
where
    T: RealField,
    S: DifferentiableDynamicSystem<T>,
{
    fn solve_jacobian_system(
        &mut self,
        sol: &mut DVectorSliceMut<T>,
        w: &DVectorSlice<T>,
        rhs: &DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        // TODO: Avoid allocation
        let u = &self.u_p + w * self.dt;
        let v = &self.v_n + w;

        self.dynamic_system
            .set_state(self.t_next, DVectorSlice::from(&u), DVectorSlice::from(&v))?;

        self.dynamic_system
            .init_solve_jacobian_combination(Some(-self.dt * self.dt), Some(-self.dt))?;

        self.dynamic_system
            .solve_jacobian_combination(DVectorSliceMut::from(&mut *sol), DVectorSlice::from(&*rhs))
    }
}

pub struct BackwardEulerSettings<T> {
    /// Tolerance for the residual of the dynamic system.
    ///
    /// More precisely, the tolerance is defined in terms of the inequality
    ///  norm(M dv/dt - f) <= tol
    /// where the derivative dv/dt is approximated numerically and norm( ) is the Euclidean 2-norm.
    pub tolerance: T,
    pub max_newton_iter: Option<usize>,
}

#[allow(non_snake_case)]
/// Performs one step of the Backward-Euler integrator on the provided dynamic system.
pub fn backward_euler_step<'a, 'b, T>(
    system: &mut impl DifferentiableDynamicSystem<T>,
    u: impl Into<DVectorSliceMut<'a, T>>,
    v: impl Into<DVectorSliceMut<'b, T>>,
    t0: T,
    dt: T,
    settings: BackwardEulerSettings<T>,
) -> Result<usize, Box<dyn Error>>
where
    T: RealField,
{
    let mut u = u.into();
    let mut v = v.into();

    // TODO: Avoid heap-allocating vectors
    let dim = u.len();
    let t_next = t0 + dt;
    let v_n = v.clone_owned();
    let u_p = &u + &v * dt;

    let F = BackwardEulerImplicitFunction {
        dynamic_system: system,
        u_p,
        v_n,
        t_next,
        dt,
    };

    // TODO: Avoid allocation
    let mut w = DVector::zeros(dim);
    let mut f = DVector::zeros(dim);
    let mut dw = DVector::zeros(dim);

    let settings = NewtonSettings {
        tolerance: dt * settings.tolerance,
        max_iterations: settings.max_newton_iter,
    };

    let mut line_search = BacktrackingLineSearch {};
    let iter = newton_line_search(F, &mut w, &mut f, &mut dw, settings, &mut line_search)?;
    v += w;
    u += &v * dt;

    Ok(iter)
}

#[cfg(test)]
mod tests {
    use crate::calculus::{approximate_jacobian, DifferentiableVectorFunction};
    use crate::dynamic_system::{IntoStateful, StatelessDifferentiableDynamicSystem, StatelessDynamicSystem};
    use crate::integrators::euler::BackwardEulerImplicitFunction;

    use nalgebra::{DMatrix, DVector, DVectorSlice, DVectorSliceMut};
    use std::error::Error;

    /// Mock dynamic system defined by
    ///
    /// M = some invertible matrix
    /// f(t, u, v) = t * dot(v, v) * u
    struct MockNonlinearDynamicSystem {
        mass: DMatrix<f64>,
    }

    impl StatelessDynamicSystem<f64> for MockNonlinearDynamicSystem {
        fn apply_mass_matrix(&self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) {
            assert!(self.mass.is_square());
            y.gemv(1.0, &self.mass, &x, 1.0);
        }

        fn apply_inverse_mass_matrix(
            &self,
            mut y: DVectorSliceMut<f64>,
            x: DVectorSlice<f64>,
        ) -> Result<(), Box<dyn Error>> {
            let lu = self.mass.clone().lu();
            y.copy_from(&lu.solve(&x).ok_or("LU decomposition failed")?);
            Ok(())
        }

        fn eval_f(&self, mut f: DVectorSliceMut<f64>, t: f64, u: DVectorSlice<f64>, v: DVectorSlice<f64>) {
            let v_dot_v = v.dot(&v);
            f.copy_from(&(t * v_dot_v * u));
        }
    }

    impl StatelessDifferentiableDynamicSystem<f64> for MockNonlinearDynamicSystem {
        fn apply_jacobian_combination(
            &self,
            _y: DVectorSliceMut<f64>,
            _x: DVectorSlice<f64>,
            _t: f64,
            _u: DVectorSlice<f64>,
            _v: DVectorSlice<f64>,
            _alpha: Option<f64>,
            _beta: Option<f64>,
            _gamma: Option<f64>,
        ) -> Result<(), Box<dyn Error>> {
            unimplemented!();
        }

        fn solve_jacobian_combination(
            &self,
            mut sol: DVectorSliceMut<f64>,
            rhs: DVectorSlice<f64>,
            t: f64,
            u: DVectorSlice<f64>,
            v: DVectorSlice<f64>,
            alpha: Option<f64>,
            beta: Option<f64>,
        ) -> Result<(), Box<dyn Error>> {
            let alpha = alpha.unwrap_or(0.0);
            let beta = beta.unwrap_or(0.0);
            // We have
            //  f(t, u, v) = t * dot(v, v) * u
            // so
            //  df/du = t * dot(v, v) * I
            //  df/dv = 2 * t * u * v^T
            let dim = u.len();
            let df_du = t * v.dot(&v) * DMatrix::identity(dim, dim);
            let df_dv = 2.0 * t * u * v.transpose();

            let df_comb = &self.mass + alpha * df_du + beta * df_dv;
            let lu = df_comb.lu();
            sol.copy_from(&(lu.solve(&rhs).ok_or("LU decomposition failed.")?));
            Ok(())
        }
    }

    #[allow(non_snake_case)]
    #[test]
    fn backward_euler_implicit_function_jacobian() {
        // Test that the implicit function defined by
        // Backward Euler computes the correct Jacobian by comparing it
        // with a numerical approximation of its Jacobian

        let dim = 5;
        let t0 = 2.0;
        let dt = 2.5;

        let mass_elements = (0..(dim * dim)).into_iter().map(|i| i as f64);
        let mass = DMatrix::from_iterator(dim, dim, mass_elements);

        let mut dynamic_system = MockNonlinearDynamicSystem { mass }.into_stateful();

        let mut F = BackwardEulerImplicitFunction {
            dynamic_system: &mut dynamic_system,
            u_p: DVector::zeros(dim),
            v_n: DVector::zeros(dim),
            t_next: t0 + dt,
            dt,
        };

        let w_elements = (0..).into_iter().map(|i| i as f64);
        let w = DVector::from_iterator(dim, w_elements.clone().take(dim));
        let rhs = DVector::from_iterator(dim, w_elements.skip(dim).take(dim));

        // We can not directly compute the Jacobian of F(w), since we only
        // abstractly have access to the action of the inverse Jacobian.
        // However, assuming that the Jacobian is invertible, we note that if A and B
        // are both invertible, and
        //  Ay = b, By = b,
        // then A = B provided that b != 0. Thus, it suffices to take some arbitrary
        // b and make sure that we get the same solution.

        let mut y_F = DVector::zeros(dim);
        F.solve_jacobian_system(
            &mut DVectorSliceMut::from(&mut y_F),
            &DVectorSlice::from(&w),
            &DVectorSlice::from(&rhs),
        )
        .unwrap();

        let j_approx = approximate_jacobian(&mut F, &w.clone_owned(), &1e-6);
        let y_approx = j_approx.lu().solve(&rhs).unwrap();

        assert!(y_F.relative_eq(&y_approx, 1e-6, 1e-12));
    }
}
