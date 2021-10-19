use hamilton2::dynamic_system::{IntoStateful, StatelessDifferentiableDynamicSystem, StatelessDynamicSystem};
use hamilton2::integrators::{backward_euler_step, symplectic_euler_step, BackwardEulerSettings};
use nalgebra::{DVector, DVectorSlice, DVectorSliceMut, Vector3};
use std::error::Error;

use crate::assert_approx_matrix_eq;

/// Define the (mock) dynamic system
///  M dv/dt + f(t, u, v) = 0
///             du/dt - v = 0,
/// with
///  M = a * I
///  f(t, u, v) = b * t * ones + c * u + d * v
pub struct MockDynamicSystem {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl StatelessDynamicSystem<f64> for MockDynamicSystem {
    fn apply_mass_matrix(&self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) {
        y.axpy(self.a, &x, 1.0);
    }

    fn apply_inverse_mass_matrix(
        &self,
        mut y: DVectorSliceMut<f64>,
        x: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        y.copy_from(&(x / self.a));
        Ok(())
    }

    fn eval_f(&self, mut f: DVectorSliceMut<f64>, t: f64, u: DVectorSlice<f64>, v: DVectorSlice<f64>) {
        let ones = DVector::repeat(u.len(), 1.0);
        f.copy_from(&(self.b * t * ones + self.c * u + self.d * v));
    }
}

impl StatelessDifferentiableDynamicSystem<f64> for MockDynamicSystem {
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
        _t: f64,
        _u: DVectorSlice<f64>,
        _v: DVectorSlice<f64>,
        alpha: Option<f64>,
        beta: Option<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let alpha = alpha.unwrap_or(0.0);
        let beta = beta.unwrap_or(0.0);
        let g = self.a + alpha * self.c + beta * self.d;
        sol.copy_from(&(rhs / g));
        Ok(())
    }
}

#[test]
fn symplectic_euler_step_mock() {
    let (a, b, c, d) = (2.0, 3.0, 4.0, 5.0);
    let mut system = MockDynamicSystem { a, b, c, d }.into_stateful();
    let t0 = 2.0;
    let dt = 0.5;

    let mut u = Vector3::new(1.0, 2.0, 3.0);
    let mut v = Vector3::new(4.0, 5.0, 6.0);

    let mut f = Vector3::zeros();
    let mut dv = Vector3::zeros();

    // Compute expected values
    let v_next_expected = &v + (dt / a) * (b * t0 * Vector3::repeat(1.0) + c * &u + d * &v);
    let u_next_expected = &u + dt * &v_next_expected;

    symplectic_euler_step(&mut system, &mut u, &mut v, &mut f, &mut dv, t0, dt).unwrap();

    assert_approx_matrix_eq!(v, v_next_expected, abstol = 1e-8);
    assert_approx_matrix_eq!(u, u_next_expected, abstol = 1e-8);
}

#[test]
fn backward_euler_step_mock() {
    let (a, b, c, d) = (2.0, 3.0, 4.0, 5.0);
    let mut system = MockDynamicSystem { a, b, c, d }.into_stateful();
    let t0 = 2.0;
    let dt = 0.5;

    let mut u = Vector3::new(1.0, 2.0, 3.0);
    let mut v = Vector3::new(4.0, 5.0, 6.0);

    let settings = BackwardEulerSettings {
        max_newton_iter: Some(100),
        tolerance: 1e-8,
    };

    let (u_next_expected, v_next_expected) = {
        let h = a - dt * d - dt * dt * c;
        let t = t0 + dt;
        let ones = Vector3::repeat(1.0);
        let v_next_expected = (a * &v + dt * (b * t * ones + c * &u)) / h;
        let u_next_expected = &u + dt * &v_next_expected;
        (u_next_expected, v_next_expected)
    };

    backward_euler_step(&mut system, &mut u, &mut v, t0, dt, settings).unwrap();

    assert_approx_matrix_eq!(v, v_next_expected, abstol = 1e-6);
    assert_approx_matrix_eq!(u, u_next_expected, abstol = 1e-6);
}
