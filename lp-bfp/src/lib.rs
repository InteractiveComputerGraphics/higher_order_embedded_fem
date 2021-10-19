use libc::{c_double, c_int};
use std::convert::TryFrom;
use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};

extern "C" {
    fn lp_bfp_solve_lp(
        x: *mut c_double,
        c: *const c_double,
        a: *const c_double,
        b: *const c_double,
        lb: *const c_double,
        ub: *const c_double,
        num_constraints: c_int,
        num_variables: c_int,
        verbose: bool,
    ) -> c_int;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Verbosity {
    NoVerbose,
    Verbose,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BfpError {
    // Private construction outside of library
    private: (),
}

impl Display for BfpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Unspecified error during solve.")
    }
}

impl Error for BfpError {}

pub fn solve_lp(
    c: &[f64],
    a: &[f64],
    b: &[f64],
    lb: &[f64],
    ub: &[f64],
    verbosity: Verbosity,
) -> Result<Vec<f64>, BfpError> {
    assert_eq!(lb.len(), ub.len(), "Lower and upper bounds must have same length.");
    let num_constraints = b.len();
    let num_variables = lb.len();
    assert_eq!(
        a.len(),
        num_constraints * num_variables,
        "Number of elements in a must be consistent with number of \
                variables and constraints."
    );
    assert_eq!(
        c.len(),
        num_variables,
        "Length of c must be equal to number of variables."
    );

    let verbose = verbosity == Verbosity::Verbose;
    let mut solution = vec![0.0; num_variables];

    // TODO: Make error?
    let num_constraints =
        c_int::try_from(num_constraints).expect("Number of constraints is too large to fit in `int`.");
    let num_variables = c_int::try_from(num_variables).expect("Number of variables is too large to fit in `int`.");

    let error_code = unsafe {
        lp_bfp_solve_lp(
            solution.as_mut_ptr(),
            c.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            lb.as_ptr(),
            ub.as_ptr(),
            num_constraints,
            num_variables,
            verbose,
        )
    };

    if error_code == 0 {
        Ok(solution)
    } else {
        Err(BfpError { private: () })
    }
}

#[cfg(test)]
mod tests {
    use super::{solve_lp, Verbosity};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn solve_basic_lp() {
        // Problem is constructed by constructing b such that A * x0 = b
        let x0 = vec![0.0, 3.0, 7.0, 0.0, 1.0, 0.0];
        let b = vec![28.0, -15.0, 58.0];
        let a = vec![
            1.0, -1.0, 4.0, 2.0, 3.0, 3.0, -5.0, 2.0, -3.0, 2.0, 0.0, 4.0, 3.0, 1.0, 7.0, -3.0, 6.0, 0.0,
        ];
        let lb = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ub = vec![2.0, 4.0, 8.0, 9.0, 2.0, 3.0];
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let x = solve_lp(&c, &a, &b, &lb, &ub, Verbosity::NoVerbose).unwrap();

        // Check bounds
        for i in 0..x.len() {
            assert!(lb[i] <= x[i] && x[i] <= ub[i]);
        }

        // Check equality constraints
        let a = DMatrix::from_row_slice(3, 6, &a);
        let x = DVector::from_column_slice(&x);
        let b = DVector::from_column_slice(&b);
        let c = DVector::from_column_slice(&c);
        let x0 = DVector::from_column_slice(&x0);
        let r = a * &x - b;

        let objective_val = c.dot(&x);
        assert!(r.norm() < 1e-12);
        // TODO: Have a better, more "complete" test?
        assert!(objective_val <= c.dot(&x0) + 1e-6);
    }
}
