use crate::embedding::LpSolver;
use crate::nalgebra::{DMatrix, DVector};
use lp_bfp::{solve_lp, Verbosity};
use std::error::Error;
use std::f64;

/// A basic feasible point solver powered by Google's GLOP LP solver.
#[derive(Debug, Clone)]
pub struct GlopSolver {
    verbosity: Verbosity,
}

impl GlopSolver {
    pub fn new() -> Self {
        Self {
            verbosity: Verbosity::NoVerbose,
        }
    }

    pub fn new_verbose() -> Self {
        Self {
            verbosity: Verbosity::Verbose,
        }
    }
}

impl LpSolver<f64> for GlopSolver {
    fn solve_lp(
        &self,
        c: &DVector<f64>,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        lb: &[Option<f64>],
        ub: &[Option<f64>],
    ) -> Result<DVector<f64>, Box<dyn Error + Sync + Send>> {
        let a_elements_row_major: Vec<_> = a.transpose().iter().copied().collect();
        let lb: Vec<_> = lb
            .iter()
            .copied()
            .map(|lb_i| lb_i.unwrap_or(-f64::INFINITY))
            .collect();
        let ub: Vec<_> = ub
            .iter()
            .copied()
            .map(|ub_i| ub_i.unwrap_or(f64::INFINITY))
            .collect();
        let bfp = solve_lp(
            c.as_slice(),
            &a_elements_row_major,
            b.as_slice(),
            &lb,
            &ub,
            self.verbosity,
        )?;
        Ok(DVector::from_column_slice(&bfp))
    }
}
