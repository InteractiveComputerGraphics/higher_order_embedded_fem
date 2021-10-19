use crate::embedding::{EmbeddedModel, EmbeddedQuadrature};
use crate::geometry::AxisAlignedBoundingBox;
use crate::quadrature::{Quadrature, QuadraturePair};
use itertools::Itertools;
use log::debug;
use nalgebra::allocator::Allocator;
use nalgebra::{DMatrix, DVector, DefaultAllocator, DimName, Point, RealField, Scalar, VectorN};
use num::integer::binomial;
use numeric_literals::replace_float_literals;
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;
use std::iter::repeat;

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn chebyshev_1d<T>(x: T, n: usize) -> T
where
    T: RealField,
{
    if n == 0 {
        1.0
    } else if n == 1 {
        x
    } else {
        2.0 * x * chebyshev_1d(x, n - 1) - chebyshev_1d(x, n - 2)
    }
}

struct LinearSystem<T: Scalar> {
    matrix: DMatrix<T>,
    rhs: DVector<T>,
}

/// Compute the dimension of the space of polynomials in `d` variables with degree at most `n`.
///
/// Assumes that `d >= 1`. Panics otherwise.
fn polynomial_space_dim(n: usize, d: usize) -> usize {
    assert!(d >= 1, "d must be 1 or greater.");
    (0..=n).map(|k| binomial(k + d - 1, k)).sum()
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn build_moment_fitting_system<T, D>(weights: &[T], points: &[VectorN<T, D>], strength: usize) -> LinearSystem<T>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    // Determine scaling for basis functions to improve conditioning
    // TODO: Fix this, it's stupid
    let points_as_vectors: Vec<_> = points.iter().map(|p| Point::from(p.clone())).collect();
    let bounds = AxisAlignedBoundingBox::from_points(&points_as_vectors).expect("TODO: Handle case of no points");
    let center = bounds.center();
    let l = bounds.extents() / 2.0;

    // Alternative scaling based on average of point data below
    // (some basic testing seems to suggest that the bounding box works just as well)
    //    // Scale basis functions to data
    //    let num_points = T::from_usize(points.len()).unwrap();
    //    let center = points.iter().fold(Vector2::zeros(), |p1, p2| p1 + p2) / num_points;
    //    let l_x = points.iter()
    //        .map(|p| p.x)
    //        .fold(T::zero(), |l, x| T::max(l, (x - center.x).abs()));
    //    let l_y = points.iter()
    //        .map(|p| p.y)
    //        .fold(T::zero(), |l, y| T::max(l, (y - center.y).abs()));

    // alpha and beta are coefficients that scale and translate the 1d basis functions
    // to better fit the problem data
    // Note: We generally rescale by the size of the bounding box in each coordinate direction.
    // However, if a direction is degenerate, we anyway cannot hope to obtain higher than
    // 0-th order accuracy in that direction, so by setting the scale to zero
    // we will effectively transform any non-constant basis function to a constant function
    let alpha = l.map(|l_i| if l_i != T::zero() { T::one() / l_i } else { T::zero() });
    let beta = VectorN::repeat(T::one()) - alpha.component_mul(&(center.coords + l));

    let mut matrix_elements = Vec::new();
    let mut rhs_elements = Vec::new();

    // Loop over tensor products of axis dimensions i.e.
    // (i, j) with i + j <= strength for 2D,
    // (i, j, k) with i + j + k <= strength for 3D,
    // TODO: lots of implicit allocations here
    for orders in repeat(0..=strength)
        .take(D::dim())
        .multi_cartesian_product()
    {
        if orders.iter().sum::<usize>() <= strength {
            let polynomial = |point: &VectorN<T, D>| {
                let mut val = T::one();
                for d in 0..D::dim() {
                    val *= chebyshev_1d(alpha[d] * point[d] + beta[d], orders[d])
                }
                val
            };

            // Evaluate matrix elements
            let row_iter = points.iter().map(|point| polynomial(point));
            matrix_elements.extend(row_iter);

            let polynomial_integral: T = weights
                .iter()
                .cloned()
                .zip(points.iter())
                .map(|(w, point)| w * polynomial(point))
                .fold(T::zero(), |sum, next_element| sum + next_element);
            rhs_elements.push(polynomial_integral);
        }
    }

    let num_rows = rhs_elements.len();
    assert_eq!(matrix_elements.len() % num_rows, 0);
    let num_cols = matrix_elements.len() / num_rows;
    assert_eq!(num_rows, (polynomial_space_dim(strength, D::dim())));

    LinearSystem {
        matrix: DMatrix::from_row_slice(num_rows, num_cols, &matrix_elements),
        rhs: DVector::from_column_slice(&rhs_elements),
    }
}

pub trait LpSolver<T: Scalar> {
    /// Solve the designated LP, or return an error.
    ///
    /// The LP is described as follows. Find x that solves:
    ///   min c^T x
    ///   s.t. Ax = b
    ///       lb <= x <= ub
    fn solve_lp(
        &self,
        c: &DVector<T>,
        a: &DMatrix<T>,
        b: &DVector<T>,
        lb: &[Option<T>],
        ub: &[Option<T>],
    ) -> Result<DVector<T>, Box<dyn Error + Sync + Send>>;
}

impl<T, X> LpSolver<T> for &X
where
    T: Scalar,
    X: LpSolver<T>,
{
    fn solve_lp(
        &self,
        c: &DVector<T>,
        a: &DMatrix<T>,
        b: &DVector<T>,
        lb: &[Option<T>],
        ub: &[Option<T>],
    ) -> Result<DVector<T>, Box<dyn Error + Sync + Send>> {
        <X as LpSolver<T>>::solve_lp(self, c, a, b, lb, ub)
    }
}

pub fn optimize_quadrature<T, D>(
    quadrature: impl Quadrature<T, D>,
    polynomial_strength: usize,
    lp_solver: &impl LpSolver<T>,
) -> Result<QuadraturePair<T, D>, Box<dyn Error + Sync + Send>>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    let LinearSystem { matrix: p, rhs: _b } =
        build_moment_fitting_system(quadrature.weights(), quadrature.points(), polynomial_strength);

    if quadrature.weights().len() <= p.nrows() {
        // We can't improve the solution, so just return the current quadrature
        Ok((quadrature.weights().to_vec(), quadrature.points().to_vec()))
    } else {
        // TODO: Consider recomposing matrix with SVD to remove small singular values,
        // since depending on point distribution, the resulting system may be
        // poorly conditioned
        let lb = vec![Some(T::zero()); p.ncols()];
        let ub = vec![None; p.ncols()];

        let w0 = DVector::from_column_slice(&quadrature.weights());

        debug!("Number of weights in quadrature before simplification: {}", w0.len());
        debug!("Size of polynomial basis: {}", p.nrows());

        // P w = P w0 is the original set of constraints.
        // Take first r rows of V^T, where r is the rank of the matrix. V^T then
        // is a basis for the null space of P. Thus we may replace Pw = P w0
        // with V^T w = V^T w0. Since V^T has orthogonal rows, it is much easier for the
        // solver to work with.
        let v_r_t = {
            // TODO: Use more accurate SVD?
            let threshold = T::default_epsilon();
            let p_svd = p.svd(false, true);
            let max_svd = p_svd.singular_values.max();
            let idx_to_remove: Vec<_> = p_svd
                .singular_values
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, val)| *val <= threshold * max_svd)
                .map(|(i, _)| i)
                .collect();
            p_svd.v_t.unwrap().remove_rows_at(&idx_to_remove)
        };

        // TODO: This should not print, we need a different mechanism to report this
        if quadrature.weights().iter().any(|w_i| w_i < &T::zero()) {
            eprintln!("Negative quadrature weights detected in optimize_quadrature()");
        }

        let b_v = &v_r_t * &w0;
        let c = DVector::zeros(w0.len());

        let w_bfp = lp_solver.solve_lp(&c, &v_r_t, &b_v, &lb, &ub)?;

        // TODO: Check residual etc???
        let (new_weights, new_points): (Vec<_>, Vec<_>) = w_bfp
            .iter()
            .copied()
            .zip(quadrature.points().iter().cloned())
            // TODO: Enable threshold filtering...?
            .filter(|(w, _)| w > &T::zero())
            .unzip();
        debug!("Number of weights in simplified quadrature: {}", new_weights.len());
        Ok((new_weights, new_points))
    }
}

#[derive(Debug)]
pub struct QuadratureOptimizationError {
    pub interface_connectivity_index: usize,
    pub optimization_error: Box<dyn Error + Sync + Send>,
}

impl fmt::Display for QuadratureOptimizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quadrature optimization error in interface connectivity {}. Error: {}",
            self.interface_connectivity_index, self.optimization_error
        )
    }
}

impl<T, D> EmbeddedQuadrature<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    VectorN<T, D>: Sync + Send,
{
    pub fn simplified(
        &self,
        polynomial_strength: usize,
        lp_solver: &(impl Sync + LpSolver<T>),
    ) -> Result<Self, QuadratureOptimizationError> {
        simplify_quadrature(polynomial_strength, self, lp_solver)
    }
}

pub fn simplify_quadrature<T, D>(
    polynomial_strength: usize,
    quadrature: &EmbeddedQuadrature<T, D>,
    lp_solver: &(impl Sync + LpSolver<T>),
) -> Result<EmbeddedQuadrature<T, D>, QuadratureOptimizationError>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    VectorN<T, D>: Sync + Send,
{
    let mut new_quadratures: Vec<_> = quadrature
        .interface_quadratures()
        .iter()
        .map(|q| (q.weights().to_vec(), q.points().to_vec()))
        .collect();

    let solver = &lp_solver;
    new_quadratures
        .par_iter_mut()
        .enumerate()
        .map(|(idx, current_quadrature)| {
            let new_quadrature = optimize_quadrature(&*current_quadrature, polynomial_strength, solver);
            match new_quadrature {
                Ok(new_quadrature) => {
                    *current_quadrature = new_quadrature;
                    Ok(())
                }
                Err(err) => Err(QuadratureOptimizationError {
                    interface_connectivity_index: idx,
                    optimization_error: err,
                }),
            }
        })
        .collect::<Result<Vec<()>, _>>()?;

    let interior_quadrature = (
        quadrature.interior_quadrature().weights().to_vec(),
        quadrature.interior_quadrature().points().to_vec(),
    );

    Ok(EmbeddedQuadrature::from_interior_and_interface(
        interior_quadrature,
        new_quadratures,
    ))
}

impl Error for QuadratureOptimizationError {}

impl<T, D, C> EmbeddedModel<T, D, C>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    VectorN<T, D>: Sync + Send,
{
    // TODO: Deprecate this
    /// Optimizes *all* quadrature rules (mass, stiffness, elliptic, ...) to the same
    /// polynomial strength.
    ///
    /// This is part of the legacy API and will be removed in the future.
    pub fn optimize_quadrature(
        &mut self,
        polynomial_strength: usize,
        lp_solver: &(impl Sync + LpSolver<T>),
    ) -> Result<(), QuadratureOptimizationError> {
        let solver = &lp_solver;
        let optimize_quadrature_rules = |rules: &mut Vec<QuadraturePair<T, D>>| {
            rules
                .par_iter_mut()
                .enumerate()
                .map(|(idx, current_quadrature)| {
                    let new_quadrature = optimize_quadrature(&*current_quadrature, polynomial_strength, solver);
                    match new_quadrature {
                        Ok(new_quadrature) => {
                            *current_quadrature = new_quadrature;
                            Ok(())
                        }
                        Err(err) => Err(QuadratureOptimizationError {
                            interface_connectivity_index: idx,
                            optimization_error: err,
                        }),
                    }
                })
                .collect::<Result<Vec<()>, _>>()
                .map(|_| ())
        };

        if let Some(quadrature) = &mut self.mass_quadrature {
            optimize_quadrature_rules(&mut quadrature.interface_quadratures)?;
        }

        if let Some(quadrature) = &mut self.stiffness_quadrature {
            optimize_quadrature_rules(&mut quadrature.interface_quadratures)?;
        }

        if let Some(quadrature) = &mut self.elliptic_quadrature {
            optimize_quadrature_rules(&mut quadrature.interface_quadratures)?;
        }

        Ok(())
    }
}
