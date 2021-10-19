use std::convert::identity;
use std::error::Error;

use coarse_prof::profile;
use fenris::assembly::{
    apply_homogeneous_dirichlet_bc_csr, DefaultSemidefiniteProjection, ElementMatrixTransformation, NoTransformation,
};
use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName};
use fenris::solid::ElasticityModelParallel;
use fenris::{solid, CsrMatrix};
use nalgebra_lapack::SymmetricEigen;

use crate::components::Gravity;
use crate::fem::{DirichletBoundaryConditions, OptionalDirichletBoundaryConditions};

pub struct LapackSemidefiniteProjection;

impl ElementMatrixTransformation<f64> for LapackSemidefiniteProjection {
    fn transform_element_matrix(&self, element_matrix: &mut DMatrixSliceMut<f64>) {
        let mut eigen_decomposition = SymmetricEigen::new(element_matrix.clone_owned());
        for eval in &mut eigen_decomposition.eigenvalues {
            *eval = f64::max(0.0, *eval);
        }
        // TODO: Don't recompose if we don't need to make changes
        element_matrix.copy_from(&eigen_decomposition.recompose());
    }
}

pub fn compute_gravity_density<D>(
    model: &dyn ElasticityModelParallel<f64, D>,
    gravity: &Gravity,
) -> Result<DVector<f64>, Box<dyn Error>>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    let ndof = model.ndof();
    let mut g_force_density = DVector::zeros(ndof);
    match gravity {
        Gravity::Scalar(g) => {
            for i in 0..ndof {
                if (i % D::dim()) + 1 == D::dim() {
                    g_force_density[i] = *g;
                }
            }
        }
        Gravity::Vec2(g_2d) => {
            if D::dim() != 2 {
                return Err(Box::from(
                    "Trying to apply 2d gravity to model that is not two dimensional",
                ));
            }

            for i in 0..ndof {
                let j = i % 2;
                g_force_density[i] = g_2d[j];
            }
        }
        Gravity::Vec3(g_3d) => {
            if D::dim() != 3 {
                return Err(Box::from(
                    "Trying to apply 3d gravity to model that is not three dimensional",
                ));
            }

            for i in 0..ndof {
                let j = i % 3;
                g_force_density[i] = g_3d[j];
            }
        }
    };

    Ok(g_force_density)
}

/// Applies the system's Dirichlet boundary conditions to the vectors of unknowns
pub fn apply_dirichlet_bcs_unknowns<'a>(
    t: f64,
    u: impl Into<DVectorSliceMut<'a, f64>>,
    v: impl Into<DVectorSliceMut<'a, f64>>,
    dirichlet_bcs: Option<&dyn DirichletBoundaryConditions>,
) {
    if let Some(dirichlet_bcs) = dirichlet_bcs {
        let mut u = u.into();
        let mut v = v.into();
        let d = dirichlet_bcs.solution_dim();

        let mut bcs_u = DVector::zeros(dirichlet_bcs.nrows());
        let mut bcs_v = DVector::zeros(dirichlet_bcs.nrows());

        // Evaluate boundary conditions
        dirichlet_bcs.apply_displacement_bcs(DVectorSliceMut::from(&mut bcs_u), t);
        dirichlet_bcs.apply_velocity_bcs(DVectorSliceMut::from(&mut bcs_v), t);

        // Apply boundary conditions
        for (i_local, i_global) in dirichlet_bcs.nodes().iter().copied().enumerate() {
            for id in 0..d {
                *u.index_mut(d * i_global + id) = *bcs_u.index(d * i_local + id);
                *v.index_mut(d * i_global + id) = *bcs_v.index(d * i_local + id);
            }
        }
    }
}

/// Note: Does not apply boundary conditions to the stiffness matrix itself
pub fn compute_stiffness_matrix_into<D: DimName>(
    model: &dyn ElasticityModelParallel<f64, D>,
    t: f64,
    u: DVectorSlice<f64>,
    v: DVectorSlice<f64>,
    stiffness_matrix: &mut CsrMatrix<f64>,
    dirichlet_bcs: Option<&dyn DirichletBoundaryConditions>,
    material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, D>),
    // Whether we should ensure that the returned matrix is semidefinite
    projected: bool,
) where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    // TODO: Avoid all this unnecessary memory allocation

    // Apply boundary conditions
    let mut u = u.clone_owned();
    let mut v = v.clone_owned();
    apply_dirichlet_bcs_unknowns(t, &mut u, &mut v, dirichlet_bcs);

    {
        profile!("assemble stiffness");
        stiffness_matrix.fill_par(0.0);

        if !projected || material_model.is_positive_semi_definite() {
            model.assemble_transformed_stiffness_into_par(stiffness_matrix, &u, material_model, &NoTransformation);
        } else {
            model.assemble_transformed_stiffness_into_par(
                stiffness_matrix,
                &u,
                material_model,
                &DefaultSemidefiniteProjection,
            );
        }
    }
}

/// Note: Does not apply boundary conditions to the damping matrix itself
pub fn compute_damping_matrix_into<D>(
    model: &dyn ElasticityModelParallel<f64, D>,
    u: DVectorSlice<f64>,
    stiffness_matrix: &mut CsrMatrix<f64>,
    damping_matrix: &mut CsrMatrix<f64>,
    mass_matrix: &CsrMatrix<f64>,
    mass_damping_coefficient: Option<f64>,
    stiffness_damping_coefficient: Option<f64>,
    material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, D>),
) -> bool
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    profile!("assemble damping matrix");

    if mass_damping_coefficient.is_none() && stiffness_damping_coefficient.is_none() {
        return false;
    }

    if stiffness_damping_coefficient.is_some() {
        compute_stiffness_matrix_into(
            model,
            0.0,
            DVectorSlice::from(u),
            DVectorSlice::from(u),
            stiffness_matrix,
            None,
            material_model,
            // Always use projection to ensure that the damping matrix is
            // at least positive semidefinite
            true,
        );
    }

    // Compute damping matrix as `D = gamma * M + alpha * K`
    compute_jacobian_combination_into(
        damping_matrix,
        stiffness_matrix,
        None,
        mass_matrix,
        stiffness_damping_coefficient.map(|alpha| -alpha),
        None,
        mass_damping_coefficient,
        None,
    );

    return true;
}

/// Computes a Jacobian linear combination `J = gamma * M - beta * D - alpha * K` and applies boundary conditions.
pub fn compute_jacobian_combination_into<'a, D: DimName>(
    jacobian_combination: &mut CsrMatrix<f64>,
    stiffness_matrix: &CsrMatrix<f64>,
    damping_matrix: Option<&CsrMatrix<f64>>,
    mass_matrix: &CsrMatrix<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    dirichlet_bcs: Option<&dyn DirichletBoundaryConditions>,
) where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    {
        profile!("compute jacobian combination");
        // J = gamma * M - beta * D - alpha * K
        let coefficient_matrix_pairs = [
            // Stiffness matrix: negate alpha because df/dx = -K
            alpha.map(|alpha| (-alpha, stiffness_matrix)),
            // Damping matrix: negate beta because df/dv = -D
            // Nested map, because there might not be a damping matrix
            beta.and_then(|beta| damping_matrix.map(|damping_matrix| (-beta, damping_matrix))),
            // Mass matrix
            gamma.map(|gamma| (gamma, mass_matrix)),
        ];

        jacobian_combination.fill_par(0.0);
        jacobian_combination.add_assign_linear_combination_par(
            coefficient_matrix_pairs
                .iter()
                .cloned()
                // Skip None entries
                .filter_map(identity),
        );
    }

    {
        profile!("apply Dirichlet BC CSR");
        apply_homogeneous_dirichlet_bc_csr::<_, D>(jacobian_combination, &dirichlet_bcs.nodes());
    }
}
