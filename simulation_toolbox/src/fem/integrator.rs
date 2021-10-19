use std::error::Error;
use std::fmt;
use std::fmt::Display;

use coarse_prof::profile;
use fenris::assembly::apply_homogeneous_dirichlet_bc_rhs;
use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName, U4};
use fenris::solid::ElasticityModelParallel;
use fenris::sparse::spmv_csr;
use fenris::{solid, CsrMatrix};
use global_stash::stash_scope;
use hamilton::{BijectiveStorage, StorageContainer, System};
use hamilton2::dynamic_system::{DifferentiableDynamicSystem, DynamicSystem};
use hamilton2::integrators::{backward_euler_step, symplectic_euler_step, BackwardEulerSettings};
use log::{info, warn};
use mkl_corrode::dss;
use mkl_corrode::dss::Definiteness;
use mkl_corrode::dss::MatrixStructure::Symmetric;

use crate::components::{get_gravity, get_simulation_time, get_time_step, Gravity, Name};
use crate::fem::system_assembly::*;
use crate::fem::{
    DirichletBoundaryConditionComponent, DirichletBoundaryConditions, ElasticModelMatrixStorage,
    FiniteElementElasticModel2d, FiniteElementElasticModel3d, IntegrationMethod, OptionalDirichletBoundaryConditions,
};
use crate::util::IfTrue;

#[derive(Debug, Clone)]
pub struct IntegratorSettings {
    project_stiffness: bool,
}

impl Default for IntegratorSettings {
    fn default() -> Self {
        Self {
            project_stiffness: true,
        }
    }
}

impl IntegratorSettings {
    pub fn project_stiffness(&self) -> bool {
        self.project_stiffness
    }

    pub fn set_project_stiffness(mut self, project: bool) -> Self {
        self.project_stiffness = project;
        self
    }
}

#[derive(Debug, Default)]
pub struct FiniteElementIntegrator {
    integrator2d: FiniteElementIntegrator2d,
    integrator3d: FiniteElementIntegrator3d,
}

#[derive(Debug, Default)]
pub struct FiniteElementIntegrator2d {
    settings: IntegratorSettings,
}

#[derive(Debug, Default)]
pub struct FiniteElementIntegrator3d {
    settings: IntegratorSettings,
}

impl FiniteElementIntegrator {
    pub fn with_settings(settings: IntegratorSettings) -> Self {
        Self {
            integrator2d: FiniteElementIntegrator2d::with_settings(settings.clone()),
            integrator3d: FiniteElementIntegrator3d::with_settings(settings),
        }
    }
}

impl FiniteElementIntegrator2d {
    pub fn with_settings(settings: IntegratorSettings) -> Self {
        Self { settings }
    }
}

impl FiniteElementIntegrator3d {
    pub fn with_settings(settings: IntegratorSettings) -> Self {
        Self { settings }
    }
}

impl Display for FiniteElementIntegrator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FiniteElementIntegrator")
    }
}

impl Display for FiniteElementIntegrator2d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FiniteElementIntegrator2d")
    }
}

impl Display for FiniteElementIntegrator3d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FiniteElementIntegrator3d")
    }
}

/// Integrates a finite element model by recursively substepping upon failures.
fn recursively_substep_finite_element_model<'a, D, Model>(
    model: &Model,
    num_substeps: usize,
    initial_t: f64,
    initial_dt: f64,
    u: impl Into<DVectorSliceMut<'a, f64>>,
    v: impl Into<DVectorSliceMut<'a, f64>>,
    dirichlet_bcs: Option<&'a dyn DirichletBoundaryConditions>,
    matrices: &mut ElasticModelMatrixStorage,
    solver: &mut Option<dss::Solver<f64>>,
    mass_damping_coefficient: Option<f64>,
    stiffness_damping_coefficient: Option<f64>,
    material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, D>),
    integrator: &IntegrationMethod,
    gravity: Option<&Gravity>,
    project_stiffness: bool,
) -> Result<(), Box<dyn Error>>
where
    Model: ElasticityModelParallel<f64, D>,
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    // If the number of substeps in a recursion exceeds this number, we abort
    let max_substeps = 1000;
    // Number of substeps to take when re-trying with a smaller time step
    let num_substeps_per_level = 5;

    let mut u = u.into();
    let mut v = v.into();

    // TODO: We're currently cloning this upon every recursion, but this is not necessary
    // (it never changes throughout all of the substepping)
    let u_initial = u.clone_owned();
    let v_initial = v.clone_owned();

    let dt = initial_dt / (num_substeps as f64);
    for step_idx in 0..num_substeps {
        let t = initial_t + (step_idx as f64) * dt;
        if let Err(err) = integrate_finite_element_model(
            model,
            t,
            dt,
            &mut u,
            &mut v,
            dirichlet_bcs,
            matrices,
            solver,
            mass_damping_coefficient,
            stiffness_damping_coefficient,
            material_model,
            integrator,
            gravity,
            project_stiffness,
        ) {
            let new_num_substeps = num_substeps * num_substeps_per_level;
            if new_num_substeps > max_substeps {
                return Err(Box::from(format!(
                    "Substepping failed.\
                    Could not find solution using {} substeps. Latest error from integrator:\
                    {}",
                    num_substeps, err
                )));
            } else {
                warn!(
                    "Integrator failed. Attempting to substep with {} substeps",
                    new_num_substeps
                );
                // Reset state to initial state given at the start of the time step.
                u.copy_from(&u_initial);
                v.copy_from(&v_initial);

                recursively_substep_finite_element_model(
                    model,
                    new_num_substeps,
                    initial_t,
                    initial_dt,
                    &mut u,
                    &mut v,
                    dirichlet_bcs,
                    matrices,
                    solver,
                    mass_damping_coefficient,
                    stiffness_damping_coefficient,
                    material_model,
                    integrator,
                    gravity,
                    project_stiffness,
                )?;
            }
        }
    }

    assert!(u.iter().all(|u_i| u_i.is_finite()));
    assert!(v.iter().all(|v_i| v_i.is_finite()));

    Ok(())
}

fn integrate_finite_element_model<'a, D, Model>(
    model: &Model,
    t: f64,
    dt: f64,
    u: impl Into<DVectorSliceMut<'a, f64>>,
    v: impl Into<DVectorSliceMut<'a, f64>>,
    dirichlet_bcs: Option<&'a dyn DirichletBoundaryConditions>,
    matrices: &mut ElasticModelMatrixStorage,
    solver: &mut Option<dss::Solver<f64>>,
    mass_damping_coefficient: Option<f64>,
    stiffness_damping_coefficient: Option<f64>,
    material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, D>),
    integrator: &IntegrationMethod,
    gravity: Option<&Gravity>,
    project_stiffness: bool,
) -> Result<(), Box<dyn Error>>
where
    Model: ElasticityModelParallel<f64, D>,
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D> + Allocator<f64, U4> + Allocator<f64, U4, U4>,
{
    let mut u = u.into();
    let mut v = v.into();

    let g_force = if let Some(gravity) = gravity {
        let g_force_density = compute_gravity_density(model, gravity)?;
        &matrices.mass_matrix * &g_force_density
    } else {
        DVector::zeros(model.ndof())
    };

    matrices.has_damping_matrix = compute_damping_matrix_into(
        model,
        DVectorSlice::from(&u),
        &mut matrices.stiffness_matrix,
        &mut matrices.damping_matrix,
        &matrices.mass_matrix,
        mass_damping_coefficient,
        stiffness_damping_coefficient,
        material_model,
    );

    let representative_force = matrices.representative_force;

    let mut system = FiniteElementElastodynamicSystem {
        model,
        dirichlet_bcs,

        matrices,
        solver,

        material_model,
        g_force: &g_force,
        project_stiffness,

        state: None,
        jacobian_apply_params: None,
        jacobian_solve_params: None,

        stiffness_matrix_outdated: true,
        inverse_mass_matrix_outdated: true,
        solve_jacobian_combination_outdated: true,
        apply_jacobian_combination_outdated: true,
    };

    let tolerance = 1e-5 * representative_force;
    let max_newton_iter = Some(100);

    {
        let u = DVectorSliceMut::from(&mut u);
        let v = DVectorSliceMut::from(&mut v);
        match integrator {
            IntegrationMethod::SymplecticEuler => {
                profile!("symplectic euler");
                let nrows = u.nrows();
                symplectic_euler_step(
                    &mut system,
                    u,
                    v,
                    &mut DVector::zeros(nrows),
                    &mut DVector::zeros(nrows),
                    t,
                    dt,
                )?;
            }
            IntegrationMethod::BackwardEuler => {
                let integrator_settings = BackwardEulerSettings {
                    max_newton_iter,
                    tolerance,
                };

                profile!("backward euler");
                stash_scope!("backward euler");
                let newton_iter = backward_euler_step(&mut system, u, v, t, dt, integrator_settings)?;
                info!("Number of newton iterations in Backward Euler step: {}", newton_iter);
                global_stash::insert_value_or_modify("newton_iter", newton_iter, |accumulated_iter| {
                    let current_iter = accumulated_iter.as_u64().unwrap();
                    *accumulated_iter = (current_iter + newton_iter as u64).into();
                });
            }
        }
    }

    apply_dirichlet_bcs_unknowns(t + dt, u, v, dirichlet_bcs);

    Ok(())
}

impl System for FiniteElementIntegrator2d {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let dt = get_time_step(data)?;
        let t = get_simulation_time(data)?;
        let g = get_gravity(data)?;

        let mut models_2d = data
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();

        let bcs = data
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow();

        for (id, model) in models_2d.entity_component_iter_mut() {
            let model_bcs = bcs
                .get_component_for_entity(*id)
                .map(|bc_comp| &*bc_comp.bc);

            model.ensure_model_matrix_storage_initialized(model_bcs);
            // TODO: This should only be necessary on the very first timestep, to make initial conditions consistent
            apply_dirichlet_bcs_unknowns(t, &mut model.u, &mut model.v, model_bcs);

            let g = model.gravity_enabled.if_true(&g);

            match_on_elastic_material_model!(model.material.elastic_model, material_model => {
                // TODO: Make this configurable also for 2D
                recursively_substep_finite_element_model(
                    &model.model,
                    1,
                    t,
                    dt,
                    &mut model.u,
                    &mut model.v,
                    model_bcs,
                    model.model_matrix_storage.as_mut().expect("ElasticModelMatrixStorage was not initialized"),
                    &mut model.factorization,
                    model.material.mass_damping_coefficient,
                    model.material.stiffness_damping_coefficient,
                    material_model,
                    &model.integrator,
                    g,
                    self.settings.project_stiffness()
                )?;
            });
        }

        Ok(())
    }
}

impl System for FiniteElementIntegrator3d {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let dt = get_time_step(data)?;
        let t = get_simulation_time(data)?;
        let g = get_gravity(data)?;

        let mut models_3d = data
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();

        let bcs = data
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow();

        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, model) in models_3d.entity_component_iter_mut() {
            let model_bcs = bcs
                .get_component_for_entity(*id)
                .map(|bc_comp| &*bc_comp.bc);

            let scope_name = if let Some(name) = name_storage.get_component(*id) {
                name.0.clone()
            } else {
                format!("{:?}", *id)
            };
            stash_scope!(scope_name);

            model.ensure_model_matrix_storage_initialized(model_bcs);
            // TODO: This should only be necessary on the very first timestep, to make initial conditions consistent
            apply_dirichlet_bcs_unknowns(t, &mut model.u, &mut model.v, model_bcs);

            let g = if model.gravity_enabled {
                // TODO: Add a EntityGravityComponent
                Some(&g)
            } else {
                None
            };

            match_on_elastic_material_model!(model.material.elastic_model, material_model => {
                recursively_substep_finite_element_model(
                    &model.model,
                    1,
                    t,
                    dt,
                    &mut model.u,
                    &mut model.v,
                    model_bcs,
                    model.model_matrix_storage.as_mut().expect("ElasticModelMatrixStorage was not initialized"),
                    &mut model.factorization,
                    model.material.mass_damping_coefficient,
                    model.material.stiffness_damping_coefficient,
                    material_model,
                    &model.integrator,
                    g,
                    self.settings.project_stiffness()
                )?;
            });
        }

        Ok(())
    }
}

impl System for FiniteElementIntegrator {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        self.integrator2d.run(data)?;
        self.integrator3d.run(data)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct State {
    t: f64,
    u: DVector<f64>,
    v: DVector<f64>,
}

// TODO: Clean up this mess,
struct FiniteElementElastodynamicSystem<'a, D: DimName>
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    model: &'a dyn ElasticityModelParallel<f64, D>,
    dirichlet_bcs: Option<&'a dyn DirichletBoundaryConditions>,

    matrices: &'a mut ElasticModelMatrixStorage,
    solver: &'a mut Option<dss::Solver<f64>>,

    material_model: &'a (dyn Sync + solid::ElasticMaterialModel<f64, D>),
    g_force: &'a DVector<f64>,

    project_stiffness: bool,

    /// The current state of the dynamic system (t, u, v)
    state: Option<State>,
    /// Coefficients (alpha, beta, gamma) for the apply_jacobian_combination method
    jacobian_apply_params: Option<(Option<f64>, Option<f64>, Option<f64>)>,
    /// Coefficients (alpha, beta) for the solve_jacobian_combination method
    jacobian_solve_params: Option<(Option<f64>, Option<f64>)>,

    /// If the stiffness matrix is outdated, everything that depends on it should be recomputed
    stiffness_matrix_outdated: bool,
    /// If the inverse mass matrix factorization is outdated, it has to be recomputed before applying it
    inverse_mass_matrix_outdated: bool,
    /// If the cached Jacobian combination is outdated, it should be recomputed before applying it
    apply_jacobian_combination_outdated: bool,
    /// If the solver is outdated, a numerical refactorization has to be performed before solving
    solve_jacobian_combination_outdated: bool,
}

impl<'a, D: DimName> FiniteElementElastodynamicSystem<'a, D>
where
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    /// Computes a new stiffness matrix from the currently cached state without checking if it is outdated
    fn update_stiffness_matrix(&mut self) -> Result<(), Box<dyn Error>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Box::<dyn Error>::from("No state was set using `set_state`"))?;

        compute_stiffness_matrix_into(
            self.model,
            state.t,
            DVectorSlice::from(&state.u),
            DVectorSlice::from(&state.v),
            &mut self.matrices.stiffness_matrix,
            self.dirichlet_bcs,
            self.material_model,
            self.project_stiffness,
        );

        self.stiffness_matrix_outdated = false;

        // The Jacobian operators are now outdated after updating the stiffness matrix
        self.apply_jacobian_combination_outdated = true;
        self.solve_jacobian_combination_outdated = true;

        Ok(())
    }

    /// Computes a new stiffness matrix if it is outdated
    fn ensure_stiffness_matrix_updated(&mut self) -> Result<(), Box<dyn Error>> {
        if self.stiffness_matrix_outdated {
            self.update_stiffness_matrix()?
        }

        Ok(())
    }

    /// Computes a new mass matrix factorization if it is outdated
    fn ensure_inverse_mass_matrix_updated(&mut self) -> Result<(), Box<dyn Error>> {
        // TODO: Maybe this should use a separate factorization? Are there integrators that use both
        //  an inverse mass matrix and a solve_jacobian_combination call?

        if self.inverse_mass_matrix_outdated {
            profile!("refactorize mass matrix");

            Self::refactorize(self.solver, &self.matrices.mass_matrix)?;
            self.inverse_mass_matrix_outdated = false;

            // The jacobian factorization was replaced by the mass matrix factorization
            self.solve_jacobian_combination_outdated = true;
        }

        Ok(())
    }

    /// Computes a new Jacobian combination if it is outdated
    fn ensure_jacobian_combination_apply_updated(&mut self) -> Result<(), Box<dyn Error>> {
        self.ensure_stiffness_matrix_updated()?;

        if self.apply_jacobian_combination_outdated {
            profile!("compute jacobian linear combination");
            let (alpha, beta, gamma) = self.jacobian_apply_params.clone().ok_or_else(|| {
                Box::<dyn Error>::from(
                    "No (alpha, beta, gamma) parameters were set using `init_apply_jacobian_combination`",
                )
            })?;

            compute_jacobian_combination_into(
                &mut self.matrices.linear_combination_apply,
                &self.matrices.stiffness_matrix,
                self.matrices
                    .has_damping_matrix
                    .if_true(&self.matrices.damping_matrix),
                &self.matrices.mass_matrix,
                alpha,
                beta,
                gamma,
                self.dirichlet_bcs,
            );

            self.apply_jacobian_combination_outdated = false;
        }

        Ok(())
    }

    /// Computes a new Jacobian factorization if it is outdated
    fn ensure_jacobian_combination_solve_updated(&mut self) -> Result<(), Box<dyn Error>> {
        self.ensure_stiffness_matrix_updated()?;

        if self.solve_jacobian_combination_outdated {
            profile!("refactorize jacobian linear combination");

            let (alpha, beta) = self.jacobian_solve_params.clone().ok_or_else(|| {
                Box::<dyn Error>::from("No (alpha, beta) parameters were set using `init_solve_jacobian_combination`")
            })?;
            assert!(-alpha.unwrap_or(0.0) > 0.0);

            // Compute the requested Jacobian combination to factorize
            compute_jacobian_combination_into(
                &mut self.matrices.linear_combination_solve,
                &self.matrices.stiffness_matrix,
                self.matrices
                    .has_damping_matrix
                    .if_true(&self.matrices.damping_matrix),
                &self.matrices.mass_matrix,
                alpha,
                beta,
                Some(1.0),
                self.dirichlet_bcs,
            );

            Self::refactorize(self.solver, &self.matrices.linear_combination_solve)?;
            self.solve_jacobian_combination_outdated = false;

            // The mass matrix factorization was replaced by this factorization
            self.inverse_mass_matrix_outdated = true;
        }

        Ok(())
    }

    /// Perform a factorization of the specified matrix in the cached solver
    fn refactorize(solver: &mut Option<dss::Solver<f64>>, matrix: &CsrMatrix<f64>) -> Result<(), Box<dyn Error>> {
        // Perform factorization
        let new_dss_solver = {
            profile!("factorization");
            let solver_status = if let Some(mut solver) = solver.take() {
                profile!("numerical refactorization");
                // MKL requires the non-zeros corresponding to the upper triangular
                // part of the matrix
                let mut upper_triangular_values = Vec::with_capacity(matrix.nnz() / 2 + matrix.nrows());
                upper_triangular_values.extend(matrix.iter().filter(|(i, j, _)| j >= i).map(|(_, _, v)| *v));
                solver
                    .refactor(&upper_triangular_values, Definiteness::PositiveDefinite)
                    .map(|_| solver)
            } else {
                profile!("full factorization");
                // MKL only accepts a triangular part of the matrix, so we must (unfortunately)
                // first convert the matrix into something DSS accepts, even if we just want to refactor.
                let matrix_dss = dss::SparseMatrix::try_convert_from_csr(
                    matrix.row_offsets(),
                    matrix.column_indices(),
                    matrix.values(),
                    Symmetric,
                )?;
                let options = dss::SolverOptions::default().parallel_reorder(true);
                dss::Solver::try_factor_with_opts(&matrix_dss, Definiteness::PositiveDefinite, &options)
            };

            match solver_status {
                Ok(solver) => solver,
                Err(e) => {
                    match e.return_code() {
                        // Usually this error indicates that the Jacobian is not positive definite
                        dss::ErrorCode::TermLvlErr => {
                            profile!("indefinite factorization");
                            warn!(
                                "Warning: Jacobian does not seem to be positive definite. \
                               Using indefinite factorization."
                            );
                            // Try to factorize again with indefinite factorization
                            let jacobian_dss = dss::SparseMatrix::try_convert_from_csr(
                                matrix.row_offsets(),
                                matrix.column_indices(),
                                matrix.values(),
                                Symmetric,
                            )?;
                            let options = dss::SolverOptions::default().parallel_reorder(true);
                            dss::Solver::try_factor_with_opts(&jacobian_dss, Definiteness::Indefinite, &options)?
                        }
                        _ => return Err(e.into()),
                    }
                }
            }
        };

        solver.replace(new_dss_solver);

        Ok(())
    }
}

impl<'a, D> DynamicSystem<f64> for FiniteElementElastodynamicSystem<'a, D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    fn apply_mass_matrix(&mut self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) {
        profile!("apply mass matrix");
        spmv_csr(1.0, &mut y, 1.0, &self.matrices.mass_matrix, &x);
    }

    fn apply_inverse_mass_matrix(
        &mut self,
        mut y: DVectorSliceMut<f64>,
        x: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        self.ensure_inverse_mass_matrix_updated()?;

        {
            profile!("apply inverse mass");

            let solution = {
                profile!("solve");
                self.solver.as_mut().unwrap().solve(x.as_slice())?
            };

            y.copy_from_slice(solution.as_slice());
        }

        Ok(())
    }

    fn eval_f(
        &mut self,
        mut f: DVectorSliceMut<f64>,
        t: f64,
        u: DVectorSlice<f64>,
        v: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        profile!("eval f");

        // TODO: Avoid copy?
        let mut u = u.clone_owned();
        let mut v = v.clone_owned();

        // Apply boundary conditions to unknowns
        apply_dirichlet_bcs_unknowns(
            t,
            DVectorSliceMut::from(&mut u),
            DVectorSliceMut::from(&mut v),
            self.dirichlet_bcs,
        );

        // Compute elastic forces
        f.fill(0.0);
        self.model.assemble_elastic_pseudo_forces_into_par(
            DVectorSliceMut::from(&mut f),
            DVectorSlice::from(&u),
            self.material_model,
        );
        // Apply gravity
        f += self.g_force;
        // Apply damping
        if self.matrices.has_damping_matrix {
            spmv_csr(1.0, &mut f, -1.0, &self.matrices.damping_matrix, &v);
        }
        // Apply boundary conditions
        apply_homogeneous_dirichlet_bc_rhs(f, self.dirichlet_bcs.nodes(), D::dim());

        Ok(())
    }
}

impl<'a, D> DifferentiableDynamicSystem<f64> for FiniteElementElastodynamicSystem<'a, D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    fn set_state(&mut self, t: f64, u: DVectorSlice<f64>, v: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        self.state.replace(State {
            t,
            u: u.clone_owned(),
            v: v.clone_owned(),
        });

        // Updating the state invalidates all cached operators (except for inverse mass matrix)
        self.stiffness_matrix_outdated = true;
        self.apply_jacobian_combination_outdated = true;
        self.solve_jacobian_combination_outdated = true;
        Ok(())
    }

    fn init_apply_jacobian_combination(
        &mut self,
        alpha: Option<f64>,
        beta: Option<f64>,
        gamma: Option<f64>,
    ) -> Result<(), Box<dyn Error>> {
        self.jacobian_apply_params.replace((alpha, beta, gamma));

        self.apply_jacobian_combination_outdated = true;
        Ok(())
    }

    fn init_solve_jacobian_combination(&mut self, alpha: Option<f64>, beta: Option<f64>) -> Result<(), Box<dyn Error>> {
        self.jacobian_solve_params.replace((alpha, beta));

        self.solve_jacobian_combination_outdated = true;
        Ok(())
    }

    fn apply_jacobian_combination(
        &mut self,
        mut y: DVectorSliceMut<f64>,
        x: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        self.ensure_jacobian_combination_apply_updated()?;

        {
            profile!("apply jacobian combination");
            spmv_csr(1.0, &mut y, 1.0, &self.matrices.linear_combination_apply, &x);
            Ok(())
        }
    }

    fn solve_jacobian_combination(
        &mut self,
        mut sol: DVectorSliceMut<f64>,
        rhs: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        self.ensure_jacobian_combination_solve_updated()?;

        {
            profile!("solve jacobian combination");

            let solution = {
                profile!("solve");
                self.solver.as_mut().unwrap().solve(rhs.as_slice())?
            };

            sol.copy_from_slice(solution.as_slice());
            Ok(())
        }
    }
}
