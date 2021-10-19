use crate::components::{get_gravity, get_simulation_time, get_time_step, Gravity};
use crate::fem::schwarz_precond::{
    build_preconditioner_pattern, build_schwarz_preconditioner_into, SchwarzPreconditionerComponent,
};
use crate::fem::system_assembly::{
    apply_dirichlet_bcs_unknowns, compute_damping_matrix_into, compute_gravity_density,
    compute_jacobian_combination_into, compute_stiffness_matrix_into,
};
use crate::fem::{
    DirichletBoundaryConditionComponent, DirichletBoundaryConditions, ElasticModelMatrixStorage,
    FiniteElementElasticModel3d, OptionalDirichletBoundaryConditions,
};
use coarse_prof::profile;
use fenris::assembly::apply_homogeneous_dirichlet_bc_rhs;
use fenris::cg::{CgWorkspace, ConjugateGradient, LinearOperator, RelativeResidualCriterion};
use fenris::nalgebra::{DVector, DVectorSlice, DVectorSliceMut, Dynamic, Matrix3, VecStorage, U1, U3};
use fenris::nested_vec::NestedVec;
use fenris::solid::{ElasticMaterialModel, ElasticityModelParallel};
use fenris::sparse::{spmv_csr, SparsityPattern};
use fenris::CsrMatrix;
use hamilton::{BijectiveStorage, BijectiveStorageMut, StorageContainer, System};
use hamilton2::dynamic_system::{DifferentiableDynamicSystem, DynamicSystem};
use hamilton2::integrators::{backward_euler_step, BackwardEulerSettings};
use log::{info, warn};
use mkl_corrode::mkl_sys::MKL_INT;
use mkl_corrode::sparse::{CsrMatrixHandle, MatrixDescription, SparseOperation};
use paradis::DisjointSubsets;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;

#[derive(Debug, Default)]
pub struct NewtonCgIntegrator3d {}

impl fmt::Display for NewtonCgIntegrator3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", std::any::type_name::<Self>())
    }
}

impl System for NewtonCgIntegrator3d {
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

        let mut schwarz_components = data
            .get_component_storage::<SchwarzPreconditionerComponent>()
            .borrow_mut();

        for (id, model) in models_3d.entity_component_iter_mut() {
            let model_bcs = bcs
                .get_component_for_entity(*id)
                .map(|bc_comp| &*bc_comp.bc);

            model.ensure_model_matrix_storage_initialized(model_bcs);
            // TODO: This should only be necessary on the very first timestep,
            // to make initial conditions consistent
            apply_dirichlet_bcs_unknowns(t, &mut model.u, &mut model.v, model_bcs);

            let g = if model.gravity_enabled {
                // TODO: Add a EntityGravityComponent
                Some(&g)
            } else {
                None
            };

            let model_storage = model.model_matrix_storage.as_mut().unwrap();
            // TODO: Dont build workspace on every iteration
            let mut workspace = NewtonCgWorkspace::build_workspace(model_storage.mass_matrix.nrows());
            workspace.prepare_workspace(model_storage.mass_matrix.nrows());

            let schwarz_component = schwarz_components.get_component_for_entity_mut(*id);
            let mut system_preconditioner = create_preconditioner(&model_storage.mass_matrix, schwarz_component)?;

            // Note: This match is absolutely necessary for avoiding overhead related to
            // dynamic dispatch.
            // TODO: Remove trait impl for ElasticMaterialModel component to avoid
            // falling into this performance trap
            match_on_elastic_material_model!(model.material.elastic_model, elastic_model => {
                integrate_model_with_substeps(
                    model_storage,
                    elastic_model,
                    model_bcs, g, &model.model,
                    t, dt,
                    model.material.mass_damping_coefficient,
                    model.material.stiffness_damping_coefficient,
                    (&mut model.u).into(), (&mut model.v).into(),
                    &mut system_preconditioner,
                    &mut workspace)?;
            });
        }

        Ok(())
    }
}

fn create_preconditioner<'a>(
    mass_matrix: &CsrMatrix<f64>,
    schwarz_component: Option<&'a mut SchwarzPreconditionerComponent>,
) -> Result<Box<dyn SystemPreconditioner + 'a>, Box<dyn Error>> {
    if let Some(schwarz_component) = schwarz_component {
        if schwarz_component.preconditioner.is_none() {
            let num_nodes = mass_matrix.nrows() / 3;
            let pattern = build_preconditioner_pattern(
                num_nodes,
                &schwarz_component.schwarz_connectivity,
                &schwarz_component.untreated_nodes,
                3,
            );
            let nnz = pattern.nnz();
            let csr = CsrMatrix::from_pattern_and_values(pattern, vec![0.0; nnz]);
            schwarz_component.preconditioner.replace(csr);
        }

        let precond = SchwarzPreconditioner {
            matrix: schwarz_component.preconditioner.as_mut().unwrap(),
            schwarz_connectivity: &schwarz_component.schwarz_connectivity,
            untreated_nodes: &schwarz_component.untreated_nodes,
            colors: schwarz_component
                .colors
                .as_ref()
                .ok_or_else(|| Box::<dyn Error>::from("Missing coloring for Schwarz preconditioner"))?,
            mkl_matrix: None,
        };

        Ok(Box::new(precond))
    } else {
        // We don't have a Schwarz component, so fall back to diagonal preconditioning
        // The preconditioner will anyway be updated, so we just create a default one for now
        let diagonal_preconditioner = DiagonalPreconditioner::default();
        Ok(Box::new(diagonal_preconditioner))
    }
}

fn integrate_model_with_substeps<'a>(
    model_matrix_storage: &mut ElasticModelMatrixStorage,
    material_model: &(dyn Sync + ElasticMaterialModel<f64, U3>),
    dirichlet_bc: Option<&dyn DirichletBoundaryConditions>,
    gravity: Option<&Gravity>,
    fe_model: &dyn ElasticityModelParallel<f64, U3>,
    t: f64,
    dt: f64,
    mass_damping_coefficient: Option<f64>,
    stiffness_damping_coefficient: Option<f64>,
    mut u: DVectorSliceMut<f64>,
    mut v: DVectorSliceMut<f64>,
    system_preconditioner: &'a mut (dyn SystemPreconditioner + 'a),
    workspace: &mut NewtonCgWorkspace,
) -> Result<(), Box<dyn Error>> {
    // If the number of substeps in a recursion exceeds this number, we abort
    let max_substeps = 1000;
    // Multiply number of substeps by this factor every time there is a failure
    let substep_factor = 5;

    // TODO: We're currently cloning this upon every recursion, but this is not necessary
    // (it never changes throughout all of the substepping)

    let mut u_substep = u.clone_owned();
    let mut v_substep = v.clone_owned();

    let mut num_substeps = 1;
    let mut substep_dt = dt / (num_substeps as f64);
    let mut substep_t;

    'outer: loop {
        for substep_idx in 0..num_substeps {
            substep_t = t + substep_idx as f64 * substep_dt;
            let substep_result = integrate_model(
                model_matrix_storage,
                material_model,
                dirichlet_bc,
                gravity,
                fe_model,
                substep_t,
                substep_dt,
                mass_damping_coefficient,
                stiffness_damping_coefficient,
                DVectorSliceMut::from(&mut u_substep),
                DVectorSliceMut::from(&mut v_substep),
                system_preconditioner,
                workspace,
            );
            if let Err(err) = substep_result {
                // TODO: We currently substep no matter what the error was. Would be good to
                // be able to distinguish recoverable errors from non-recoverable
                num_substeps *= substep_factor;

                if num_substeps < max_substeps {
                    substep_dt = dt / (num_substeps as f64);
                    // Reset state to initial state
                    u_substep.copy_from(&u);
                    v_substep.copy_from(&v);
                    // Restart with new substep parameters
                    warn!(
                        "Integrator failed. Attempting to substep with {} substeps",
                        num_substeps
                    );
                    continue 'outer;
                } else {
                    return Err(Box::from(format!(
                        "Substepping failed.\
                    Could not find solution using {} substeps. Latest error from integrator:\
                    {}",
                        num_substeps, err
                    )));
                }
            }
        }

        // We successfully got through all substeps, so return result
        u.copy_from(&u_substep);
        v.copy_from(&v_substep);
        assert!(u.iter().all(|u_i| u_i.is_finite()));
        assert!(v.iter().all(|v_i| v_i.is_finite()));
        return Ok(());
    }
}

fn integrate_model<'a>(
    model_matrix_storage: &mut ElasticModelMatrixStorage,
    material_model: &(dyn Sync + ElasticMaterialModel<f64, U3>),
    dirichlet_bc: Option<&dyn DirichletBoundaryConditions>,
    gravity: Option<&Gravity>,
    fe_model: &dyn ElasticityModelParallel<f64, U3>,
    t: f64,
    dt: f64,
    mass_damping_coefficient: Option<f64>,
    stiffness_damping_coefficient: Option<f64>,
    mut u: DVectorSliceMut<f64>,
    mut v: DVectorSliceMut<f64>,
    system_preconditioner: &'a mut (dyn SystemPreconditioner + 'a),
    workspace: &mut NewtonCgWorkspace,
) -> Result<(), Box<dyn Error>> {
    let mass_matrix = &model_matrix_storage.mass_matrix;
    let n = mass_matrix.nrows();

    let g_force = if let Some(gravity) = gravity {
        let g_force_density = compute_gravity_density(fe_model, gravity)?;
        mass_matrix * &g_force_density
    } else {
        DVector::zeros(n)
    };
    // TODO: Avoid allocating gravity vector every time step
    workspace.gravity_force.copy_from(&g_force);
    model_matrix_storage.has_damping_matrix = compute_damping_matrix_into(
        fe_model,
        DVectorSlice::from(&u),
        &mut model_matrix_storage.stiffness_matrix,
        &mut model_matrix_storage.damping_matrix,
        &model_matrix_storage.mass_matrix,
        mass_damping_coefficient,
        stiffness_damping_coefficient,
        material_model,
    );

    let representative_force = model_matrix_storage.representative_force;
    let mut system = NewtonCgDynamicSystem {
        model_matrix_storage,
        material_model,
        model: fe_model,
        bc: dirichlet_bc,
        system_preconditioner,
        workspace,
        t,
        alpha: None,
        beta: None,
        stiffness_matrix_outdated: true,
        solve_jacobian_combination_outdated: true,
    };

    let settings = BackwardEulerSettings {
        // TODO: Derive an appropriate tolerance
        tolerance: 1e-5 * representative_force,
        max_newton_iter: Some(100),
    };
    {
        profile!("backward euler");
        let newton_iter = backward_euler_step(&mut system, &mut u, &mut v, t, dt, settings)?;
        info!("Number of newton iterations in Backward Euler step: {}", newton_iter);
    }

    apply_dirichlet_bcs_unknowns(t + dt, &mut u, &mut v, dirichlet_bc);
    Ok(())
}

pub struct SchwarzPreconditioner<'a> {
    matrix: &'a mut CsrMatrix<f64>,
    schwarz_connectivity: &'a NestedVec<usize>,
    colors: &'a [DisjointSubsets],
    untreated_nodes: &'a [usize],
    mkl_matrix: Option<MklMatrix<'a>>,
}

struct MklMatrix<'a> {
    connectivity: Pin<Box<MklCsrConnectivity>>,
    values: Pin<Box<Vec<f64>>>,
    handle: Option<CsrMatrixHandle<'a, f64>>,
}

impl<'a> MklMatrix<'a> {
    pub fn from_matrix<'b>(csr_matrix: &'b CsrMatrix<f64>) -> Self {
        let mut mkl_matrix = MklMatrix {
            connectivity: Box::pin(MklCsrConnectivity::from_csr_pattern(&csr_matrix.sparsity_pattern())),
            values: Box::pin(csr_matrix.values().to_vec()),
            handle: None,
        };
        let connectivity_ptr = mkl_matrix.connectivity.deref() as *const MklCsrConnectivity;
        let connectivity_ref = unsafe { &*connectivity_ptr };
        let values_ptr = mkl_matrix.values.deref() as *const Vec<_>;
        let values_ref = unsafe { &*values_ptr };
        mkl_matrix.handle = Some(connectivity_ref.as_mkl_csr(values_ref.as_slice()));

        mkl_matrix
    }

    pub fn handle(&self) -> &CsrMatrixHandle<'a, f64> {
        self.handle.as_ref().unwrap()
    }
}

impl<'a> Drop for MklMatrix<'a> {
    fn drop(&mut self) {
        // Make sure that the handle gets dropped by removing the contents of the Option
        self.handle.take();
    }
}

impl<'a> LinearOperator<f64> for SchwarzPreconditioner<'a> {
    fn apply(&self, y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        let mkl_matrix = self
            .mkl_matrix
            .as_ref()
            .ok_or_else(|| Box::<dyn Error>::from("MKL Matrix has not yet been constructed"))?;
        MklCsrLinearOperator {
            csr: mkl_matrix.handle(),
        }
        .apply(y, x)
    }
}

impl<'a> SystemPreconditioner for SchwarzPreconditioner<'a> {
    fn update(&mut self, system_matrix: &CsrMatrix<f64>) -> Result<(), Box<dyn Error>> {
        self.matrix.fill_par(0.0);
        build_schwarz_preconditioner_into(
            self.matrix,
            system_matrix,
            self.schwarz_connectivity,
            self.untreated_nodes,
            &self.colors,
            3,
        )?;
        self.mkl_matrix = Some(MklMatrix::from_matrix(&*self.matrix));
        Ok(())
    }

    fn as_linear_operator(&self) -> &dyn LinearOperator<f64> {
        self
    }
}

#[derive(Debug)]
struct NewtonCgWorkspace {
    gravity_force: DVector<f64>,
    u: DVector<f64>,
    v: DVector<f64>,
    cg_workspace: CgWorkspace<f64>,
}

impl NewtonCgWorkspace {
    fn build_workspace(ndof: usize) -> Self {
        Self {
            gravity_force: DVector::zeros(ndof),
            u: DVector::zeros(ndof),
            v: DVector::zeros(ndof),
            cg_workspace: CgWorkspace::default(),
        }
    }

    fn prepare_workspace(&mut self, ndof: usize) {
        self.gravity_force.resize_vertically_mut(ndof, 0.0);
        self.u.resize_vertically_mut(ndof, 0.0);
        self.v.resize_vertically_mut(ndof, 0.0);
    }
}

pub trait SystemPreconditioner {
    fn update(&mut self, system_matrix: &CsrMatrix<f64>) -> Result<(), Box<dyn Error>>;

    fn as_linear_operator(&self) -> &dyn LinearOperator<f64>;
}

impl<'a> SystemPreconditioner for Box<dyn SystemPreconditioner + 'a> {
    fn update(&mut self, system_matrix: &CsrMatrix<f64>) -> Result<(), Box<dyn Error>> {
        self.deref_mut().update(system_matrix)
    }

    fn as_linear_operator(&self) -> &dyn LinearOperator<f64> {
        self.deref().as_linear_operator()
    }
}

struct NewtonCgDynamicSystem<'a> {
    model_matrix_storage: &'a mut ElasticModelMatrixStorage,
    material_model: &'a (dyn Sync + ElasticMaterialModel<f64, U3>),
    model: &'a dyn ElasticityModelParallel<f64, U3>,
    bc: Option<&'a dyn DirichletBoundaryConditions>,
    system_preconditioner: &'a mut dyn SystemPreconditioner,
    workspace: &'a mut NewtonCgWorkspace,
    t: f64,
    alpha: Option<f64>,
    beta: Option<f64>,
    stiffness_matrix_outdated: bool,
    solve_jacobian_combination_outdated: bool,
}

impl<'a> DynamicSystem<f64> for NewtonCgDynamicSystem<'a> {
    fn apply_mass_matrix(&mut self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) {
        profile!("apply mass");
        spmv_csr(1.0, &mut y, 1.0, &self.model_matrix_storage.mass_matrix, &x);
    }

    fn apply_inverse_mass_matrix(
        &mut self,
        _sol: DVectorSliceMut<f64>,
        _rhs: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        unimplemented!()
    }

    fn eval_f(
        &mut self,
        mut f: DVectorSliceMut<f64>,
        t: f64,
        u: DVectorSlice<f64>,
        v: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        // f <- Mg
        f.axpy(1.0, &self.workspace.gravity_force, 0.0);

        // TODO: Avoid copy?
        let mut u = u.clone_owned();
        let mut v = v.clone_owned();

        // Apply boundary conditions to unknowns
        apply_dirichlet_bcs_unknowns(t, DVectorSliceMut::from(&mut u), DVectorSliceMut::from(&mut v), self.bc);

        // f += f_internal
        {
            profile!("assemble elastic forces");
            self.model.assemble_elastic_pseudo_forces_into_par(
                DVectorSliceMut::from(&mut f),
                DVectorSlice::from(&u),
                self.material_model,
            );
        }

        // f += -Dv
        if self.model_matrix_storage.has_damping_matrix {
            spmv_csr(1.0, &mut f, -1.0, &self.model_matrix_storage.damping_matrix, &v);
        }

        // Apply boundary conditions
        let dim = 3;
        apply_homogeneous_dirichlet_bc_rhs(f, self.bc.nodes(), dim);

        Ok(())
    }
}

pub struct BlockDiagonalPreconditioner3d {
    blocks: Vec<Matrix3<f64>>,
}

impl BlockDiagonalPreconditioner3d {
    pub fn from_csr(matrix: &CsrMatrix<f64>) -> Result<Self, Box<dyn Error>> {
        assert_eq!(matrix.nrows(), matrix.ncols());
        assert_eq!(matrix.nrows() % 3, 0);
        let num_blocks = matrix.nrows() / 3;

        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let mut block = Matrix3::zeros();

            for i in 0..3 {
                let row_idx = 3 * block_idx + i;
                let csr_row = matrix.row(row_idx);

                let mut block_row = block.row_mut(i);
                // TODO: This incurs three binary searches, which is unnecessarily expensive
                block_row[(i + 3 - 1) % 3] = csr_row.get(3 * block_idx + (i + 3 - 1) % 3).unwrap_or(0.0);
                block_row[(i + 3 - 0) % 3] = csr_row.get(3 * block_idx + (i + 3 - 0) % 3).unwrap_or(0.0);
                block_row[(i + 3 + 1) % 3] = csr_row.get(3 * block_idx + (i + 3 + 1) % 3).unwrap_or(0.0);
            }

            let block_inverse = block
                .try_inverse()
                .ok_or_else(|| Box::<dyn Error>::from("Matrix has singular blocks"))?;
            blocks.push(block_inverse);
        }

        Ok(Self { blocks })
    }
}

impl LinearOperator<f64> for BlockDiagonalPreconditioner3d {
    fn apply(&self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len() % 3, 0);
        let num_blocks = y.len() / 3;

        for block_idx in 0..num_blocks {
            let block = &self.blocks[block_idx];
            let mut y_block = y.fixed_rows_mut::<U3>(3 * block_idx);
            let x_block = x.fixed_rows::<U3>(3 * block_idx);
            y_block.gemv(1.0, &block, &x_block, 0.0);
        }

        Ok(())
    }
}

pub struct DiagonalPreconditioner {
    diag: DVector<f64>,
}

impl Default for DiagonalPreconditioner {
    fn default() -> Self {
        Self {
            diag: DVector::zeros(0),
        }
    }
}

impl DiagonalPreconditioner {
    pub fn from_csr(matrix: &CsrMatrix<f64>) -> Result<Self, Box<dyn Error>> {
        let diag_entries: Vec<_> = matrix
            .diag_iter()
            .map(|d_i| {
                if d_i != 0.0 {
                    Ok(d_i.recip())
                } else {
                    Err(Box::<dyn Error>::from("Diagonal matrix is not invertible."))
                }
            })
            .collect::<Result<_, _>>()?;

        let diag_storage = VecStorage::new(Dynamic::new(matrix.nrows()), U1, diag_entries);
        Ok(Self {
            diag: DVector::from_data(diag_storage),
        })
    }
}

impl SystemPreconditioner for DiagonalPreconditioner {
    fn update(&mut self, system_matrix: &CsrMatrix<f64>) -> Result<(), Box<dyn Error>> {
        self.diag.resize_vertically_mut(system_matrix.nrows(), 0.0);
        self.diag
            .iter_mut()
            .zip(system_matrix.diag_iter())
            .map(|(d_i, a_ii)| {
                if a_ii != 0.0 {
                    *d_i = a_ii.recip();
                    Ok(())
                } else {
                    Err(Box::<dyn Error>::from("Diagonal matrix is not invertible."))
                }
            })
            .collect::<Result<_, _>>()
    }

    fn as_linear_operator(&self) -> &dyn LinearOperator<f64> {
        self
    }
}

impl LinearOperator<f64> for DiagonalPreconditioner {
    fn apply(&self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        y.zip_zip_apply(&x, &self.diag, |_, x_i, d_i| d_i * x_i);
        Ok(())
    }
}

pub struct MklCsrLinearOperator<'a> {
    csr: &'a CsrMatrixHandle<'a, f64>,
}

impl<'a> LinearOperator<f64> for MklCsrLinearOperator<'a> {
    fn apply(&self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        profile!("MKL sparse matrix-vector multiply");
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.csr.rows());
        assert_eq!(x.len(), self.csr.cols());

        let description = MatrixDescription::default();
        mkl_corrode::sparse::spmv_csr(
            SparseOperation::NonTranspose,
            1.0,
            self.csr,
            &description,
            x.as_slice(),
            0.0,
            y.as_mut_slice(),
        )
        .map_err(|_| Box::<dyn Error>::from("MKL error during sparse spmv"))?;
        Ok(())
    }
}

/// Helper struct to store connectivity of an MKL sparse matrix (needs different index type).
pub struct MklCsrConnectivity {
    rows: usize,
    cols: usize,
    row_offsets: Vec<MKL_INT>,
    column_indices: Vec<MKL_INT>,
}

impl MklCsrConnectivity {
    pub fn from_csr_pattern(pattern: &SparsityPattern) -> Self {
        let to_mkl_int_vec = |indices: &[usize]| {
            indices
                .iter()
                .map(|idx| MKL_INT::try_from(*idx).expect("TODO: Handle properly if indices are too large for MKL_INT"))
                .collect::<Vec<_>>()
        };

        Self {
            row_offsets: to_mkl_int_vec(pattern.major_offsets()),
            column_indices: to_mkl_int_vec(pattern.minor_indices()),
            rows: pattern.major_dim(),
            cols: pattern.minor_dim(),
        }
    }

    pub fn as_mkl_csr<'a>(&'a self, values: &'a [f64]) -> CsrMatrixHandle<'a, f64> {
        assert_eq!(self.column_indices.len(), values.len());
        CsrMatrixHandle::from_csr_data(
            self.rows,
            self.cols,
            &self.row_offsets[..self.rows],
            &self.row_offsets[1..],
            &self.column_indices,
            values,
        )
        .expect("Sparse matrix construction should never fail")
    }
}

impl<'a> DifferentiableDynamicSystem<f64> for NewtonCgDynamicSystem<'a> {
    fn set_state(&mut self, t: f64, u: DVectorSlice<f64>, v: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        self.alpha = None;
        self.beta = None;
        self.t = t;
        self.solve_jacobian_combination_outdated = true;
        self.stiffness_matrix_outdated = true;
        self.workspace.u.copy_from(&u);
        self.workspace.v.copy_from(&v);
        apply_dirichlet_bcs_unknowns(t, &mut self.workspace.u, &mut self.workspace.v, self.bc);
        Ok(())
    }

    fn init_solve_jacobian_combination(&mut self, alpha: Option<f64>, beta: Option<f64>) -> Result<(), Box<dyn Error>> {
        self.alpha = alpha;
        self.beta = beta;
        self.solve_jacobian_combination_outdated = true;
        Ok(())
    }

    fn solve_jacobian_combination(
        &mut self,
        mut sol: DVectorSliceMut<f64>,
        rhs: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        profile!("solve jacobian combination");
        let matrices = &mut self.model_matrix_storage;

        if self.solve_jacobian_combination_outdated {
            if self.stiffness_matrix_outdated {
                let projected = false;
                compute_stiffness_matrix_into(
                    self.model,
                    self.t,
                    DVectorSlice::from(&self.workspace.u),
                    DVectorSlice::from(&self.workspace.v),
                    &mut matrices.stiffness_matrix,
                    self.bc,
                    self.material_model,
                    projected,
                );
                self.stiffness_matrix_outdated = false;
            }
            let damping_matrix = if matrices.has_damping_matrix {
                Some(&matrices.damping_matrix)
            } else {
                None
            };
            compute_jacobian_combination_into::<U3>(
                &mut matrices.linear_combination_solve,
                &matrices.stiffness_matrix,
                damping_matrix,
                &matrices.mass_matrix,
                self.alpha,
                self.beta,
                Some(1.0),
                self.bc,
            );

            {
                profile!("update preconditioner");
                self.system_preconditioner
                    .update(&matrices.linear_combination_solve)?;
            }
            self.solve_jacobian_combination_outdated = false;
        }

        {
            profile!("conjugate gradient");
            let system_matrix = &matrices.linear_combination_solve;

            let mkl_connectivity = MklCsrConnectivity::from_csr_pattern(&system_matrix.sparsity_pattern());
            let mkl_system_matrix = mkl_connectivity.as_mkl_csr(system_matrix.values());

            // TODO: Move this into the linear operator construction?
            {
                profile!("MKL sparse analysis");
                mkl_system_matrix
                    .set_mv_hint(SparseOperation::NonTranspose, &MatrixDescription::default(), 50)
                    .map_err(|_| Box::<dyn Error>::from("MKL error during set_mv_hint"))?;
                mkl_system_matrix
                    .optimize()
                    .map_err(|_| Box::<dyn Error>::from("MKL error during set_mv_hint"))?;
            }

            let mkl_linear_operator = MklCsrLinearOperator {
                csr: &mkl_system_matrix,
            };

            // TODO: Use block diag preconditioner instead? From some quick tests
            // it actually seems to perform worse than the diagonal preconditioner though
            // let block_diag_preconditioner = BlockDiagonalPreconditioner3d::from_csr(&system_matrix)?;

            // TODO: Consider using a different initial guess than zero?
            sol.fill(0.0);

            // TODO: Would be nice to have an adaptive criterion which would be scale-invariant,
            // see e.g. Walker and Eisenstat's work on inexact Newton methods,
            // or better yet we would use a criterion like that proposed by Nash,
            // but then we'd need to additionally know the scalar value of the
            // scalar potential implicitly being optimal
            let residual_eps = 1e-5;

            let result = {
                profile!("cg solve");
                ConjugateGradient::with_workspace(&mut self.workspace.cg_workspace)
                    .with_operator(mkl_linear_operator)
                    // TODO: Use a more effective diagonal preconditioner
                    .with_preconditioner(self.system_preconditioner.as_linear_operator())
                    // .with_preconditioner(&block_diag_preconditioner)
                    // TODO: Use different stopping criterion?
                    .with_stopping_criterion(RelativeResidualCriterion::new(residual_eps))
                    .with_max_iter(500)
                    .solve_with_guess(&rhs, &mut sol)
            };

            use fenris::cg::SolveErrorKind::*;
            let output = match result {
                Ok(output) => output,
                Err(error) => {
                    match error.kind {
                        MaxIterationsReached { max_iter } => {
                            // We don't abort if we reach maximum number of iterations
                            // (Newton might still converge), but we make sure to log it.
                            warn!("CG reached maximum number of iterations ({}).", max_iter);
                        }
                        IndefiniteOperator => {
                            // We can still use the step produced by CG up to this point,
                            // because it is still ensured to be a descent direction.
                            warn!("CG encountered indefinite operator.");
                        }
                        // TODO: We don't expect preconditioner to be indefinite at the moment,
                        // so we treat that as an actual error
                        _ => return Err(Box::new(error)),
                    }
                    error.output
                }
            };

            info!("CG iterations: {}", output.num_iterations);
        }
        Ok(())
    }

    fn apply_jacobian_combination(
        &mut self,
        _y: DVectorSliceMut<f64>,
        _x: DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        unimplemented!()
    }

    fn init_apply_jacobian_combination(
        &mut self,
        _alpha: Option<f64>,
        _beta: Option<f64>,
        _gamma: Option<f64>,
    ) -> Result<(), Box<dyn Error>> {
        unimplemented!()
    }
}

// TODO: Move these to separate folder?
#[cfg(test)]
mod tests {
    use super::*;
    use fenris::nalgebra::DMatrix;
    #[test]
    fn build_block_preconditioner() {
        let matrix = DMatrix::from_row_slice(
            9,
            9,
            &[
                26.0, 4.0, 12.0, -5.0, -7.0, 1.0, 2.0, 14.0, 11.0, 4.0, 23.0, 0.0, 2.0, -15.0, -11.0, 22.0, 0.0, 5.0,
                12.0, 0.0, 37.0, 6.0, -10.0, -13.0, 6.0, -4.0, 5.0, -5.0, 2.0, 6.0, 36.0, -8.0, 2.0, 5.0, -1.0, 6.0,
                -7.0, -15.0, -10.0, -8.0, 24.0, -4.0, -19.0, -2.0, -13.0, 1.0, -11.0, -13.0, 2.0, -4.0, 31.0, -10.0,
                7.0, 4.0, 2.0, 22.0, 6.0, 5.0, -19.0, -10.0, 33.0, -7.0, 0.0, 14.0, 0.0, -4.0, -1.0, -2.0, 7.0, -7.0,
                37.0, 9.0, 11.0, 5.0, 5.0, 6.0, -13.0, 4.0, 0.0, 9.0, 26.0,
            ],
        );

        let csr = CsrMatrix::from(&matrix);
        let preconditioner = BlockDiagonalPreconditioner3d::from_csr(&csr).unwrap();

        let expected_block1 = Matrix3::new(26.0, 4.0, 12.0, 4.0, 23.0, 0.0, 12.0, 0.0, 37.0)
            .try_inverse()
            .unwrap();
        let expected_block2 = Matrix3::new(36.0, -8.0, 2.0, -8.0, 24.0, -4.0, 2.0, -4.0, 31.0)
            .try_inverse()
            .unwrap();
        let expected_block3 = Matrix3::new(33.0, -7.0, 0.0, -7.0, 37.0, 9.0, 0.0, 9.0, 26.0)
            .try_inverse()
            .unwrap();

        assert!((preconditioner.blocks[0] - expected_block1).norm() < 1e-13);
        assert!((preconditioner.blocks[1] - expected_block2).norm() < 1e-13);
        assert!((preconditioner.blocks[2] - expected_block3).norm() < 1e-13);

        let x = DVector::from_column_slice(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
        let mut y = DVector::from_column_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        preconditioner
            .apply(DVectorSliceMut::from(&mut y), DVectorSlice::from(&x))
            .unwrap();

        let mut y_expected = DVector::zeros(9);
        y_expected
            .index_mut((0..3, 0))
            .copy_from(&(&expected_block1 * x.index((0..3, 0))));
        y_expected
            .index_mut((3..6, 0))
            .copy_from(&(&expected_block2 * x.index((3..6, 0))));
        y_expected
            .index_mut((6..9, 0))
            .copy_from(&(&expected_block3 * x.index((6..9, 0))));

        assert!((y - y_expected).norm() < 1e-13);
    }
}
