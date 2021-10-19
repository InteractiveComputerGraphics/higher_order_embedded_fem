use std::fmt::Debug;
use std::ops::Add;

use coarse_prof::profile;
use fenris::assembly::{apply_homogeneous_dirichlet_bc_csr, ElementMatrixTransformation};
use fenris::connectivity::{
    Hex20Connectivity, Hex27Connectivity, Hex8Connectivity, Segment2d2Connectivity, Tet10Connectivity, Tet4Connectivity,
};
use fenris::embedding::{
    EmbeddedModel3d, EmbeddedQuad4Model, EmbeddedQuad9Model, EmbeddedTri3Model, EmbeddedTri6Model,
};
use fenris::mesh::ClosedSurfaceMesh2d;
use fenris::model::{FiniteElementInterpolator, NodalModel3d, Quad4Model, Quad9Model, Tri3d2Model, Tri6d2Model};
use fenris::nalgebra::{
    DVector, DVectorSlice, DVectorSliceMut, Matrix2, Matrix3, UnitQuaternion, Vector2, Vector3, U2, U3,
};
use fenris::solid::assembly::ScalarMaterialSpaceFunction;
use fenris::solid::materials::{
    CorotatedLinearElasticMaterial, InvertibleMaterial, LinearElasticMaterial, ProjectedStableNeoHookeanMaterial,
    StVKMaterial, StableNeoHookeanMaterial, YoungPoisson,
};
use fenris::solid::{ElasticityModel, ElasticityModelParallel};
use fenris::{solid, CooMatrix, CsrMatrix};
use hamilton::storages::VecStorage;
use hamilton::Component;
use log::info;
use mkl_corrode::dss;
use serde::{Deserialize, Serialize};

use crate::components::{VolumeMesh2d, VolumeMesh3d};
use crate::fem::bcs::{DirichletBoundaryConditions, OptionalDirichletBoundaryConditions};
use crate::fem::IntegrationMethod;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElasticMaterialModel {
    LinearElastic(LinearElasticMaterial<f64>),
    CorotatedLinearElastic(CorotatedLinearElasticMaterial<f64>),
    StableNeoHookean(StableNeoHookeanMaterial<f64>),
    ProjectedStableNeoHookean(ProjectedStableNeoHookeanMaterial<f64>),
    StVenantKirchhoff(StVKMaterial<f64>),
    InvertibleLinearElastic(InvertibleMaterial<f64, LinearElasticMaterial<f64>>),
    InvertibleStableNeoHookean(InvertibleMaterial<f64, StableNeoHookeanMaterial<f64>>),
    InvertibleStVenantKirchhoff(InvertibleMaterial<f64, StVKMaterial<f64>>),
}

/// Helper macro to call e.g. a generic method on the specific underlying material type of
/// an instance of `ElasticMaterialModel`.
#[macro_export]
macro_rules! match_on_elastic_material_model {
    ($on_model:expr, $model:ident => $e:expr) => {{
        use $crate::fem::ElasticMaterialModel::*;
        match $on_model {
            LinearElastic(ref $model) => $e,
            CorotatedLinearElastic(ref $model) => $e,
            StableNeoHookean(ref $model) => $e,
            ProjectedStableNeoHookean(ref $model) => $e,
            StVenantKirchhoff(ref $model) => $e,
            InvertibleLinearElastic(ref $model) => $e,
            InvertibleStableNeoHookean(ref $model) => $e,
            InvertibleStVenantKirchhoff(ref $model) => $e,
        }
    }};
}

impl fenris::solid::ElasticMaterialModel<f64, U2> for ElasticMaterialModel {
    fn compute_strain_energy_density(&self, deformation_gradient: &Matrix2<f64>) -> f64 {
        match_on_elastic_material_model!(self,
            material => material.compute_strain_energy_density(deformation_gradient))
    }

    fn compute_stress_tensor(&self, deformation_gradient: &Matrix2<f64>) -> Matrix2<f64> {
        match_on_elastic_material_model!(self,
            material => material.compute_stress_tensor(deformation_gradient))
    }

    fn contract_stress_tensor_with(
        &self,
        deformation_gradient: &Matrix2<f64>,
        a: &Vector2<f64>,
        b: &Vector2<f64>,
    ) -> Matrix2<f64> {
        match_on_elastic_material_model!(self,
            material => material.contract_stress_tensor_with(deformation_gradient, a, b))
    }
}

impl fenris::solid::ElasticMaterialModel<f64, U3> for ElasticMaterialModel {
    fn compute_strain_energy_density(&self, deformation_gradient: &Matrix3<f64>) -> f64 {
        match_on_elastic_material_model!(self,
            material => material.compute_strain_energy_density(deformation_gradient))
    }

    fn compute_stress_tensor(&self, deformation_gradient: &Matrix3<f64>) -> Matrix3<f64> {
        match_on_elastic_material_model!(self,
            material => material.compute_stress_tensor(deformation_gradient))
    }

    fn contract_stress_tensor_with(
        &self,
        deformation_gradient: &Matrix3<f64>,
        a: &Vector3<f64>,
        b: &Vector3<f64>,
    ) -> Matrix3<f64> {
        match_on_elastic_material_model!(self,
            material => material.contract_stress_tensor_with(deformation_gradient, a, b))
    }
}

macro_rules! impl_from_material {
    ($material:ty, $invariant:ident) => {
        impl From<$material> for ElasticMaterialModel {
            fn from(material: $material) -> Self {
                use ElasticMaterialModel::*;
                $invariant(material)
            }
        }
    };
}

impl_from_material!(StableNeoHookeanMaterial<f64>, StableNeoHookean);
impl_from_material!(ProjectedStableNeoHookeanMaterial<f64>, ProjectedStableNeoHookean);
impl_from_material!(StVKMaterial<f64>, StVenantKirchhoff);
impl_from_material!(LinearElasticMaterial<f64>, LinearElastic);
impl_from_material!(CorotatedLinearElasticMaterial<f64>, CorotatedLinearElastic);
impl_from_material!(InvertibleMaterial<f64, LinearElasticMaterial<f64>>, InvertibleLinearElastic);
impl_from_material!(InvertibleMaterial<f64, StVKMaterial<f64>>, InvertibleStVenantKirchhoff);
impl_from_material!(InvertibleMaterial<f64, StableNeoHookeanMaterial<f64>>,
    InvertibleStableNeoHookean);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub density: f64,
    pub mass_damping_coefficient: Option<f64>,
    pub stiffness_damping_coefficient: Option<f64>,
    pub elastic_model: ElasticMaterialModel,
}

pub fn default_young_poisson() -> YoungPoisson<f64> {
    YoungPoisson {
        young: 1e5,
        poisson: 0.4,
    }
}

impl Default for Material {
    fn default() -> Self {
        let material = StableNeoHookeanMaterial::from(default_young_poisson());
        let inversion_threshold = 1e-6;
        let material = InvertibleMaterial::new(material, inversion_threshold);
        Self {
            /// Default density is adapted to 3D density for water, e.g. 1000 kg/m3
            density: 1000.0,
            mass_damping_coefficient: None,
            stiffness_damping_coefficient: None,
            elastic_model: ElasticMaterialModel::from(material),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FiniteElementModel2d {
    // TODO: Implement more models
    Tri3d2NodalModel(Tri3d2Model<f64>),
    Tri6d2NodalModel(Tri6d2Model<f64>),
    Quad4NodalModel(Quad4Model<f64>),
    Quad9NodalModel(Quad9Model<f64>),
    EmbeddedTri3d2Model(EmbeddedTri3Model<f64>),
    EmbeddedTri6d2Model(EmbeddedTri6Model<f64>),
    EmbeddedQuad4Model(EmbeddedQuad4Model<f64>),
    EmbeddedQuad9Model(EmbeddedQuad9Model<f64>),
}

impl From<Tri3d2Model<f64>> for FiniteElementModel2d {
    fn from(model: Tri3d2Model<f64>) -> Self {
        Self::Tri3d2NodalModel(model)
    }
}

impl From<Tri6d2Model<f64>> for FiniteElementModel2d {
    fn from(model: Tri6d2Model<f64>) -> Self {
        Self::Tri6d2NodalModel(model)
    }
}

impl From<Quad4Model<f64>> for FiniteElementModel2d {
    fn from(model: Quad4Model<f64>) -> Self {
        Self::Quad4NodalModel(model)
    }
}

impl From<Quad9Model<f64>> for FiniteElementModel2d {
    fn from(model: Quad9Model<f64>) -> Self {
        Self::Quad9NodalModel(model)
    }
}

impl From<EmbeddedTri3Model<f64>> for FiniteElementModel2d {
    fn from(model: EmbeddedTri3Model<f64>) -> Self {
        Self::EmbeddedTri3d2Model(model)
    }
}

impl From<EmbeddedTri6Model<f64>> for FiniteElementModel2d {
    fn from(model: EmbeddedTri6Model<f64>) -> Self {
        Self::EmbeddedTri6d2Model(model)
    }
}

impl From<EmbeddedQuad4Model<f64>> for FiniteElementModel2d {
    fn from(model: EmbeddedQuad4Model<f64>) -> Self {
        Self::EmbeddedQuad4Model(model)
    }
}

impl From<EmbeddedQuad9Model<f64>> for FiniteElementModel2d {
    fn from(model: EmbeddedQuad9Model<f64>) -> Self {
        Self::EmbeddedQuad9Model(model)
    }
}

/// Helper macro to call e.g. a generic method on the specific underlying model type of
/// an instance of `FiniteElementModel`.
#[macro_export]
macro_rules! match_on_finite_element_model_2d {
    ($on_model:expr, $model:ident => $e:expr) => {{
        use $crate::fem::FiniteElementModel2d::*;
        match $on_model {
            Tri3d2NodalModel(ref $model) => $e,
            Tri6d2NodalModel(ref $model) => $e,
            Quad4NodalModel(ref $model) => $e,
            Quad9NodalModel(ref $model) => $e,
            EmbeddedTri3d2Model(ref $model) => $e,
            EmbeddedTri6d2Model(ref $model) => $e,
            EmbeddedQuad4Model(ref $model) => $e,
            EmbeddedQuad9Model(ref $model) => $e,
        }
    }};
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FiniteElementModel3d {
    // TODO: Implement more models
    Hex8NodalModel(NodalModel3d<f64, Hex8Connectivity>),
    Hex20NodalModel(NodalModel3d<f64, Hex20Connectivity>),
    Hex27NodalModel(NodalModel3d<f64, Hex27Connectivity>),
    Tet4NodalModel(NodalModel3d<f64, Tet4Connectivity>),
    Tet10NodalModel(NodalModel3d<f64, Tet10Connectivity>),
    EmbeddedHex8Model(EmbeddedModel3d<f64, Hex8Connectivity>),
    EmbeddedHex20Model(EmbeddedModel3d<f64, Hex20Connectivity>),
    EmbeddedHex27Model(EmbeddedModel3d<f64, Hex27Connectivity>),
    EmbeddedTet4Model(EmbeddedModel3d<f64, Tet4Connectivity>),
    EmbeddedTet10Model(EmbeddedModel3d<f64, Tet10Connectivity>),
}

#[macro_export]
macro_rules! match_on_nodal_element_model_3d {
    ($on_model:expr, $model:ident => $e:expr, $fallback:expr) => {{
        use $crate::fem::FiniteElementModel3d::*;
        match $on_model {
            Hex8NodalModel(ref $model) => $e,
            Hex20NodalModel(ref $model) => $e,
            Hex27NodalModel(ref $model) => $e,
            Tet4NodalModel(ref $model) => $e,
            Tet10NodalModel(ref $model) => $e,
            EmbeddedHex8Model(_) => $fallback,
            EmbeddedHex20Model(_) => $fallback,
            EmbeddedHex27Model(_) => $fallback,
            EmbeddedTet4Model(_) => $fallback,
            EmbeddedTet10Model(_) => $fallback,
        }
    }};
}

#[macro_export]
macro_rules! match_on_finite_element_model_3d {
    ($on_model:expr, $model:ident => $e:expr) => {{
        use $crate::fem::FiniteElementModel3d::*;
        match $on_model {
            Hex8NodalModel(ref $model) => $e,
            Hex20NodalModel(ref $model) => $e,
            Hex27NodalModel(ref $model) => $e,
            Tet4NodalModel(ref $model) => $e,
            Tet10NodalModel(ref $model) => $e,
            EmbeddedHex8Model(ref $model) => $e,
            EmbeddedHex20Model(ref $model) => $e,
            EmbeddedHex27Model(ref $model) => $e,
            EmbeddedTet4Model(ref $model) => $e,
            EmbeddedTet10Model(ref $model) => $e,
        }
    }};
}

impl From<NodalModel3d<f64, Hex8Connectivity>> for FiniteElementModel3d {
    fn from(model: NodalModel3d<f64, Hex8Connectivity>) -> Self {
        Self::Hex8NodalModel(model)
    }
}

impl From<NodalModel3d<f64, Hex20Connectivity>> for FiniteElementModel3d {
    fn from(model: NodalModel3d<f64, Hex20Connectivity>) -> Self {
        Self::Hex20NodalModel(model)
    }
}

impl From<NodalModel3d<f64, Hex27Connectivity>> for FiniteElementModel3d {
    fn from(model: NodalModel3d<f64, Hex27Connectivity>) -> Self {
        Self::Hex27NodalModel(model)
    }
}

impl From<NodalModel3d<f64, Tet4Connectivity>> for FiniteElementModel3d {
    fn from(model: NodalModel3d<f64, Tet4Connectivity>) -> Self {
        Self::Tet4NodalModel(model)
    }
}

impl From<NodalModel3d<f64, Tet10Connectivity>> for FiniteElementModel3d {
    fn from(model: NodalModel3d<f64, Tet10Connectivity>) -> Self {
        Self::Tet10NodalModel(model)
    }
}

impl From<EmbeddedModel3d<f64, Hex8Connectivity>> for FiniteElementModel3d {
    fn from(model: EmbeddedModel3d<f64, Hex8Connectivity>) -> Self {
        Self::EmbeddedHex8Model(model)
    }
}

impl From<EmbeddedModel3d<f64, Hex20Connectivity>> for FiniteElementModel3d {
    fn from(model: EmbeddedModel3d<f64, Hex20Connectivity>) -> Self {
        Self::EmbeddedHex20Model(model)
    }
}

impl From<EmbeddedModel3d<f64, Hex27Connectivity>> for FiniteElementModel3d {
    fn from(model: EmbeddedModel3d<f64, Hex27Connectivity>) -> Self {
        Self::EmbeddedHex27Model(model)
    }
}

impl From<EmbeddedModel3d<f64, Tet4Connectivity>> for FiniteElementModel3d {
    fn from(model: EmbeddedModel3d<f64, Tet4Connectivity>) -> Self {
        Self::EmbeddedTet4Model(model)
    }
}

impl From<EmbeddedModel3d<f64, Tet10Connectivity>> for FiniteElementModel3d {
    fn from(model: EmbeddedModel3d<f64, Tet10Connectivity>) -> Self {
        Self::EmbeddedTet10Model(model)
    }
}

#[derive(Debug)]
pub struct ElasticModelMatrixStorage {
    pub mass_matrix: CsrMatrix<f64>,
    // TODO: Better solution than a separate bool. Something like an Option
    //  that keeps it's value when going from Some -> None -> Some.
    pub has_damping_matrix: bool,
    pub damping_matrix: CsrMatrix<f64>,
    pub stiffness_matrix: CsrMatrix<f64>,
    pub linear_combination_apply: CsrMatrix<f64>,
    pub linear_combination_solve: CsrMatrix<f64>,

    /// Used for stopping criteria of integrators,
    /// e.g. ||M v_dot - f|| <= eps * representative_force
    /// TODO: Move this somewhere else
    pub representative_force: f64,
}

impl ElasticModelMatrixStorage {
    fn new_from_mass(mass: CsrMatrix<f64>, representative_force: f64) -> Self {
        let mut damping_matrix = mass.clone();
        damping_matrix.fill_par(0.0);

        let stiffness_matrix = damping_matrix.clone();
        let linear_combination_apply = damping_matrix.clone();
        let linear_combination_solve = damping_matrix.clone();

        ElasticModelMatrixStorage {
            mass_matrix: mass,
            has_damping_matrix: false,
            damping_matrix,
            stiffness_matrix,
            linear_combination_apply,
            linear_combination_solve,
            representative_force,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FiniteElementElasticModel2d {
    pub model: FiniteElementModel2d,
    pub u: DVector<f64>,
    pub v: DVector<f64>,
    pub material_volume_mesh: VolumeMesh2d,
    pub material_volume_interpolator: FiniteElementInterpolator<f64>,

    #[serde(skip)]
    // Cache the factorization here so that we can reuse the symbolic factorization
    // Any system making changes to topology should set this variable to `None`
    // to force a refactorization
    pub factorization: Option<dss::Solver<f64>>,

    // Material surface is used for deforming contact geometry,
    // so the surface mesh must be compatible with contact geometry
    pub material_surface: Option<ClosedSurfaceMesh2d<f64, Segment2d2Connectivity>>,
    pub material_surface_interpolator: Option<FiniteElementInterpolator<f64>>,
    pub material: Material,

    pub integrator: IntegrationMethod,

    // TODO: All these should be stored in a separate components
    pub gravity_enabled: bool,
    #[serde(skip)]
    pub model_matrix_storage: Option<ElasticModelMatrixStorage>,
}

impl Component for FiniteElementElasticModel2d {
    type Storage = VecStorage<Self>;
}

fn compute_representative_force(mass_matrix: &CsrMatrix<f64>, dim: usize) -> f64 {
    // TODO: Don't hardcode this
    let representative_acceleration_norm = 10.0;
    let a = representative_acceleration_norm;
    // This is just an algebraic trick to ensure that the acceleration vector
    // is consistent independent of direction, but still has the correct acceleration norm
    let representative_acceleration = DVector::repeat(mass_matrix.nrows(), ((a * a) / dim as f64).sqrt());
    (mass_matrix * &representative_acceleration).norm()
}

impl FiniteElementElasticModel2d {
    pub fn ensure_model_matrix_storage_initialized(&mut self, dirichlet_bcs: Option<&dyn DirichletBoundaryConditions>) {
        if self.model_matrix_storage.is_none() {
            let mut mass_matrix = self
                .model
                .assemble_mass(self.material.density)
                .to_csr(Add::add);
            apply_homogeneous_dirichlet_bc_csr::<f64, U2>(&mut mass_matrix, dirichlet_bcs.nodes());
            info!(
                "Assembled mass matrix ({} x {}): {} nnz",
                mass_matrix.nrows(),
                mass_matrix.ncols(),
                mass_matrix.nnz()
            );

            let representative_force = compute_representative_force(&mass_matrix, 2);
            self.model_matrix_storage = Some(ElasticModelMatrixStorage::new_from_mass(
                mass_matrix,
                representative_force,
            ));
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FiniteElementElasticModel3d {
    pub model: FiniteElementModel3d,
    pub u: DVector<f64>,
    pub v: DVector<f64>,

    #[serde(skip)]
    // Cache the factorization here so that we can reuse the symbolic factorization
    // Any system making changes to topology should set this variable to `None`
    // to force a refactorization
    pub factorization: Option<dss::Solver<f64>>,

    pub material_volume_mesh: VolumeMesh3d,
    pub material_volume_interpolator: FiniteElementInterpolator<f64>,

    pub material: Material,

    pub integrator: IntegrationMethod,

    // TODO: All these should be stored in a separate components
    pub gravity_enabled: bool,

    #[serde(skip)]
    pub rotations: Option<Vec<UnitQuaternion<f64>>>,
    #[serde(skip)]
    pub model_matrix_storage: Option<ElasticModelMatrixStorage>,
    // TODO: Consider splitting up the FE model components into several components,
    //  since some components can be re-used across 2d and 3d (material, u, v, static_nodes),
    //  and it might anyway be convenient to store the material volume/surface meshes
    //  as optional components of the respective volume mesh/contact geometry components.
}

impl FiniteElementElasticModel3d {
    pub fn ensure_model_matrix_storage_initialized(&mut self, dirichlet_bcs: Option<&dyn DirichletBoundaryConditions>) {
        if self.model_matrix_storage.is_none() {
            let mass_matrix = {
                profile!("assemble mass matrix");
                let mut mass_matrix = self
                    .model
                    .assemble_mass(self.material.density)
                    .to_csr(Add::add);
                apply_homogeneous_dirichlet_bc_csr::<f64, U3>(&mut mass_matrix, dirichlet_bcs.nodes());
                mass_matrix
            };
            info!(
                "Assembled mass matrix ({} x {}): {} nnz",
                mass_matrix.nrows(),
                mass_matrix.ncols(),
                mass_matrix.nnz()
            );

            let representative_force = compute_representative_force(&mass_matrix, 3);
            self.model_matrix_storage = Some(ElasticModelMatrixStorage::new_from_mass(
                mass_matrix,
                representative_force,
            ));
        }
    }
}

impl Component for FiniteElementElasticModel3d {
    type Storage = VecStorage<Self>;
}

impl ElasticityModel<f64, U2> for FiniteElementModel2d {
    fn ndof(&self) -> usize {
        match_on_finite_element_model_2d!(self, model => model.ndof())
    }

    fn assemble_stiffness_into(
        &self,
        csr: &mut CsrMatrix<f64>,
        u: &DVector<f64>,
        material_model: &dyn solid::ElasticMaterialModel<f64, U2>,
    ) {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_stiffness_into(csr, u, material_model))
    }

    fn assemble_stiffness(
        &self,
        u: &DVector<f64>,
        material_model: &dyn solid::ElasticMaterialModel<f64, U2>,
    ) -> CooMatrix<f64> {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_stiffness(u, material_model))
    }

    fn assemble_mass(&self, density: f64) -> CooMatrix<f64> {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_mass(density))
    }

    fn assemble_elastic_pseudo_forces(
        &self,
        u: DVectorSlice<f64>,
        material_model: &dyn solid::ElasticMaterialModel<f64, U2>,
    ) -> DVector<f64> {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_elastic_pseudo_forces(u, material_model))
    }

    fn compute_scalar_element_integrals(
        &self,
        u: DVectorSlice<f64>,
        integrand: &dyn ScalarMaterialSpaceFunction<f64, U2, U2>,
    ) -> DVector<f64> {
        match_on_finite_element_model_2d!(self,
            model => model.compute_scalar_element_integrals(u, integrand))
    }
}

impl ElasticityModel<f64, U3> for FiniteElementModel3d {
    fn ndof(&self) -> usize {
        match_on_finite_element_model_3d!(self, model => model.ndof())
    }

    fn assemble_stiffness_into(
        &self,
        csr: &mut CsrMatrix<f64>,
        u: &DVector<f64>,
        material_model: &dyn solid::ElasticMaterialModel<f64, U3>,
    ) {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_stiffness_into(csr, u, material_model))
    }

    fn assemble_stiffness(
        &self,
        u: &DVector<f64>,
        material_model: &dyn solid::ElasticMaterialModel<f64, U3>,
    ) -> CooMatrix<f64> {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_stiffness(u, material_model))
    }

    fn assemble_mass(&self, density: f64) -> CooMatrix<f64> {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_mass(density))
    }

    fn assemble_elastic_pseudo_forces(
        &self,
        u: DVectorSlice<f64>,
        material_model: &dyn solid::ElasticMaterialModel<f64, U3>,
    ) -> DVector<f64> {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_elastic_pseudo_forces(u, material_model))
    }

    fn compute_scalar_element_integrals(
        &self,
        u: DVectorSlice<f64>,
        integrand: &dyn ScalarMaterialSpaceFunction<f64, U3, U3>,
    ) -> DVector<f64> {
        match_on_finite_element_model_3d!(self,
            model => model.compute_scalar_element_integrals(u, integrand))
    }
}

impl ElasticityModelParallel<f64, U2> for FiniteElementModel2d {
    fn assemble_elastic_pseudo_forces_into_par(
        &self,
        f: DVectorSliceMut<f64>,
        u: DVectorSlice<f64>,
        material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, U2>),
    ) {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_elastic_pseudo_forces_into_par(f, u, material_model))
    }

    fn assemble_transformed_stiffness_par(
        &self,
        u: &DVector<f64>,
        material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, U2>),
        transformation: &(dyn Sync + ElementMatrixTransformation<f64>),
    ) -> CooMatrix<f64> {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_transformed_stiffness_par(u, material_model, transformation))
    }

    fn assemble_transformed_stiffness_into_par(
        &self,
        csr: &mut CsrMatrix<f64>,
        u: &DVector<f64>,
        material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, U2>),
        transformation: &(dyn Sync + ElementMatrixTransformation<f64>),
    ) {
        match_on_finite_element_model_2d!(self,
            model => model.assemble_transformed_stiffness_into_par(
                            csr, u, material_model, transformation))
    }

    fn compute_scalar_element_integrals_par(
        &self,
        u: DVectorSlice<f64>,
        integrand: &(dyn Sync + ScalarMaterialSpaceFunction<f64, U2, U2>),
    ) -> DVector<f64> {
        match_on_finite_element_model_2d!(self,
            model => model.compute_scalar_element_integrals_par(u, integrand))
    }
}

impl ElasticityModelParallel<f64, U3> for FiniteElementModel3d {
    fn assemble_elastic_pseudo_forces_into_par(
        &self,
        f: DVectorSliceMut<f64>,
        u: DVectorSlice<f64>,
        material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, U3>),
    ) {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_elastic_pseudo_forces_into_par(f, u, material_model))
    }

    fn assemble_transformed_stiffness_par(
        &self,
        u: &DVector<f64>,
        material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, U3>),
        transformation: &(dyn Sync + ElementMatrixTransformation<f64>),
    ) -> CooMatrix<f64> {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_transformed_stiffness_par(u, material_model, transformation))
    }

    fn assemble_transformed_stiffness_into_par(
        &self,
        csr: &mut CsrMatrix<f64>,
        u: &DVector<f64>,
        material_model: &(dyn Sync + solid::ElasticMaterialModel<f64, U3>),
        transformation: &(dyn Sync + ElementMatrixTransformation<f64>),
    ) {
        match_on_finite_element_model_3d!(self,
            model => model.assemble_transformed_stiffness_into_par(
                            csr, u, material_model, transformation))
    }

    fn compute_scalar_element_integrals_par(
        &self,
        u: DVectorSlice<f64>,
        integrand: &(dyn Sync + ScalarMaterialSpaceFunction<f64, U3, U3>),
    ) -> DVector<f64> {
        match_on_finite_element_model_3d!(self,
            model => model.compute_scalar_element_integrals_par(u, integrand))
    }
}
