mod embedding2d;
mod embedding3d;
mod quadrature_reduction;

pub use embedding2d::*;
pub use embedding3d::*;
pub use quadrature_reduction::*;

use crate::allocators::{FiniteElementMatrixAllocator, VolumeFiniteElementAllocator};
use crate::assembly::{color_nodes, ElementMatrixTransformation};
use crate::connectivity::{CellConnectivity, Connectivity, ConnectivityMut};
use crate::element::ElementConnectivity;
use crate::geometry::{Distance, DistanceQuery, GeometryCollection};
use crate::mesh::{Mesh, Mesh3d};
use crate::model::{FiniteElementInterpolator, MakeInterpolator};
use crate::quadrature::QuadraturePair;
use crate::solid::assembly::{
    assemble_mass_into, assemble_pseudo_forces_into, assemble_pseudo_forces_into_par, assemble_stiffness_into,
    assemble_stiffness_into_csr, assemble_transformed_stiffness_into_csr_par, assemble_transformed_stiffness_par,
    ScalarMaterialSpaceFunction,
};
use crate::solid::{ElasticMaterialModel, ElasticityModel, ElasticityModelParallel};
use crate::{CooMatrix, CsrMatrix};
use nalgebra::allocator::Allocator;
use nalgebra::{
    DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimMin, DimName, DimNameMul, Point, RealField, Scalar,
    VectorN, U2, U3,
};
use paradis::DisjointSubsets;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize,\
                Connectivity: Serialize,\
                <DefaultAllocator as Allocator<T, D>>::Buffer: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>,\
                   Connectivity: Deserialize<'de>,\
                   <DefaultAllocator as Allocator<T, D>>::Buffer: Deserialize<'de>"))]
pub struct EmbeddedModel<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    // Store all interior connectivity first, then interface connectivity
    background_mesh: Mesh<T, D, Connectivity>,
    // Number of interior connectivities
    num_interior: usize,

    // Colors for parallel assembly
    interior_colors: Vec<DisjointSubsets>,
    interface_colors: Vec<DisjointSubsets>,

    mass_quadrature: Option<EmbeddedQuadrature<T, D>>,
    stiffness_quadrature: Option<EmbeddedQuadrature<T, D>>,
    elliptic_quadrature: Option<EmbeddedQuadrature<T, D>>,

    mass_regularization_factor: T,
}

pub type EmbeddedModel2d<T, C> = EmbeddedModel<T, U2, C>;
pub type EmbeddedModel3d<T, C> = EmbeddedModel<T, U3, C>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize, VectorN<T, D>: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>, VectorN<T, D>: Deserialize<'de>"))]
pub struct EmbeddedQuadrature<T: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    interior_quadrature: QuadraturePair<T, D>,
    // TODO: Use NestedVec?
    interface_quadratures: Vec<QuadraturePair<T, D>>,
}

impl<T, D> EmbeddedQuadrature<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn from_interior_and_interface(
        interior_quadrature: QuadraturePair<T, D>,
        interface_quadratures: Vec<QuadraturePair<T, D>>,
    ) -> Self {
        Self {
            interior_quadrature,
            interface_quadratures,
        }
    }

    pub fn interior_quadrature(&self) -> &QuadraturePair<T, D> {
        &self.interior_quadrature
    }

    pub fn interface_quadratures(&self) -> &[QuadraturePair<T, D>] {
        &self.interface_quadratures
    }
}

impl<T, D, Connectivity> EmbeddedModel<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn vertices(&self) -> &[Point<T, D>] {
        self.background_mesh.vertices()
    }

    pub fn interior_connectivity(&self) -> &[Connectivity] {
        &self.background_mesh.connectivity()[0..self.num_interior]
    }

    pub fn interface_connectivity(&self) -> &[Connectivity] {
        &self.background_mesh.connectivity()[self.num_interior..]
    }

    pub fn background_mesh(&self) -> &Mesh<T, D, Connectivity> {
        &self.background_mesh
    }

    pub fn set_mass_regularization_factor(&mut self, factor: T) {
        self.mass_regularization_factor = factor;
    }

    pub fn mass_quadrature(&self) -> Option<&EmbeddedQuadrature<T, D>> {
        self.mass_quadrature.as_ref()
    }

    pub fn stiffness_quadrature(&self) -> Option<&EmbeddedQuadrature<T, D>> {
        self.stiffness_quadrature.as_ref()
    }

    pub fn elliptic_quadrature(&self) -> Option<&EmbeddedQuadrature<T, D>> {
        self.elliptic_quadrature.as_ref()
    }
}

impl<T, D, Connectivity> EmbeddedModel<T, D, Connectivity>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    Connectivity: CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    Connectivity::Cell: Distance<T, Point<T, D>>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, D, Connectivity::NodalDim>,
{
    pub fn make_interpolator(
        &self,
        interpolation_points: &[Point<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn std::error::Error>> {
        FiniteElementInterpolator::interpolate_space(&self.background_mesh, interpolation_points)
    }
}

impl<T, D, Connectivity> MakeInterpolator<T, D> for EmbeddedModel<T, D, Connectivity>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    Connectivity: CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    Connectivity::Cell: Distance<T, Point<T, D>>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, D, Connectivity::NodalDim>,
{
    fn make_interpolator(
        &self,
        interpolation_points: &[Point<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn std::error::Error>> {
        self.make_interpolator(interpolation_points)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedModelBuilder<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    vertices: Option<Vec<Point<T, D>>>,
    interior_connectivity: Option<Vec<Connectivity>>,
    interface_connectivity: Option<Vec<Connectivity>>,
    mass_quadrature: Option<EmbeddedQuadrature<T, D>>,
    stiffness_quadrature: Option<EmbeddedQuadrature<T, D>>,
    elliptic_quadrature: Option<EmbeddedQuadrature<T, D>>,
}

impl<T, D, C> EmbeddedModelBuilder<T, D, C>
where
    T: RealField,
    D: DimName,
    C: Clone,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn new() -> Self {
        Self {
            vertices: None,
            interior_connectivity: None,
            interface_connectivity: None,
            mass_quadrature: None,
            stiffness_quadrature: None,
            elliptic_quadrature: None,
        }
    }

    pub fn vertices(&mut self, vertices: Vec<Point<T, D>>) -> &mut Self {
        self.vertices = Some(vertices);
        self
    }

    pub fn interior_connectivity(&mut self, interior_connectivity: Vec<C>) -> &mut Self {
        self.interior_connectivity = Some(interior_connectivity);
        self
    }

    pub fn interface_connectivity(&mut self, interface_connectivity: Vec<C>) -> &mut Self {
        self.interface_connectivity = Some(interface_connectivity);
        self
    }

    pub fn mass_quadrature(&mut self, quadrature: EmbeddedQuadrature<T, D>) -> &mut Self {
        self.mass_quadrature = Some(quadrature);
        self
    }

    pub fn stiffness_quadrature(&mut self, quadrature: EmbeddedQuadrature<T, D>) -> &mut Self {
        self.stiffness_quadrature = Some(quadrature);
        self
    }

    pub fn elliptic_quadrature(&mut self, quadrature: EmbeddedQuadrature<T, D>) -> &mut Self {
        self.elliptic_quadrature = Some(quadrature);
        self
    }

    /// Sets all quadratures to the same quadrature.
    pub fn catchall_quadrature(&mut self, quadrature: EmbeddedQuadrature<T, D>) -> &mut Self {
        self.mass_quadrature = Some(quadrature.clone());
        self.stiffness_quadrature = Some(quadrature.clone());
        self.elliptic_quadrature = Some(quadrature);
        self
    }

    pub fn build(&self) -> EmbeddedModel<T, D, C>
    where
        C: Connectivity,
    {
        let interior_connectivity = self
            .interior_connectivity
            .clone()
            .expect("Missing interior connectivity");
        let num_interior = interior_connectivity.len();
        let mut connectivity = interior_connectivity;
        connectivity.extend(
            self.interface_connectivity
                .clone()
                .expect("Missing interface connectivity."),
        );
        let vertices = self.vertices.clone().expect("Missing vertices.");

        let background_mesh = Mesh::from_vertices_and_connectivity(vertices, connectivity);

        let interior_colors = {
            let interior_connectivity = &background_mesh.connectivity()[0..num_interior];
            color_nodes(interior_connectivity)
        };

        let interface_colors = {
            let interface_connectivity = &background_mesh.connectivity()[num_interior..];
            color_nodes(interface_connectivity)
        };

        EmbeddedModel {
            background_mesh,
            num_interior,
            interior_colors,
            interface_colors,
            mass_quadrature: self.mass_quadrature.clone(),
            stiffness_quadrature: self.stiffness_quadrature.clone(),
            elliptic_quadrature: self.elliptic_quadrature.clone(),
            mass_regularization_factor: T::zero(),
        }
    }
}

impl<T, C> EmbeddedModelBuilder<T, U3, C>
where
    T: Scalar,
    C: Clone + ConnectivityMut,
    DefaultAllocator: Allocator<T, U3>,
{
    pub fn from_embedding(background_mesh: &Mesh3d<T, C>, embedding: Embedding<T>) -> Self {
        // The embedding will mark some cells as exterior, which also means that some vertices
        // might have no associated cells. To account for this, we reconstruct the background
        // mesh with only the relevant connectivity, thereby removing unconnected vertices.
        let num_interior = embedding.interior_cells.len();
        let mut keep_cells = embedding.interior_cells;
        keep_cells.extend_from_slice(&embedding.interface_cells);
        let new_background_mesh = background_mesh.keep_cells(&keep_cells);
        let interior_connectivity = &new_background_mesh.connectivity()[0..num_interior];
        let interface_connectivity = &new_background_mesh.connectivity()[num_interior..];

        // TODO: Store new background directly in builder instead of copying connectivity around
        // (this is for legacy reasons)

        Self {
            vertices: Some(new_background_mesh.vertices().to_vec()),
            interior_connectivity: Some(interior_connectivity.to_vec()),
            interface_connectivity: Some(interface_connectivity.to_vec()),
            mass_quadrature: None,
            stiffness_quadrature: None,
            elliptic_quadrature: None,
        }
    }
}

impl<T, D, Connectivity> ElasticityModel<T, D> for EmbeddedModel<T, D, Connectivity>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    Connectivity: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    DefaultAllocator: FiniteElementMatrixAllocator<T, D, D, Connectivity::NodalDim>,
    D: DimNameMul<Connectivity::NodalDim>,
    Connectivity::NodalDim: DimNameMul<D>,
{
    fn ndof(&self) -> usize {
        D::dim() * self.vertices().len()
    }

    fn assemble_stiffness_into(
        &self,
        csr: &mut CsrMatrix<T>,
        u: &DVector<T>,
        material_model: &dyn ElasticMaterialModel<T, D>,
    ) {
        let error_msg = "Need stiffness quadrature for assembling stiffness matrix.";
        assemble_stiffness_into_csr(
            csr,
            self.vertices(),
            self.interior_connectivity(),
            material_model,
            u,
            &|_| {
                self.stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
        );

        assemble_stiffness_into_csr(
            csr,
            self.vertices(),
            self.interface_connectivity(),
            material_model,
            u,
            &|i| {
                &self
                    .stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
        );
    }

    fn assemble_stiffness(&self, u: &DVector<T>, material_model: &dyn ElasticMaterialModel<T, D>) -> CooMatrix<T> {
        let ndof = self.ndof();
        let mut coo = CooMatrix::new(ndof, ndof);

        let error_msg = "Need stiffness quadrature for assembling stiffness matrix.";

        assemble_stiffness_into(
            &mut coo,
            self.vertices(),
            self.interior_connectivity(),
            material_model,
            u,
            &|_| {
                self.stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
        );

        assemble_stiffness_into(
            &mut coo,
            self.vertices(),
            self.interface_connectivity(),
            material_model,
            u,
            &|i| {
                &self
                    .stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
        );

        coo
    }

    fn assemble_mass(&self, density: T) -> CooMatrix<T> {
        let ndof = self.ndof();
        let mut coo = CooMatrix::new(ndof, ndof);

        let error_msg = "Need mass quadrature to assemble mass matrix.";

        assemble_mass_into(
            &mut coo,
            self.vertices(),
            self.interior_connectivity(),
            density,
            &|_| {
                self.mass_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
        );

        assemble_mass_into(
            &mut coo,
            self.vertices(),
            self.interface_connectivity(),
            density,
            &|i| {
                &self
                    .mass_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
        );

        // TODO: Move this into the quadrature instead?
        if self.mass_regularization_factor > T::zero() {
            assemble_mass_into(
                &mut coo,
                self.vertices(),
                self.background_mesh.connectivity(),
                self.mass_regularization_factor * density,
                &|_| {
                    self.mass_quadrature
                        .as_ref()
                        .expect(&error_msg)
                        .interior_quadrature()
                },
            );
        }

        coo
    }

    fn assemble_elastic_pseudo_forces(
        &self,
        u: DVectorSlice<T>,
        material_model: &dyn ElasticMaterialModel<T, D>,
    ) -> DVector<T> {
        let mut f = DVector::zeros(u.len());

        let error_msg = "Need elliptic quadrature to assemble elastic pseudo forces.";

        assemble_pseudo_forces_into(
            DVectorSliceMut::from(&mut f),
            self.vertices(),
            self.interior_connectivity(),
            material_model,
            u,
            &|_| {
                self.elliptic_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
        );

        assemble_pseudo_forces_into(
            DVectorSliceMut::from(&mut f),
            self.vertices(),
            self.interface_connectivity(),
            material_model,
            u,
            &|i| {
                &self
                    .elliptic_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
        );

        f
    }

    fn compute_scalar_element_integrals(
        &self,
        _u: DVectorSlice<T>,
        _integrand: &dyn ScalarMaterialSpaceFunction<T, D, D>,
    ) -> DVector<T> {
        unimplemented!("Strain energy computation not implemented for EmbeddedModel");
    }
}

impl<T, D, C> ElasticityModelParallel<T, D> for EmbeddedModel<T, D, C>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    C: Sync + ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    D: DimNameMul<C::NodalDim>,
    C::NodalDim: DimNameMul<D>,
    DefaultAllocator: FiniteElementMatrixAllocator<T, D, D, C::NodalDim>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Sync,
{
    fn assemble_elastic_pseudo_forces_into_par(
        &self,
        mut f: DVectorSliceMut<T>,
        u: DVectorSlice<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
    ) {
        let error_msg = "Need elliptic quadrature to assemble pseudo forces.";

        assemble_pseudo_forces_into_par(
            DVectorSliceMut::from(&mut f),
            self.vertices(),
            self.interior_connectivity(),
            material_model,
            u,
            &|_| {
                self.elliptic_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
            &self.interior_colors,
        );

        assemble_pseudo_forces_into_par(
            DVectorSliceMut::from(&mut f),
            self.vertices(),
            self.interface_connectivity(),
            material_model,
            u,
            &|i| {
                &self
                    .elliptic_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
            &self.interface_colors,
        );
    }

    fn assemble_transformed_stiffness_par(
        &self,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
        transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    ) -> CooMatrix<T> {
        let error_msg = "Need stiffness quadrature for assembling stiffness matrix.";

        let coo_interior = assemble_transformed_stiffness_par(
            self.vertices(),
            self.interior_connectivity(),
            material_model,
            u,
            &|_| {
                self.stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
            transformation,
        );

        let coo_interface = assemble_transformed_stiffness_par(
            self.vertices(),
            self.interface_connectivity(),
            material_model,
            u,
            &|i| {
                &self
                    .stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
            transformation,
        );

        let mut coo = coo_interior;
        coo += &coo_interface;
        coo
    }

    fn assemble_transformed_stiffness_into_par(
        &self,
        csr: &mut CsrMatrix<T>,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
        transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    ) {
        let error_msg = "Need stiffness quadrature for assembling stiffness matrix.";

        assemble_transformed_stiffness_into_csr_par(
            csr,
            self.vertices(),
            self.interior_connectivity(),
            material_model,
            u,
            &|_| {
                self.stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interior_quadrature()
            },
            transformation,
            &self.interior_colors,
        );

        assemble_transformed_stiffness_into_csr_par(
            csr,
            self.vertices(),
            self.interface_connectivity(),
            material_model,
            u,
            &|i| {
                &self
                    .stiffness_quadrature
                    .as_ref()
                    .expect(&error_msg)
                    .interface_quadratures()[i]
            },
            transformation,
            &self.interface_colors,
        );
    }

    fn compute_scalar_element_integrals_par(
        &self,
        _u: DVectorSlice<T>,
        _integrand: &(dyn Sync + ScalarMaterialSpaceFunction<T, D, D>),
    ) -> DVector<T> {
        unimplemented!("Strain energy computation not implemented for EmbeddedModel");
    }
}

impl<'a, T, D, C> GeometryCollection<'a> for EmbeddedModel<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: CellConnectivity<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    type Geometry = C::Cell;

    fn num_geometries(&self) -> usize {
        self.background_mesh.num_geometries()
    }

    fn get_geometry(&'a self, index: usize) -> Option<Self::Geometry> {
        self.background_mesh.get_geometry(index)
    }
}

impl<'a, T, D, C, QueryGeometry> DistanceQuery<'a, QueryGeometry> for EmbeddedModel<T, D, C>
where
    T: RealField,
    D: DimName,
    C: CellConnectivity<T, D>,
    Mesh<T, D, C>: DistanceQuery<'a, QueryGeometry>,
    DefaultAllocator: Allocator<T, D>,
{
    fn nearest(&'a self, query_geometry: &QueryGeometry) -> Option<usize> {
        self.background_mesh.nearest(query_geometry)
    }
}
