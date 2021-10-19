use crate::assembly::{
    assemble_generalized_elliptic_term_into, assemble_generalized_elliptic_term_into_par,
    assemble_generalized_mass_into, assemble_generalized_stiffness_into, assemble_generalized_stiffness_into_csr,
    assemble_generalized_stiffness_into_csr_par, assemble_transformed_generalized_stiffness_into_csr_par,
    assemble_transformed_generalized_stiffness_par, compute_element_integral, ElementMatrixTransformation,
    GeneralizedEllipticContraction, GeneralizedEllipticOperator, NoTransformation, QuadratureTable,
};
use crate::element::{ElementConnectivity, VolumetricFiniteElement};
use crate::solid::ElasticMaterialModel;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimMin, DimName, DimNameMul, Dynamic,
    MatrixMN, MatrixN, MatrixSliceMN, Point, RealField, VectorN, U1,
};
use rayon::prelude::*;

use crate::allocators::{ElementConnectivityAllocator, FiniteElementMatrixAllocator, VolumeFiniteElementAllocator};
use crate::{CooMatrix, CsrMatrix};
use paradis::DisjointSubsets;

/// A wrapper for a material model that allows it to be interpreted as a
/// generalized elliptic operator for use in assembly.
pub struct MaterialEllipticOperator<'a, Material: ?Sized>(pub &'a Material);

impl<'a, T, D, M> GeneralizedEllipticOperator<T, D, D> for MaterialEllipticOperator<'a, M>
where
    T: RealField,
    D: DimName,
    M: ?Sized + ElasticMaterialModel<T, D>,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    #[allow(non_snake_case)]
    fn compute_elliptic_term(&self, gradient: &MatrixN<T, D>) -> MatrixN<T, D> {
        // TODO: Avoid double transpose somehow?
        let F = gradient.transpose() + MatrixN::<_, D>::identity();
        -self.0.compute_stress_tensor(&F).transpose()
    }
}

impl<'a, T, D, M> GeneralizedEllipticContraction<T, D, D> for MaterialEllipticOperator<'a, M>
where
    T: RealField,
    D: DimName,
    M: ?Sized + ElasticMaterialModel<T, D>,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D> + Allocator<T, U1, D>,
{
    #[allow(non_snake_case)]
    fn contract(&self, gradient: &MatrixN<T, D>, a: &VectorN<T, D>, b: &VectorN<T, D>) -> MatrixN<T, D> {
        let F = gradient.transpose() + MatrixN::<_, D>::identity();
        self.0.contract_stress_tensor_with(&F, a, b)
    }

    #[allow(non_snake_case)]
    fn contract_multiple_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        gradient: &MatrixN<T, D>,
        a: &MatrixSliceMN<T, D, Dynamic>,
    ) {
        let F = gradient.transpose() + MatrixN::<_, D>::identity();
        self.0.contract_multiple_stress_tensors_into(output, &F, a)
    }
}

pub fn assemble_stiffness_into<T, C>(
    coo: &mut CooMatrix<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, C::GeometryDim>,
) where
    T: RealField,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::Element: VolumetricFiniteElement<T>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>
        + FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
{
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_generalized_stiffness_into(coo, vertices, connectivity, &elliptic_operator, u, quadrature_table)
}

pub fn assemble_stiffness_into_csr<T, C>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, C::GeometryDim>,
) where
    T: RealField,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::Element: VolumetricFiniteElement<T>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>
        + FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
{
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_generalized_stiffness_into_csr(csr, vertices, connectivity, &elliptic_operator, u, quadrature_table)
}

pub fn assemble_stiffness_into_csr_par<T, C>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl Sync + ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, C::GeometryDim>),
    colors: &[DisjointSubsets],
) where
    T: RealField,
    C: Sync + ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::Element: VolumetricFiniteElement<T>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>
        + FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
    <DefaultAllocator as Allocator<T, C::GeometryDim>>::Buffer: Sync,
{
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_generalized_stiffness_into_csr_par(
        csr,
        vertices,
        connectivity,
        &elliptic_operator,
        u,
        quadrature_table,
        colors,
    )
}

pub fn assemble_transformed_stiffness_into_csr_par<T, C>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl Sync + ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, C::GeometryDim>),
    transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    colors: &[DisjointSubsets],
) where
    T: RealField,
    C: Sync + ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::Element: VolumetricFiniteElement<T>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>
        + FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
    <DefaultAllocator as Allocator<T, C::GeometryDim>>::Buffer: Sync,
{
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_transformed_generalized_stiffness_into_csr_par(
        csr,
        vertices,
        connectivity,
        &elliptic_operator,
        u,
        quadrature_table,
        transformation,
        colors,
    )
}

pub fn assemble_transformed_stiffness_par<T, C>(
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl Sync + ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, C::GeometryDim>),
    transformation: &(dyn Sync + ElementMatrixTransformation<T>),
) -> CooMatrix<T>
where
    T: RealField,
    C: Sync + ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::Element: VolumetricFiniteElement<T>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>
        + FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
    <DefaultAllocator as Allocator<T, C::GeometryDim>>::Buffer: Sync,
{
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_transformed_generalized_stiffness_par(
        vertices,
        connectivity,
        &elliptic_operator,
        u,
        quadrature_table,
        transformation,
    )
}

pub fn assemble_stiffness_par<T, C>(
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl Sync + ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, C::GeometryDim>),
) -> CooMatrix<T>
where
    T: RealField,
    C: Sync + ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::Element: VolumetricFiniteElement<T>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>
        + FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
    <DefaultAllocator as Allocator<T, C::GeometryDim>>::Buffer: Sync,
{
    assemble_transformed_stiffness_par(
        vertices,
        connectivity,
        material_model,
        u,
        quadrature_table,
        &NoTransformation,
    )
}

pub fn assemble_mass_into<T, C>(
    coo: &mut CooMatrix<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    // TODO: Generalize density somehow? Attach properties to quadrature points?
    density: T,
    quadrature_table: &impl QuadratureTable<T, C::GeometryDim>,
) where
    T: RealField,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    C::NodalDim: DimNameMul<C::GeometryDim>,
    DefaultAllocator: FiniteElementMatrixAllocator<T, C::GeometryDim, C::GeometryDim, C::NodalDim>,
{
    assemble_generalized_mass_into(coo, vertices, connectivity, density, quadrature_table)
}

pub fn assemble_pseudo_forces_into<'a, T, C>(
    f: DVectorSliceMut<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl ?Sized + ElasticMaterialModel<T, C::GeometryDim>),
    u: impl Into<DVectorSlice<'a, T>>,
    quadrature_table: &impl QuadratureTable<T, C::GeometryDim>,
) where
    T: RealField,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, C::GeometryDim, C::NodalDim>,
{
    let u = u.into();
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_generalized_elliptic_term_into(f, vertices, connectivity, &elliptic_operator, &u, quadrature_table)
}

pub fn assemble_pseudo_forces_into_par<'a, T, C>(
    f: DVectorSliceMut<T>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    material_model: &(impl ?Sized + Sync + ElasticMaterialModel<T, C::GeometryDim>),
    u: impl Into<DVectorSlice<'a, T>>,
    quadrature_table: &(impl Sync + QuadratureTable<T, C::GeometryDim>),
    colors: &[DisjointSubsets],
) where
    T: RealField,
    C: Sync + ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, C::GeometryDim, C::NodalDim>,
    <DefaultAllocator as Allocator<T, C::GeometryDim>>::Buffer: Sync,
{
    let u = u.into();
    let elliptic_operator = MaterialEllipticOperator(material_model);
    assemble_generalized_elliptic_term_into_par(
        f,
        vertices,
        connectivity,
        &elliptic_operator,
        &u,
        quadrature_table,
        colors,
    )
}

/// A scalar function that is evaluated in material space
pub trait ScalarMaterialSpaceFunction<T, GeometryDim, SolutionDim>
where
    T: RealField,
    GeometryDim: DimName,
    SolutionDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, SolutionDim> + Allocator<T, GeometryDim> + Allocator<T, SolutionDim>,
{
    fn evaluate(
        &self,
        material_coords: &VectorN<T, GeometryDim>,
        u: &VectorN<T, SolutionDim>,
        u_grad: &MatrixMN<T, GeometryDim, SolutionDim>,
    ) -> T;
}

impl<F, T, GeometryDim, SolutionDim> ScalarMaterialSpaceFunction<T, GeometryDim, SolutionDim> for F
where
    T: RealField,
    GeometryDim: DimName,
    SolutionDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, SolutionDim> + Allocator<T, GeometryDim> + Allocator<T, SolutionDim>,
    F: Fn(&VectorN<T, GeometryDim>, &VectorN<T, SolutionDim>, &MatrixMN<T, GeometryDim, SolutionDim>) -> T,
{
    fn evaluate(
        &self,
        material_coords: &VectorN<T, GeometryDim>,
        u: &VectorN<T, SolutionDim>,
        u_grad: &MatrixMN<T, GeometryDim, SolutionDim>,
    ) -> T {
        self(material_coords, u, u_grad)
    }
}

#[allow(non_snake_case)]
pub fn compute_scalar_element_integrals_into<'a, 'b, T, C>(
    e: impl Into<DVectorSliceMut<'a, T>>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    integrand: &(impl ?Sized + ScalarMaterialSpaceFunction<T, C::GeometryDim, C::GeometryDim>),
    u: impl Into<DVectorSlice<'b, T>>,
    quadrature_table: &impl QuadratureTable<T, C::GeometryDim>,
) where
    T: RealField,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, C::GeometryDim, C::NodalDim>,
{
    let mut e = e.into();
    let u = u.into();

    let fun = |material_coords: &VectorN<T, C::GeometryDim>,
               u: &VectorN<T, C::GeometryDim>,
               u_grad: &MatrixMN<T, C::GeometryDim, C::GeometryDim>|
     -> T { integrand.evaluate(material_coords, u, u_grad) };

    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );
        let u_element = connectivity.element_variables(u);

        let element_energy =
            compute_element_integral(&element, &u_element, &quadrature_table.quadrature_for_element(i), fun);

        e[i] = element_energy;
    }
}

#[allow(non_snake_case)]
pub fn compute_scalar_element_integrals_into_par<'a, 'b, T, C>(
    e: impl Into<DVectorSliceMut<'a, T>>,
    vertices: &[Point<T, C::GeometryDim>],
    connectivity: &[C],
    integrand: &(impl ?Sized + Sync + ScalarMaterialSpaceFunction<T, C::GeometryDim, C::GeometryDim>),
    u: impl Into<DVectorSlice<'b, T>>,
    quadrature_table: &(impl Sync + QuadratureTable<T, C::GeometryDim>),
) where
    T: RealField,
    C: Sync + ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::GeometryDim: DimNameMul<C::NodalDim> + DimMin<C::GeometryDim, Output = C::GeometryDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, C::GeometryDim, C::NodalDim>,
    <DefaultAllocator as Allocator<T, C::GeometryDim>>::Buffer: Sync,
{
    let mut e = e.into();
    let u = u.into();

    let fun = |material_coords: &VectorN<T, C::GeometryDim>,
               u: &VectorN<T, C::GeometryDim>,
               u_grad: &MatrixMN<T, C::GeometryDim, C::GeometryDim>|
     -> T { integrand.evaluate(material_coords, u, u_grad) };

    connectivity
        .par_iter()
        .enumerate()
        .zip(e.as_mut_slice().par_iter_mut())
        .for_each(|((i, connectivity), e)| {
            let element = connectivity.element(vertices).expect(
                "All vertices of element are assumed to be in bounds.\
                     TODO: Ensure this upon construction of basis?",
            );
            let u_element = connectivity.element_variables(u);

            let element_energy =
                compute_element_integral(&element, &u_element, &quadrature_table.quadrature_for_element(i), fun);

            *e = element_energy;
        });
}
