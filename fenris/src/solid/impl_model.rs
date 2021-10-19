use crate::allocators::{ElementConnectivityAllocator, FiniteElementMatrixAllocator};
use crate::assembly::ElementMatrixTransformation;
use crate::element::ElementConnectivity;
use crate::model::NodalModel;
use crate::solid::assembly::{
    assemble_mass_into, assemble_pseudo_forces_into, assemble_pseudo_forces_into_par, assemble_stiffness_into,
    assemble_stiffness_into_csr, assemble_transformed_stiffness_into_csr_par, assemble_transformed_stiffness_par,
    compute_scalar_element_integrals_into, compute_scalar_element_integrals_into_par, ScalarMaterialSpaceFunction,
};
use crate::solid::{ElasticMaterialModel, ElasticityModel, ElasticityModelParallel};
use crate::{CooMatrix, CsrMatrix};
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimMin, DimNameMul, RealField};

impl<T, D, C> ElasticityModel<T, D> for NodalModel<T, D, C>
where
    T: RealField,
    C: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    C::NodalDim: DimNameMul<D>,
    D: DimNameMul<C::NodalDim> + DimMin<D, Output = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C> + FiniteElementMatrixAllocator<T, D, D, C::NodalDim>,
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
        let error_msg = "Need stiffness quadrature to assemble stiffness matrix";
        assemble_stiffness_into_csr(csr, self.vertices(), self.connectivity(), material_model, u, &|_| {
            self.stiffness_quadrature().expect(&error_msg)
        })
    }

    fn assemble_stiffness(&self, u: &DVector<T>, material_model: &dyn ElasticMaterialModel<T, D>) -> CooMatrix<T> {
        let ndof = self.ndof();
        let mut coo = CooMatrix::new(ndof, ndof);
        let error_msg = "Need stiffness quadrature to assemble stiffness matrix";
        assemble_stiffness_into(
            &mut coo,
            self.vertices(),
            self.connectivity(),
            material_model,
            u,
            &|_| self.stiffness_quadrature().expect(&error_msg),
        );
        coo
    }

    fn assemble_mass(&self, density: T) -> CooMatrix<T> {
        let ndof = self.ndof();
        let mut coo = CooMatrix::new(ndof, ndof);
        let error_msg = "Need mass quadrature to assemble mass matrix";
        assemble_mass_into(&mut coo, self.vertices(), self.connectivity(), density, &|_| {
            self.mass_quadrature().expect(&error_msg)
        });
        coo
    }

    fn assemble_elastic_pseudo_forces(
        &self,
        u: DVectorSlice<T>,
        material_model: &dyn ElasticMaterialModel<T, D>,
    ) -> DVector<T> {
        let mut f = DVector::zeros(u.len());
        let error_msg = "Need elliptic quadrature to assemble pseudo forces";
        assemble_pseudo_forces_into(
            DVectorSliceMut::from(&mut f),
            self.vertices(),
            self.connectivity(),
            material_model,
            u,
            &|_| self.elliptic_quadrature().expect(&error_msg),
        );
        f
    }

    fn compute_scalar_element_integrals(
        &self,
        u: DVectorSlice<T>,
        integrand: &dyn ScalarMaterialSpaceFunction<T, D, D>,
    ) -> DVector<T> {
        let mut e = DVector::zeros(self.connectivity().len());
        let error_msg = "Need elliptic quadrature to compute scalar integral";
        compute_scalar_element_integrals_into(
            DVectorSliceMut::from(&mut e),
            self.vertices(),
            self.connectivity(),
            integrand,
            u,
            // TODO: Is this a reasonable choice for computing the element integrals?
            &|_| self.elliptic_quadrature().expect(&error_msg),
        );
        e
    }
}

impl<T, D, C> ElasticityModelParallel<T, D> for NodalModel<T, D, C>
where
    T: RealField,
    C: Sync + ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    C::NodalDim: DimNameMul<D>,
    D: DimNameMul<C::NodalDim> + DimMin<D, Output = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C> + FiniteElementMatrixAllocator<T, D, D, C::NodalDim>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Sync,
{
    fn assemble_elastic_pseudo_forces_into_par(
        &self,
        f: DVectorSliceMut<T>,
        u: DVectorSlice<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
    ) {
        let error_msg = "Need elliptic quadrature to assemble pseudo forces";
        assemble_pseudo_forces_into_par(
            f,
            self.vertices(),
            self.connectivity(),
            material_model,
            u,
            &|_| self.elliptic_quadrature().expect(&error_msg),
            self.colors(),
        );
    }

    fn assemble_transformed_stiffness_par(
        &self,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
        transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    ) -> CooMatrix<T> {
        let error_msg = "Need stiffness quadrature to assemble stiffness matrix";
        assemble_transformed_stiffness_par(
            self.vertices(),
            self.connectivity(),
            material_model,
            u,
            &|_| self.stiffness_quadrature().expect(&error_msg),
            transformation,
        )
    }

    fn assemble_transformed_stiffness_into_par(
        &self,
        csr: &mut CsrMatrix<T>,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
        transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    ) {
        let error_msg = "Need stiffness quadrature to assemble stiffness matrix";
        assemble_transformed_stiffness_into_csr_par(
            csr,
            self.vertices(),
            self.connectivity(),
            material_model,
            u,
            &|_| self.stiffness_quadrature().expect(&error_msg),
            transformation,
            self.colors(),
        )
    }

    fn compute_scalar_element_integrals_par(
        &self,
        u: DVectorSlice<T>,
        integrand: &(dyn Sync + ScalarMaterialSpaceFunction<T, D, D>),
    ) -> DVector<T> {
        let mut e = DVector::zeros(self.connectivity().len());
        let error_msg = "Need elliptic quadrature to compute scalar integral";
        compute_scalar_element_integrals_into_par(
            DVectorSliceMut::from(&mut e),
            self.vertices(),
            self.connectivity(),
            integrand,
            u,
            // TODO: Is this a reasonable choice for quadrature rule?
            &|_| self.elliptic_quadrature().expect(&error_msg),
        );
        e
    }
}
