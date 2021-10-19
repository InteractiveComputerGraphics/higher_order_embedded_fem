//! Solid mechanics functionality.

pub mod assembly;
mod impl_model;
pub mod materials;

use nalgebra::{
    DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName, Dynamic, MatrixN,
    MatrixSliceMN, RealField, Scalar, VectorN, U1,
};

use crate::{CooMatrix, CsrMatrix};

use crate::assembly::{ElementMatrixTransformation, NoTransformation};
use assembly::ScalarMaterialSpaceFunction;
use delegate::delegate;
use nalgebra::allocator::Allocator;
use std::fmt::Debug;
use std::ops::{AddAssign, Deref};

pub trait ElasticMaterialModel<T, D>: Debug
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Indicates whether the energy Hessian is positive semi-definite.
    ///
    /// This is used by solvers to determine whether system matrices may become indefinite,
    /// which may require special care.
    fn is_positive_semi_definite(&self) -> bool {
        false
    }

    fn compute_strain_energy_density(&self, deformation_gradient: &MatrixN<T, D>) -> T;
    fn compute_stress_tensor(&self, deformation_gradient: &MatrixN<T, D>) -> MatrixN<T, D>;
    fn contract_stress_tensor_with(
        &self,
        deformation_gradient: &MatrixN<T, D>,
        a: &VectorN<T, D>,
        b: &VectorN<T, D>,
    ) -> MatrixN<T, D>;

    /// Compute stress tensor contractions for several vector pairs simultaneously.
    ///
    /// The matrix `a` is a `D x N` matrix, where each column is associated with a node.
    /// Let c(F, a, b) denote a contraction of two vectors a and b, and let a_I denote
    /// the Ith column in `a`. Let `output_IJ` denote the `D x D` block in position IJ
    /// in `output`. At the end of this method call, the following property must hold:
    ///  output_IJ = output_IJ + c(F, a_I, a_J)
    /// for all I, J = 1, ..., N.
    ///
    /// The default implementation will individually call the contraction for each
    /// block `IJ`. By overriding the default implementation, it is possible to reuse
    /// data across these computations and achieve higher performance.
    fn contract_multiple_stress_tensors_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        deformation_gradient: &MatrixN<T, D>,
        a: &MatrixSliceMN<T, D, Dynamic>,
    ) {
        let num_nodes = a.ncols();
        let output_dim = num_nodes * D::dim();
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let d = D::dim();
        for i in 0..num_nodes {
            for j in i..num_nodes {
                let a_i = a.fixed_slice::<D, U1>(0, i).clone_owned();
                let a_j = a.fixed_slice::<D, U1>(0, j).clone_owned();
                let contraction = self.contract_stress_tensor_with(deformation_gradient, &a_i, &a_j);
                output
                    .fixed_slice_mut::<D, D>(i * d, j * d)
                    .add_assign(&contraction);

                // TODO: We currently assume symmetry. Should maybe have a method that
                // says whether it is symmetric or not?
                if i != j {
                    output
                        .fixed_slice_mut::<D, D>(j * d, i * d)
                        .add_assign(&contraction.transpose());
                }
            }
        }
    }
}

pub trait ElasticityModel<T: Scalar, D: DimName>: Debug
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn ndof(&self) -> usize;

    fn assemble_stiffness_into(
        &self,
        csr: &mut CsrMatrix<T>,
        u: &DVector<T>,
        material_model: &dyn ElasticMaterialModel<T, D>,
    );

    fn assemble_stiffness(&self, u: &DVector<T>, material_model: &dyn ElasticMaterialModel<T, D>) -> CooMatrix<T>;

    // TODO: Come up with a general abstraction for variable density (e.g. per-quadrature point)
    fn assemble_mass(&self, density: T) -> CooMatrix<T>;

    fn assemble_elastic_pseudo_forces(
        &self,
        u: DVectorSlice<T>,
        material_model: &dyn ElasticMaterialModel<T, D>,
    ) -> DVector<T>;

    fn compute_scalar_element_integrals(
        &self,
        u: DVectorSlice<T>,
        integrand: &dyn ScalarMaterialSpaceFunction<T, D, D>,
    ) -> DVector<T>;
}

/// An extension of elasticity model that allows assembling in parallel.
///
/// In general, the purpose of this trait is not to *guarantee* parallel execution, but to
/// allow parallel assembly given the right circumstances.
pub trait ElasticityModelParallel<T: Scalar, D: DimName>: ElasticityModel<T, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn assemble_elastic_pseudo_forces_into_par(
        &self,
        f: DVectorSliceMut<T>,
        u: DVectorSlice<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
    );

    fn assemble_stiffness_par(
        &self,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
    ) -> CooMatrix<T> {
        self.assemble_transformed_stiffness_par(u, material_model, &NoTransformation)
    }

    fn assemble_stiffness_into_par(
        &self,
        csr: &mut CsrMatrix<T>,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
    ) {
        self.assemble_transformed_stiffness_into_par(csr, u, material_model, &NoTransformation)
    }

    fn assemble_transformed_stiffness_par(
        &self,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
        transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    ) -> CooMatrix<T>;

    fn assemble_transformed_stiffness_into_par(
        &self,
        csr: &mut CsrMatrix<T>,
        u: &DVector<T>,
        material_model: &(dyn Sync + ElasticMaterialModel<T, D>),
        transformation: &(dyn Sync + ElementMatrixTransformation<T>),
    );

    fn compute_scalar_element_integrals_par(
        &self,
        u: DVectorSlice<T>,
        integrand: &(dyn Sync + ScalarMaterialSpaceFunction<T, D, D>),
    ) -> DVector<T>;
}

impl<T, D> ElasticMaterialModel<T, D> for Box<dyn ElasticMaterialModel<T, D>>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    delegate! {
        to self.deref() {
            fn compute_strain_energy_density(&self, deformation_gradient: &MatrixN<T, D>) -> T;
            fn compute_stress_tensor(&self, deformation_gradient: &MatrixN<T, D>) -> MatrixN<T, D>;
            fn contract_stress_tensor_with(
                &self,
                deformation_gradient: &MatrixN<T, D>,
                a: &VectorN<T, D>,
                b: &VectorN<T, D>
            ) -> MatrixN<T, D>;
        }
    }
}

impl<T, D, X> ElasticMaterialModel<T, D> for &X
where
    T: RealField,
    D: DimName,
    X: ElasticMaterialModel<T, D>,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    delegate! {
        to self.deref() {
            fn compute_strain_energy_density(&self, deformation_gradient: &MatrixN<T, D>) -> T;
            fn compute_stress_tensor(&self, deformation_gradient: &MatrixN<T, D>) -> MatrixN<T, D>;
            fn contract_stress_tensor_with(
                &self,
                deformation_gradient: &MatrixN<T, D>,
                a: &VectorN<T, D>,
                b: &VectorN<T, D>
            ) -> MatrixN<T, D>;
        }
    }
}
