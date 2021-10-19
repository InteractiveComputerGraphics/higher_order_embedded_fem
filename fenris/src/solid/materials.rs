use crate::solid::ElasticMaterialModel;
use nalgebra::{
    DMatrixSliceMut, DefaultAllocator, DimMin, DimMul, DimName, DimProd, DimSub, Dynamic, Matrix2, Matrix3, Matrix4,
    MatrixN, MatrixSliceMN, RealField, Scalar, SymmetricEigen, Vector2, Vector3, VectorN, U1, U2, U3, U4, U9,
};

use crate::util::{cross_product_matrix, diag_left_mul, rotation_svd, try_transmute_ref, try_transmute_ref_mut};
use nalgebra::allocator::Allocator;
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::ops::AddAssign;
use std::ops::SubAssign;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LameParameters<T> {
    pub mu: T,
    pub lambda: T,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct YoungPoisson<T> {
    pub young: T,
    pub poisson: T,
}

impl<T> From<YoungPoisson<T>> for LameParameters<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
    fn from(params: YoungPoisson<T>) -> Self {
        let YoungPoisson { young, poisson } = params;
        let mu = 0.5 * young / (1.0 + poisson);
        let lambda = 0.5 * mu * poisson / (1.0 - 2.0 * poisson);
        Self { mu, lambda }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearElasticMaterial<T> {
    pub lame: LameParameters<T>,
}

impl<T, X> From<X> for LinearElasticMaterial<T>
where
    X: Into<LameParameters<T>>,
{
    fn from(params: X) -> Self {
        Self { lame: params.into() }
    }
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> ElasticMaterialModel<T, D> for LinearElasticMaterial<T>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D> + Allocator<T, U1, D>,
{
    fn compute_strain_energy_density(&self, deformation_gradient: &MatrixN<T, D>) -> T {
        let F = deformation_gradient;
        let eps = -MatrixN::<T, D>::identity() + (F + F.transpose()) / 2.0;

        let eps_trace = eps.trace();
        let eps_frobenius_sq = eps.fold(0.0, |acc, x| acc + x * x);

        self.lame.mu * eps_frobenius_sq + 0.5 * self.lame.lambda * eps_trace * eps_trace
    }

    fn compute_stress_tensor(&self, deformation_gradient: &MatrixN<T, D>) -> MatrixN<T, D> {
        let F = deformation_gradient;
        let eps = -MatrixN::<T, D>::identity() + (F + F.transpose()) / 2.0;
        &eps * 2.0 * self.lame.mu + MatrixN::<T, D>::identity() * self.lame.lambda * eps.trace()
    }

    fn contract_stress_tensor_with(
        &self,
        _deformation_gradient: &MatrixN<T, D>,
        a: &VectorN<T, D>,
        b: &VectorN<T, D>,
    ) -> MatrixN<T, D> {
        let B = a * b.transpose();
        let I = &MatrixN::<T, D>::identity();
        let mu = self.lame.mu;
        let lambda = self.lame.lambda;
        (I * B.trace() + B.transpose()) * mu + B * lambda
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorotatedLinearElasticMaterial<T> {
    pub lame: LameParameters<T>,
}

impl<T, X> From<X> for CorotatedLinearElasticMaterial<T>
where
    X: Into<LameParameters<T>>,
{
    fn from(params: X) -> Self {
        Self { lame: params.into() }
    }
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> ElasticMaterialModel<T, D> for CorotatedLinearElasticMaterial<T>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D> + DimSub<U1>,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, D, D>
        + Allocator<T, D>
        + Allocator<T, U1, D>
        + Allocator<T, <D as DimSub<U1>>::Output>
        + Allocator<(usize, usize), D>,
{
    fn compute_strain_energy_density(&self, deformation_gradient: &MatrixN<T, D>) -> T {
        let F = deformation_gradient;
        let eps = -MatrixN::<T, D>::identity() + (F + F.transpose()) / 2.0;

        let eps_trace = eps.trace();
        let eps_frobenius_sq = eps.fold(0.0, |acc, x| acc + x * x);

        self.lame.mu * eps_frobenius_sq + 0.5 * self.lame.lambda * eps_trace * eps_trace
    }

    fn compute_stress_tensor(&self, deformation_gradient: &MatrixN<T, D>) -> MatrixN<T, D> {
        let mu = self.lame.mu;
        let lambda = self.lame.lambda;

        let F = deformation_gradient;
        let I = &MatrixN::<T, D>::identity();

        let (U, _, V_T) = rotation_svd(F);
        let R = &(U * V_T);
        let eps = (R.transpose() * F) - MatrixN::<T, D>::identity();

        R * (&eps * 2.0 * mu + I * lambda * eps.trace())
    }

    fn contract_stress_tensor_with(
        &self,
        deformation_gradient: &MatrixN<T, D>,
        a: &VectorN<T, D>,
        b: &VectorN<T, D>,
    ) -> MatrixN<T, D> {
        let mu = self.lame.mu;
        let lambda = self.lame.lambda;

        let B = a * b.transpose();
        let F = deformation_gradient;
        let I = &MatrixN::<T, D>::identity();

        let (U, _, V_T) = rotation_svd(F);
        let R = &(U * V_T);

        R * ((I * B.trace() + B.transpose()) * mu + B * lambda) * R.transpose()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StVKMaterial<T> {
    pub lame: LameParameters<T>,
}

impl<T, X> From<X> for StVKMaterial<T>
where
    X: Into<LameParameters<T>>,
{
    fn from(params: X) -> Self {
        Self { lame: params.into() }
    }
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> ElasticMaterialModel<T, D> for StVKMaterial<T>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D> + Allocator<(usize, usize), D> + Allocator<T, U1, D>,
{
    fn compute_strain_energy_density(&self, deformation_gradient: &MatrixN<T, D>) -> T {
        let I = &MatrixN::<T, D>::identity();
        let F = deformation_gradient;
        let E = (F.transpose() * F - I) * 0.5;

        let E_trace = E.trace();
        let E_frobenius_sq = E.fold(0.0, |acc, x| acc + x * x);

        self.lame.mu * E_frobenius_sq + 0.5 * self.lame.lambda * E_trace * E_trace
    }

    fn compute_stress_tensor(&self, deformation_gradient: &MatrixN<T, D>) -> MatrixN<T, D> {
        let I = &MatrixN::<T, D>::identity();
        let F = deformation_gradient;
        let E = (F.transpose() * F - I) * 0.5;
        F * (&E * 2.0 * self.lame.mu + I * self.lame.lambda * E.trace())
    }

    fn contract_stress_tensor_with(
        &self,
        deformation_gradient: &MatrixN<T, D>,
        a: &VectorN<T, D>,
        b: &VectorN<T, D>,
    ) -> MatrixN<T, D> {
        let B = a * b.transpose();
        let I = &MatrixN::<T, D>::identity();
        let mu = self.lame.mu;
        let lambda = self.lame.lambda;
        let F = deformation_gradient;

        let E = (F.transpose() * F - I) * 0.5;
        I * (E.dot(&B) * 2.0 * mu + lambda * E.trace() * B.trace())
            + F * (B.transpose() * mu + &B * lambda + I * mu * B.trace()) * F.transpose()
    }
}

/// A Neo-Hookean type material model that is stable and robust to inversions.
///
/// Implements the material model proposed by Smith et al. [2018] in the paper
/// "Stable Neo-Hookean Flesh Simulation".
///
/// This model does *not* include the projection onto semi-definiteness,
/// and as such will produce contractions which are indefinite.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableNeoHookeanMaterial<T> {
    pub lame: LameParameters<T>,
}

impl<T, X> From<X> for StableNeoHookeanMaterial<T>
where
    X: Into<LameParameters<T>>,
{
    fn from(params: X) -> Self {
        Self { lame: params.into() }
    }
}

#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
fn reparametrize_lame_for_stable_neo_hookean<T>(d: T, lame: &LameParameters<T>) -> LameParameters<T>
where
    T: RealField,
{
    // Use the reparametrization described in section 3.4 of the paper,
    // so that the material parameters are consistent with linear solid.
    // Here we have generalized the results found in the paper to arbitrary dimension d.
    let mu = (d + 1.0) / d * lame.mu;
    let lambda = lame.lambda + (1.0 - 2.0 / (d * (d + 1.0))) * lame.mu;
    LameParameters { mu, lambda }
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> ElasticMaterialModel<T, D> for StableNeoHookeanMaterial<T>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D> + DimMul<D>,
    DimProd<D, D>: DimName,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, D, D>
        + Allocator<(usize, usize), D>
        + Allocator<T, U1, D>
        + Allocator<T, DimProd<D, D>, DimProd<D, D>>
        + Allocator<T, DimProd<D, D>>,
{
    fn compute_strain_energy_density(&self, F: &MatrixN<T, D>) -> T {
        let d = T::from_usize(D::dim()).unwrap();
        let LameParameters { mu, lambda } = reparametrize_lame_for_stable_neo_hookean(d, &self.lame);
        // Note: This expression is generalized from the paper, in which it was implicitly
        // assumed that the geometrical dimension is 3.
        let a = mu / lambda * (d / (d + 1.0));
        let alpha = 1.0 + a;

        let C = F.transpose() * F;
        let I_C = C.trace();
        let J = F.determinant();

        let J_minus_alpha = J - alpha;

        /*
        let identity_strain_energy_density=
            0.5 * self.lame.lambda * a * a
            - 0.5 * self.lame.mu * (d + 1.0).ln();


        0.5 * self.lame.mu * (I_C - d)
            + 0.5 * self.lame.lambda * (J_minus_alpha * J_minus_alpha)
            - 0.5 * self.lame.mu * (I_C + 1.0).ln()
            - identity_strain_energy_density
        */

        0.5 * self.lame.mu * (I_C - d) + 0.5 * self.lame.lambda * ((J_minus_alpha * J_minus_alpha) - (a * a))
            - 0.5 * self.lame.mu * ((I_C + 1.0) / (d + 1.0)).ln()
    }

    fn compute_stress_tensor(&self, F: &MatrixN<T, D>) -> MatrixN<T, D> {
        let d = T::from_usize(D::dim()).unwrap();
        let LameParameters { mu, lambda } = reparametrize_lame_for_stable_neo_hookean(d, &self.lame);
        // Note: This expression is generalized from the paper, in which it was implicitly
        // assumed that the geometrical dimension is 3.
        let alpha = 1.0 + mu / lambda * (d / (d + 1.0));

        let C = F.transpose() * F;
        let I_C = C.trace();
        let J = F.determinant();

        // dJ_dF = J * F^{-T}
        // Note: more specialized expressions can be derived for e.g. 2 and 3 dimensions,
        // which would allow us to avoid computing the inverse of F
        // Note: In general, dJ/dF = transpose(cofactor matrix of F)
        let dJ_dF = F
            .clone()
            .try_inverse()
            .expect("TODO: Handle singular F?")
            .transpose()
            * J;

        F * mu * (1.0 - 1.0 / (I_C + 1.0)) + dJ_dF * lambda * (J - alpha)
    }

    fn contract_stress_tensor_with(&self, F: &MatrixN<T, D>, a: &VectorN<T, D>, b: &VectorN<T, D>) -> MatrixN<T, D> {
        let B = a * b.transpose();
        let d = T::from_usize(F.nrows()).unwrap();
        let LameParameters { mu, lambda } = reparametrize_lame_for_stable_neo_hookean(d, &self.lame);
        let alpha = 1.0 + mu / lambda * (d / (d + 1.0));

        let I = &MatrixN::<T, D>::identity();
        let C = F.transpose() * F;
        let I_C = C.trace();
        let J = F.determinant();
        let beta = I_C + 1.0;
        let beta2 = beta * beta;
        let G = 1.0 - 1.0 / beta;
        let H = (J - alpha) * J;

        let F_inv = F.clone().try_inverse().expect("TODO: Handle singular F?");
        let F_inv_t = F_inv.transpose();

        let Q = F * &B * F.transpose() * 2.0 / beta2 + I * G * B.trace();
        let R = &F_inv_t * &B * &F_inv * (2.0 * J - alpha) * J - &F_inv_t * B.transpose() * &F_inv * H;

        Q * mu + R * lambda
    }

    #[inline(never)]
    fn contract_multiple_stress_tensors_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        F: &MatrixN<T, D>,
        a: &MatrixSliceMN<T, D, Dynamic>,
    ) {
        let d = D::dim();
        let num_nodes = a.ncols();
        let output_dim = num_nodes * D::dim();
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let lame_reparam = reparametrize_lame_for_stable_neo_hookean(T::from_usize(d).unwrap(), &self.lame);
        let dp_df = build_stable_neohookean_dp_df(lame_reparam, F);

        // We have that F = Identity + sum_J U_J \otimes (grad phi_J),
        // where \otimes denotes the outer product, U_J is the d-dimensional weight
        // associated with node J, and grad phi_J is the gradient of basis function J.
        // The result is that, with A given by
        //  A = [ grad phi_1   grad phi_2   ...   grad phi_N ],
        // the Jacobian matrix df/du is given by
        //  df/du = A \otimes I,
        // where \otimes is the Kronecker product and I the d x d identity matrix.
        // However, we don't actually compute it this way, because
        // this does not take into account the inherent sparsity in the Kronecker expression,
        // thus we instead use a custom implementation that works with the tensor indices directly.
        contract_stiffness_tensor(output, &dp_df, &a);
    }
}

/// A semi-definite Neo-Hookean type material model that is stable and robust to inversions.
///
/// Implements the material model proposed by Smith et al. [2018] in the paper
/// "Stable Neo-Hookean Flesh Simulation".
///
/// This model includes the projection onto semi-definiteness,
/// and as such will produce contractions which are semi-definite.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectedStableNeoHookeanMaterial<T> {
    pub lame: LameParameters<T>,
}

impl<T, X> From<X> for ProjectedStableNeoHookeanMaterial<T>
where
    X: Into<LameParameters<T>>,
{
    fn from(params: X) -> Self {
        Self { lame: params.into() }
    }
}

#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T> ElasticMaterialModel<T, U3> for ProjectedStableNeoHookeanMaterial<T>
where
    T: RealField,
{
    fn is_positive_semi_definite(&self) -> bool {
        true
    }

    #[allow(non_snake_case)]
    fn compute_strain_energy_density(&self, _F: &Matrix3<T>) -> T {
        todo!()
    }

    #[allow(non_snake_case)]
    fn compute_stress_tensor(&self, F: &Matrix3<T>) -> Matrix3<T> {
        StableNeoHookeanMaterial {
            lame: self.lame.clone(),
        }
        .compute_stress_tensor(F)
    }

    #[allow(non_snake_case)]
    fn contract_stress_tensor_with(&self, _F: &Matrix3<T>, _a: &Vector3<T>, _b: &Vector3<T>) -> Matrix3<T> {
        todo!()
    }

    #[allow(non_snake_case)]
    #[inline(never)]
    fn contract_multiple_stress_tensors_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        F: &Matrix3<T>,
        a: &MatrixSliceMN<T, U3, Dynamic>,
    ) {
        let d = 3;
        let num_nodes = a.ncols();
        let output_dim = num_nodes * d;
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let lame_reparam = reparametrize_lame_for_stable_neo_hookean(T::from_usize(d).unwrap(), &self.lame);
        let mut dp_df_eigendecomp = build_stable_neohookean_dp_df_eigen_3d(lame_reparam, F);
        for eval in &mut dp_df_eigendecomp.eigenvalues {
            *eval = T::max(T::zero(), *eval);
        }
        let dp_df = dp_df_eigendecomp.recompose();
        contract_stiffness_tensor(output, &dp_df, &a);
    }
}

#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T> ElasticMaterialModel<T, U2> for ProjectedStableNeoHookeanMaterial<T>
where
    T: RealField,
{
    fn is_positive_semi_definite(&self) -> bool {
        true
    }

    #[allow(non_snake_case)]
    fn compute_strain_energy_density(&self, _F: &Matrix2<T>) -> T {
        todo!()
    }

    #[allow(non_snake_case)]
    fn compute_stress_tensor(&self, F: &Matrix2<T>) -> Matrix2<T> {
        StableNeoHookeanMaterial {
            lame: self.lame.clone(),
        }
        .compute_stress_tensor(F)
    }

    #[allow(non_snake_case)]
    fn contract_stress_tensor_with(&self, _F: &Matrix2<T>, _a: &Vector2<T>, _b: &Vector2<T>) -> Matrix2<T> {
        todo!()
    }

    #[allow(non_snake_case)]
    #[inline(never)]
    fn contract_multiple_stress_tensors_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        F: &Matrix2<T>,
        a: &MatrixSliceMN<T, U2, Dynamic>,
    ) {
        let d = 2;
        let num_nodes = a.ncols();
        let output_dim = num_nodes * d;
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let lame_reparam = reparametrize_lame_for_stable_neo_hookean(T::from_usize(d).unwrap(), &self.lame);
        let dp_df = build_stable_neohookean_dp_df(lame_reparam, F);
        // TODO: We currently don't have analytic projection, so we resort to numerical
        let mut dp_df_eigendecomp = dp_df.symmetric_eigen();
        for eval in &mut dp_df_eigendecomp.eigenvalues {
            *eval = T::max(T::zero(), *eval);
        }
        let dp_df = dp_df_eigendecomp.recompose();
        contract_stiffness_tensor(output, &dp_df, &a);
    }
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
fn build_stable_neohookean_dp_df<T, D>(
    reparametrized_lame: LameParameters<T>,
    F: &MatrixN<T, D>,
) -> MatrixN<T, DimProd<D, D>>
where
    T: RealField,
    D: DimName + DimMul<D> + DimMin<D, Output = D>,
    DimProd<D, D>: DimName,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, D, D>
        + Allocator<(usize, usize), D>
        + Allocator<T, U1, D>
        + Allocator<T, DimProd<D, D>, DimProd<D, D>>
        + Allocator<T, DimProd<D, D>>,
{
    let d = T::from_usize(D::dim()).unwrap();
    let LameParameters { mu, lambda } = reparametrized_lame;
    let alpha = 1.0 + mu / lambda * (d / (d + 1.0));

    let C = F.transpose() * F;
    let I_C = C.trace();
    let J = F.determinant();
    let beta = I_C + 1.0;
    let beta2 = beta * beta;

    // TODO: Use cofactor matrix instead of
    // relying on inverse
    let dJ_dF = F
        .clone()
        .try_inverse()
        .expect("TODO: Handle singular F?")
        .transpose()
        * J;

    let f = vectorize(F);
    let g = vectorize(&dJ_dF);

    // Build dP/dF in vec( ) form, i.e. dp/df,
    // where p = vec(P) and f = vec(F)
    let mut dp_df = MatrixN::<_, DimProd<D, D>>::zeros();
    // Tikhonov term
    dp_df.fill_diagonal(mu * (1.0 - 1.0 / beta));

    // M-term, const * f f^T
    dp_df.ger(mu * 2.0 / beta2, &f, &f, 1.0);

    // G-term, const * g g^T
    dp_df.ger(lambda, &g, &g, 1.0);

    add_volume_hessian(&mut dp_df, lambda * (J - alpha), F);
    dp_df
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
fn build_stable_neohookean_dp_df_eigen_3d<T>(
    reparametrized_lame: LameParameters<T>,
    F: &Matrix3<T>,
) -> SymmetricEigen<T, U9>
where
    T: RealField,
{
    let (u, s, v_t) = rotation_svd(F);

    // J = det(F) = s_1 * s_2 * s_3
    // let J = s[0] * s[1] * s[2];
    // TODO: Use the above expression instead once we've confirmed that things work
    let J = F.determinant();
    let C = F.transpose() * F;
    let I_C = C.trace();

    let d = 3.0;
    let LameParameters { mu, lambda } = reparametrized_lame;
    let alpha = 1.0 + mu / lambda * (d / (d + 1.0));

    // Regularization term corresponding to the Identity (Tikhonov) regularization
    let mu_t = mu * (1.0 - 1.0 / (I_C + 1.0));

    let mut eigenvalues = VectorN::<T, U9>::zeros();
    let mut eigenvectors = MatrixN::<T, U9>::zeros();

    // The first six eigenpairs are given by the so-called Twist & Flip matrices
    // (see Smith et al., Analytic Eigensystems for Isotropic Distortion Energies)
    // Note that we use the conventions from the original Stable Neo-Hookean paper, however
    {
        let sv_scale = lambda * (J - alpha);
        let mu_t_vec = Vector3::repeat(mu_t.clone());
        eigenvalues
            .fixed_slice_mut::<U3, U1>(0, 0)
            .copy_from(&(s * sv_scale + mu_t_vec));
        eigenvalues
            .fixed_slice_mut::<U3, U1>(3, 0)
            .copy_from(&(-s * sv_scale + mu_t_vec));

        // The eigenmatrices D_i are given by
        //  D_0 = (1/sqrt(2)) * [ u1 o v2 - u2 o v1 ]
        //  D_1 = (1/sqrt(2)) * [ u0 o v2 - u2 o v0 ]
        //  D_2 = (1/sqrt(2)) * [ u0 o v1 - u1 o v0 ]
        //
        //  D_3 = (1/sqrt(2)) * [ u1 o v2 + u2 o v1 ]
        //  D_4 = (1/sqrt(2)) * [ u0 o v2 + u2 o v0 ]
        //  D_5 = (1/sqrt(2)) * [ u0 o v1 + u1 o v0 ]
        // where u0 is the 0th col of U, v0 is the 0th col of V and so on. "o" here denotes
        // the outer product of column vectors.

        // We bake the constant into V to avoid some unnecessary multiplications
        // Moreover, since we only have the transpose of V, we must take *rows* of V rather
        // than columns.
        let v_t_scaled = v_t / T::sqrt(2.0);
        let u0v1 = u.column(0) * v_t_scaled.row(1);
        let u0v2 = u.column(0) * v_t_scaled.row(2);
        let u1v0 = u.column(1) * v_t_scaled.row(0);
        let u1v2 = u.column(1) * v_t_scaled.row(2);
        let u2v0 = u.column(2) * v_t_scaled.row(0);
        let u2v1 = u.column(2) * v_t_scaled.row(1);

        let mut set_eigenmatrix = |index, eigenmatrix| {
            eigenvectors
                .column_mut(index)
                .copy_from(&vectorize(&eigenmatrix));
        };

        set_eigenmatrix(0, u1v2 - u2v1);
        set_eigenmatrix(1, u0v2 - u2v0);
        set_eigenmatrix(2, u0v1 - u1v0);

        set_eigenmatrix(3, u1v2 + u2v1);
        set_eigenmatrix(4, u0v2 + u2v0);
        set_eigenmatrix(5, u0v1 + u1v0);
    }

    // The final three eigenpairs are associated with the roots of a cubic equation,
    // as detailed in the Stable Neo-Hookean paper. However, this is fairly complicated,
    // so instead we take the much more practical route described in
    // Smith et al., Analytic Eigensystems for Isotropic Distortion Energies
    // of exploiting the fact that the remaining eigenpairs are related to the eigenpairs
    // of a dxd matrix, which may be numerically computed.
    {
        let beta = I_C + 1.0;
        let beta2 = beta * beta;
        // Construct the 3x3 matrix A
        let mut a = Matrix3::zeros();
        // Diagonal
        a[(0, 0)] = mu_t + 2.0 * s[0] * s[0] * mu / beta2 + s[1] * s[1] * s[2] * s[2] * lambda;
        a[(1, 1)] = mu_t + 2.0 * s[1] * s[1] * mu / beta2 + s[0] * s[0] * s[2] * s[2] * lambda;
        a[(2, 2)] = mu_t + 2.0 * s[2] * s[2] * mu / beta2 + s[0] * s[0] * s[1] * s[1] * lambda;

        // Off-diagonal (A is symmetric)
        let gamma = lambda * (2.0 * J - alpha);
        a[(0, 1)] = s[2] * gamma + 2.0 * s[0] * s[1] * mu / beta2;
        a[(0, 2)] = s[1] * gamma + 2.0 * s[0] * s[2] * mu / beta2;
        a[(1, 2)] = s[0] * gamma + 2.0 * s[1] * s[2] * mu / beta2;
        a[(1, 0)] = a[(0, 1)];
        a[(2, 0)] = a[(0, 2)];
        a[(2, 1)] = a[(1, 2)];

        let a_eigen = a.symmetric_eigen();

        // Each eigenvalue of a is directly an eigenvalue of dp/df
        eigenvalues
            .fixed_slice_mut::<U3, U1>(6, 0)
            .copy_from(&a_eigen.eigenvalues);

        // For each eigenvector of A, the components of the eigenvectors correspond to
        // weights of the remaining "scaling directions", i.e.
        //  q_i = vec(Q_i), i = 1, 2, 3
        // with
        //  Q_i = U * (e_i o e_i) V^T
        let Q_1 = &u * diag_left_mul(&a_eigen.eigenvectors.column(0), &v_t);
        let Q_2 = &u * diag_left_mul(&a_eigen.eigenvectors.column(1), &v_t);
        let Q_3 = &u * diag_left_mul(&a_eigen.eigenvectors.column(2), &v_t);

        eigenvectors.column_mut(6).copy_from(&vectorize(&Q_1));
        eigenvectors.column_mut(7).copy_from(&vectorize(&Q_2));
        eigenvectors.column_mut(8).copy_from(&vectorize(&Q_3));
    }

    SymmetricEigen {
        eigenvectors,
        eigenvalues,
    }
}

#[allow(non_snake_case)]
fn contract_stiffness_tensor<T, D>(
    output: &mut DMatrixSliceMut<T>,
    dp_df: &MatrixN<T, DimProd<D, D>>,
    a: &MatrixSliceMN<T, D, Dynamic>,
) where
    T: RealField,
    D: DimName + DimMul<D>,
    DimProd<D, D>: DimName,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, DimProd<D, D>, DimProd<D, D>> + Allocator<T, DimProd<D, D>>,
{
    let d = D::dim();
    assert_eq!(output.nrows(), output.ncols());
    assert_eq!(output.nrows() % d, 0);
    let num_nodes = output.nrows() / d;

    // Capital I, J denote numberings, lower-case i, j denote dimension numbering
    for J in 0..num_nodes {
        for I in J..num_nodes {
            let mut result_IJ = MatrixN::<_, D>::zeros();
            let a_I = a.fixed_slice::<D, U1>(0, I);
            let a_J = a.fixed_slice::<D, U1>(0, J);

            for l in 0..d {
                for k in 0..d {
                    for j in 0..d {
                        for i in 0..d {
                            // Convert tensor indices to linear row/col indices in dp/df
                            let linear_col = d * l + j;
                            let linear_row = d * k + i;
                            unsafe {
                                let dp_df_ikjn = dp_df
                                    .get_unchecked((linear_row, linear_col))
                                    .inlined_clone();
                                let a_Ik = a_I.get_unchecked(k).inlined_clone();
                                let a_Jl = a_J.get_unchecked(l).inlined_clone();
                                *result_IJ.get_unchecked_mut((i, j)) += dp_df_ikjn * a_Ik * a_Jl;
                            }
                        }
                    }
                }
            }

            output
                .fixed_slice_mut::<D, D>(I * d, J * d)
                .add_assign(&result_IJ);

            if I != J {
                output
                    .fixed_slice_mut::<D, D>(J * d, I * d)
                    .add_assign(&result_IJ.transpose());
            }
        }
    }
}

fn vectorize<T, D>(matrix: &MatrixN<T, D>) -> VectorN<T, DimProd<D, D>>
where
    T: RealField,
    D: DimName + DimMul<D>,
    DimProd<D, D>: DimName,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, DimProd<D, D>>,
{
    let mut result = VectorN::zeros();
    let m = matrix.nrows();
    let n = matrix.ncols();
    for j in 0..n {
        for i in 0..m {
            result[n * j + i] = matrix[(i, j)];
        }
    }
    result
}

fn add_volume_hessian_3d<T: RealField>(matrix: &mut MatrixN<T, U9>, scale: T, deformation_gradient: &Matrix3<T>) {
    // Pre-multiply the scale into the hat matrices, so that it suffices
    // to add them to the output matrix afterwards
    let f0_hat = cross_product_matrix(&deformation_gradient.column(0).clone_owned()) * scale;
    let f1_hat = cross_product_matrix(&deformation_gradient.column(1).clone_owned()) * scale;
    let f2_hat = cross_product_matrix(&deformation_gradient.column(2).clone_owned()) * scale;

    matrix.fixed_slice_mut::<U3, U3>(3, 0).add_assign(&f2_hat);
    matrix.fixed_slice_mut::<U3, U3>(6, 0).sub_assign(&f1_hat);
    matrix.fixed_slice_mut::<U3, U3>(0, 3).sub_assign(&f2_hat);
    matrix.fixed_slice_mut::<U3, U3>(6, 3).add_assign(&f0_hat);
    matrix.fixed_slice_mut::<U3, U3>(0, 6).add_assign(&f1_hat);
    matrix.fixed_slice_mut::<U3, U3>(3, 6).sub_assign(&f0_hat);
}

#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
fn add_volume_hessian_2d<T: RealField>(matrix: &mut MatrixN<T, U4>, scale: T, _deformation_gradient: &Matrix2<T>) {
    // Pre-multiply the scale into the hat matrices, so that it suffices
    // to add them to the output matrix afterwards
    let s = scale;
    matrix.add_assign(&Matrix4::new(
        0.0, 0.0, 0.0, s, 0.0, 0.0, -s, 0.0, 0.0, -s, 0.0, 0.0, s, 0.0, 0.0, 0.0,
    ));
}

fn add_volume_hessian<T, D>(
    matrix: &mut MatrixN<T, <D as DimMul<D>>::Output>,
    scale: T,
    deformation_gradient: &MatrixN<T, D>,
) where
    T: RealField,
    D: DimName + DimMul<D>,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, D, D> + Allocator<T, <D as DimMul<D>>::Output, <D as DimMul<D>>::Output>,
{
    if TypeId::of::<D>() == TypeId::of::<U3>() {
        assert_eq!(TypeId::of::<DimProd<D, D>>(), TypeId::of::<U9>());
        let matrix = try_transmute_ref_mut(matrix).unwrap();
        let deformation_gradient = try_transmute_ref(deformation_gradient).unwrap();
        add_volume_hessian_3d(matrix, scale, deformation_gradient);
    } else if TypeId::of::<D>() == TypeId::of::<U2>() {
        let matrix = try_transmute_ref_mut(matrix).unwrap();
        let deformation_gradient = try_transmute_ref(deformation_gradient).unwrap();
        add_volume_hessian_2d(matrix, scale, deformation_gradient);
    } else {
        unimplemented!("Only 2D and 3D are supported");
    }
}

/// Approximate the material model stiffness contraction
/// with B, given the deformation gradient F.
///
/// Uses finite differences with parameter h.
#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn approximate_stiffness_contraction<T, D>(
    material: &impl ElasticMaterialModel<T, D>,
    F: &MatrixN<T, D>,
    a: &VectorN<T, D>,
    b: &VectorN<T, D>,
    h: T,
) -> MatrixN<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    let mut result: MatrixN<T, D> = MatrixN::<T, D>::zeros();

    // Construct a matrix e_ij whose coefficients are all zero, except for ij, which satisfies
    // (e_ij)_ij == 1
    let e = |i, j| {
        let mut result = MatrixN::<T, D>::zeros();
        result[(i, j)] = 1.0;
        result
    };

    let P = |F| material.compute_stress_tensor(&F);

    // Use finite differences to compute a numerical approximation of the
    // contraction between dP/dF and B
    for i in 0..D::dim() {
        for j in 0..D::dim() {
            for l in 0..D::dim() {
                for n in 0..D::dim() {
                    let dF_jn_plus = F + e(j, n) * h;
                    let dF_jn_minus = F - e(j, n) * h;
                    // Second order central difference
                    let D_iljn = (P(dF_jn_plus)[(i, l)] - P(dF_jn_minus)[(i, l)]) / (2.0 * h);
                    result[(i, j)] += D_iljn * a[l] * b[n];
                }
            }
        }
    }

    result
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvertibleMaterial<T, M> {
    material: M,
    threshold: T,
}

impl<T, M> InvertibleMaterial<T, M> {
    pub fn new(material: M, threshold: T) -> Self {
        Self { material, threshold }
    }
}

/// Projects the eigenvalues of the stretch tensor S onto an admissible set of eigenvalues.
#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
fn project_deformation_gradient<D, T>(F: &MatrixN<T, D>, threshold: T) -> MatrixN<T, D>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D> + DimSub<U1>,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, D, D> + Allocator<T, <D as DimSub<U1>>::Output> + Allocator<(usize, usize), D>,
{
    let mut F_svd = F.clone().svd(true, true);
    let u = F_svd.u.as_mut().unwrap();
    let v_t = F_svd.v_t.as_mut().unwrap();

    let inversion = u.determinant().signum() != v_t.determinant().signum();

    if inversion {
        let smallest_sv_index = F_svd.singular_values.imin();
        let mut u_col = u.index_mut((.., smallest_sv_index));
        u_col *= -1.0;
        F_svd.singular_values[smallest_sv_index] *= -1.0;
    }

    let mut correction_necessary = false;
    for sv in &mut F_svd.singular_values {
        if *sv < threshold {
            *sv = threshold;
            correction_necessary = true;
        }
    }

    // Since the SVD may be inaccurate, make sure to only recompose
    // when we actually changed something
    let F = if correction_necessary {
        F_svd
            .recompose()
            .expect("Can not fail, since we have computed u and v_t")
    } else {
        F.clone()
    };
    // Sanity check
    assert!(F.iter().all(|x_i| x_i.is_finite()));
    F
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D, M> ElasticMaterialModel<T, D> for InvertibleMaterial<T, M>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D> + DimSub<U1>,
    M: ElasticMaterialModel<T, D>,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, D, D>
        + Allocator<T, <D as DimSub<U1>>::Output>
        + Allocator<(usize, usize), D>
        + Allocator<T, U1, D>,
{
    fn compute_strain_energy_density(&self, F: &MatrixN<T, D>) -> T {
        let F = project_deformation_gradient(F, self.threshold);
        self.material.compute_strain_energy_density(&F)
    }

    fn compute_stress_tensor(&self, F: &MatrixN<T, D>) -> MatrixN<T, D> {
        let F = project_deformation_gradient(F, self.threshold);
        self.material.compute_stress_tensor(&F)
    }

    fn contract_stress_tensor_with(&self, F: &MatrixN<T, D>, a: &VectorN<T, D>, b: &VectorN<T, D>) -> MatrixN<T, D> {
        let F = project_deformation_gradient(F, self.threshold);
        self.material.contract_stress_tensor_with(&F, a, b)
    }
}

#[cfg(test)]
mod tests {
    use crate::solid::materials::{
        build_stable_neohookean_dp_df, build_stable_neohookean_dp_df_eigen_3d,
        reparametrize_lame_for_stable_neo_hookean, YoungPoisson,
    };
    use nalgebra::Matrix3;

    #[allow(non_snake_case)]
    #[test]
    fn stable_neo_hookean_analytic_decomposition_matches_numerical_3d() {
        let young_poisson = YoungPoisson {
            young: 1e2f64,
            poisson: 0.2f64,
        };
        let F = Matrix3::new(0.5, 0.1, -0.2, 1.0, 1.5, 0.0, -0.1, -0.7, 0.9);
        let lame = reparametrize_lame_for_stable_neo_hookean(3.0, &young_poisson.into());
        let mut df_dp_eigen_analytic = build_stable_neohookean_dp_df_eigen_3d(lame, &F);
        let mut df_dp_eigen_numerical = build_stable_neohookean_dp_df(lame, &F).symmetric_eigen();

        let df_dp_analytic = df_dp_eigen_analytic.recompose();
        let df_dp_numerical = df_dp_eigen_numerical.recompose();

        // Sort eigenvalues before comparison
        df_dp_eigen_analytic
            .eigenvalues
            .as_mut_slice()
            .sort_by(|a, b| b.partial_cmp(&a).unwrap());
        df_dp_eigen_numerical
            .eigenvalues
            .as_mut_slice()
            .sort_by(|a, b| b.partial_cmp(&a).unwrap());

        let eigenvalue_diff = df_dp_eigen_analytic.eigenvalues - df_dp_eigen_numerical.eigenvalues;
        let tol = 1e-12 * df_dp_eigen_numerical.eigenvalues.amax();
        assert!(eigenvalue_diff.amax() <= tol);

        // Note: We don't compare eigenvectors directly, because they may differ in sign.
        // Instead, we reconstruct dp_df and compare.
        let tol = 1e-12 * df_dp_numerical.amax();
        let df_dp_diff = df_dp_analytic - df_dp_numerical;
        assert!(df_dp_diff.amax() <= tol);
    }
}
