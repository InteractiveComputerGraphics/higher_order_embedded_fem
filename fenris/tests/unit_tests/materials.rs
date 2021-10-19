use fenris::solid::materials::*;
use fenris::solid::ElasticMaterialModel;
use nalgebra::{
    DMatrix, DMatrixSliceMut, DefaultAllocator, DimName, Dynamic, Matrix2, Matrix3, MatrixMN, MatrixN, MatrixSliceMN,
    Vector2, Vector3, U1, U2, U3,
};

use crate::assert_approx_matrix_eq;
use nalgebra::allocator::Allocator;
use paste;

/// Assert that material contractions are consistent with finite difference results
#[allow(non_snake_case)]
fn assert_material_contraction_consistent_with_finite_difference<D>(
    material_instances: &[impl ElasticMaterialModel<f64, D>],
    deformation_gradients: &[MatrixN<f64, D>],
    a: &MatrixMN<f64, D, Dynamic>,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D> + Allocator<usize, D, D>,
{
    // Finite difference step parameter
    let h = 1e-6;
    let num_nodes = a.ncols();
    for material in material_instances {
        for F in deformation_gradients {
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    let a_i = a.fixed_slice::<D, U1>(0, i).clone_owned();
                    let a_j = a.fixed_slice::<D, U1>(0, j).clone_owned();

                    let finite_diff = approximate_stiffness_contraction(material, F, &a_i, &a_j, h);
                    let contraction = material.contract_stress_tensor_with(&F, &a_i, &a_j);

                    let scale = f64::max(finite_diff.amax(), contraction.amax());
                    let abstol = scale * h;

                    assert_approx_matrix_eq!(&finite_diff, &contraction, abstol = abstol);
                }
            }
        }
    }
}

/// Assert that material contractions are consistent between single and batch contractions.
#[allow(non_snake_case)]
fn assert_material_consistent_contractions<D>(
    material_instances: &[impl ElasticMaterialModel<f64, D>],
    deformation_gradients: &[MatrixN<f64, D>],
    a: &MatrixMN<f64, D, Dynamic>,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    use std::ops::AddAssign;

    let dim = D::dim();
    let num_nodes = a.ncols();

    // Arbitrary value to initialize matrices with. We use this to test that the batch
    // contraction does not overwrite existing elements in the matrix, but rather
    // adds its contributions.
    let fill_value = 4.0;

    for material in material_instances {
        for F in deformation_gradients {
            let output_dim = num_nodes * dim;
            let mut output = DMatrix::repeat(output_dim, output_dim, fill_value);

            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    let a_i = a.fixed_slice::<D, U1>(0, i).clone_owned();
                    let a_j = a.fixed_slice::<D, U1>(0, j).clone_owned();
                    let block_ij = material.contract_stress_tensor_with(F, &a_i, &a_j);
                    output
                        .index_mut((i * dim..(i + 1) * dim, j * dim..(j + 1) * dim))
                        .add_assign(&block_ij);
                }
            }

            let mut batch_output = DMatrix::repeat(output_dim, output_dim, fill_value);
            material.contract_multiple_stress_tensors_into(
                &mut DMatrixSliceMut::from(&mut batch_output),
                F,
                &MatrixSliceMN::from(a),
            );

            let scale = f64::max(output.amax(), batch_output.amax());
            let abstol = 1e-12 * scale;

            assert_approx_matrix_eq!(&output, &batch_output, abstol = abstol);
        }
    }
}

macro_rules! test_material_derivatives {
    ($material_name:ident,
        $materials:expr,
        $deformation_gradients:expr,
        $contraction_vectors:expr
        $(, postfix = $postfix:ident)?) => {
        // Use paste to automatically generate method names for the tests,
        // and include a postfix for all test names
        paste::item! {
            /// Assert that the material is consistent between contract and contract_multiple
            #[test]
            #[allow(non_snake_case)]
            pub fn [<$material_name _contraction_batch_consistency $($postfix )?>]() {
                use std::borrow::Borrow;
                assert_material_consistent_contractions(
                    $materials.as_ref(),
                    $deformation_gradients.as_ref(),
                    $contraction_vectors.borrow()
                );
            }

            /// Assert that the material is approximately consistent with a finite difference
            /// discretization of the derivative
            #[test]
            #[allow(non_snake_case)]
            pub fn [<$material_name _contraction_consistent_with_finite_difference $($postfix )?>]() {
                use std::borrow::Borrow;
                assert_material_contraction_consistent_with_finite_difference(
                    $materials.as_ref(),
                    $deformation_gradients.as_ref(),
                    $contraction_vectors.borrow()
                );
            }
        }
    }
}

macro_rules! test_material_derivatives_2d {
    ($material_name:ident, $materials:expr) => {
        test_material_derivatives!(
            $material_name,
            $materials,
            test_deformation_gradients_2d(),
            contraction_test_vectors_2d(),
            postfix = _2d
        );
    };
}

macro_rules! test_material_derivatives_3d {
    ($material_name:ident, $materials:expr) => {
        test_material_derivatives!(
            $material_name,
            $materials,
            test_deformation_gradients_3d(),
            contraction_test_vectors_3d(),
            postfix = _3d
        );
    };
}

#[allow(non_snake_case)]
fn test_deformation_gradients_2d() -> Vec<Matrix2<f64>> {
    vec![
        // Identity corresponds to zero deformation
        Matrix2::identity(),
        // Singular values [2, 0.4] (non-inverted, mild anisotropic deformation)
        Matrix2::new(-1.65561115, 0.85243405, -0.83017955, -0.05576592),
        // Singular values [1e2, 1e-2] (non-inverted, strong anisotropic deformation)
        Matrix2::new(-11.11160623, -44.72115803, 21.42155763, 86.12587997),
    ]
}

#[allow(non_snake_case)]
fn test_deformation_gradients_3d() -> Vec<Matrix3<f64>> {
    vec![
        // Identity corresponds to zero deformation
        Matrix3::identity(),
        // Singular values [2.0, 0.7, 0.5] (non-inverted, mild anisotropic deformation)
        Matrix3::new(
            0.28316466,
            1.08445104,
            -1.38765817,
            0.03281728,
            -1.01281521,
            0.3086884,
            -0.54924397,
            -0.28600612,
            -0.22925947,
        ),
        // Singular values [1e2, 2, 1e-2] (non-inverted, strong anisotropic deformation)
        Matrix3::new(
            52.85734952,
            -19.73633697,
            -30.87845429,
            26.67331831,
            -10.35380109,
            -16.15435165,
            58.20810674,
            -20.01345825,
            -31.60294891,
        ),
    ]
}

fn contraction_test_vectors_2d() -> MatrixMN<f64, U2, Dynamic> {
    MatrixMN::<_, U2, Dynamic>::from_columns(&[Vector2::new(2.0, 3.0), Vector2::new(-1.0, 2.0), Vector2::new(4.0, 1.0)])
}

fn contraction_test_vectors_3d() -> MatrixMN<f64, U3, Dynamic> {
    MatrixMN::<_, U3, Dynamic>::from_columns(&[
        Vector3::new(1.0, -2.0, 3.0),
        Vector3::new(-3.0, 0.5, 1.0),
        Vector3::new(2.0, -1.0, -0.5),
    ])
}

fn young_poisson_test_parameters() -> Vec<YoungPoisson<f64>> {
    vec![
        YoungPoisson {
            young: 1e2,
            poisson: 0.1,
        },
        YoungPoisson {
            young: 1e6,
            poisson: 0.1,
        },
        YoungPoisson {
            young: 1e2,
            poisson: 0.45,
        },
        YoungPoisson {
            young: 1e6,
            poisson: 0.45,
        },
    ]
}

fn lame_test_parameters() -> Vec<LameParameters<f64>> {
    young_poisson_test_parameters()
        .into_iter()
        .map(LameParameters::from)
        .collect()
}

fn stable_neo_hookean_test_materials() -> Vec<StableNeoHookeanMaterial<f64>> {
    lame_test_parameters()
        .into_iter()
        .map(StableNeoHookeanMaterial::from)
        .collect()
}

fn linear_elastic_test_materials() -> Vec<LinearElasticMaterial<f64>> {
    lame_test_parameters()
        .into_iter()
        .map(LinearElasticMaterial::from)
        .collect()
}

fn stvk_test_materials() -> Vec<StVKMaterial<f64>> {
    lame_test_parameters()
        .into_iter()
        .map(StVKMaterial::from)
        .collect()
}

test_material_derivatives_2d!(stable_neo_hookean, stable_neo_hookean_test_materials());
test_material_derivatives_3d!(stable_neo_hookean, stable_neo_hookean_test_materials());
test_material_derivatives_2d!(linear_elastic, linear_elastic_test_materials());
test_material_derivatives_3d!(linear_elastic, linear_elastic_test_materials());
test_material_derivatives_2d!(stvk, stvk_test_materials());
test_material_derivatives_3d!(stvk, stvk_test_materials());
