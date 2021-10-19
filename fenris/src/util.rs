use crate::assembly::CsrParAssembler;
use crate::connectivity::Connectivity;
use crate::mesh::Mesh;
use crate::{CooMatrix, CsrMatrix};
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use nalgebra::storage::{Storage, StorageMut};
use nalgebra::{
    DMatrixSlice, DVector, DVectorSlice, DefaultAllocator, Dim, DimDiff, DimMin, DimName, DimSub, Dynamic, Matrix,
    Matrix3, MatrixMN, MatrixN, MatrixSlice, MatrixSliceMut, Quaternion, RealField, Scalar, SliceStorage,
    SliceStorageMut, SquareMatrix, UnitQuaternion, Vector, Vector3, VectorN, U1,
};
use num::Zero;
use numeric_literals::replace_float_literals;
use std::error::Error;
use std::fmt::Display;
use std::fmt::LowerExp;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Add;
use std::path::Path;
use std::sync::Arc;

/// Creates a column-major slice from the given matrix.
///
/// Panics if the matrix does not have column-major storage.
pub fn coerce_col_major_slice<T, R, C, S, RSlice, CSlice>(
    matrix: &Matrix<T, R, C, S>,
    slice_rows: RSlice,
    slice_cols: CSlice,
) -> MatrixSlice<T, RSlice, CSlice, U1, RSlice>
where
    T: Scalar,
    R: Dim,
    RSlice: Dim,
    C: Dim,
    CSlice: Dim,
    S: Storage<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>,
{
    assert_eq!(slice_rows.value(), matrix.nrows());
    assert_eq!(slice_cols.value(), matrix.ncols());
    let (rstride, cstride) = matrix.strides();
    assert!(
        rstride == 1 && cstride == matrix.nrows(),
        "Matrix must have column-major storage."
    );

    unsafe {
        let data =
            SliceStorage::new_with_strides_unchecked(&matrix.data, (0, 0), (slice_rows, slice_cols), (U1, slice_rows));
        Matrix::from_data_statically_unchecked(data)
    }
}

/// An SVD-like decomposition in which the orthogonal matrices `U` and `V` are rotation matrices.
///
/// Given a matrix `A`, this method returns factors `U`, `S` and `V` such that
/// `A = U S V^T`, with `U, V` orthogonal and `det(U) = det(V) = 1` and `S` a diagonal matrix
/// whose entries are represented by a vector.
///
/// Note that unlike the standard SVD, `S` may contain negative entries, and so they do not
/// generally coincide with singular values. However, it holds that `S(i)^2 == sigma_i^2`, where
/// `sigma_i` is the `i`th singular value of `A`.
///
/// Returns a tuple `(U, S, V^T)`.
pub fn rotation_svd<T, D>(matrix: &MatrixN<T, D>) -> (MatrixN<T, D>, VectorN<T, D>, MatrixN<T, D>)
where
    T: RealField,
    D: DimName + DimMin<D, Output = D> + DimSub<U1>,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, D, D> + Allocator<T, <D as DimSub<U1>>::Output> + Allocator<(usize, usize), D>,
{
    let minus_one = T::from_f64(-1.0).unwrap();
    let mut svd = matrix.clone().svd(true, true);
    let min_val_idx = svd.singular_values.imin();

    let mut u = svd.u.unwrap();
    if u.determinant() < T::zero() {
        let mut u_col = u.column_mut(min_val_idx);
        u_col *= minus_one;
        svd.singular_values[min_val_idx] *= minus_one;
    }

    let mut v_t = svd.v_t.unwrap();
    if v_t.determinant() < T::zero() {
        let mut v_t_row = v_t.row_mut(min_val_idx);
        v_t_row *= minus_one;
        svd.singular_values[min_val_idx] *= minus_one;
    }

    (u, svd.singular_values, v_t)
}

/// "Analytic polar decomposition"
///
/// Translated to Rust from https://github.com/InteractiveComputerGraphics/FastCorotatedFEM/blob/351b007b6bb6e8d97f457766e9ecf9b2bced7079/FastCorotFEM.cpp#L413
///
/// ```
/// use fenris::util::apd;
/// use nalgebra::{Matrix3, UnitQuaternion, Quaternion, Vector3};
///
/// let eps: f64 = 1e-12;
/// let guess = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2);
/// assert!((apd::<f64>(&Matrix3::identity(), &guess, 100, eps).as_ref() - &Quaternion::identity()).norm() < 1.0e1 * eps);
/// assert!((apd::<f64>(&Matrix3::identity(), &guess, 100, eps).as_ref() - guess.as_ref()).norm() > 1.0e2 * eps);
/// ```
///
#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn apd<T: RealField>(
    deformation_grad: &Matrix3<T>,
    initial_guess: &UnitQuaternion<T>,
    max_iter: usize,
    tol: T,
) -> UnitQuaternion<T> {
    let F = deformation_grad;
    let mut q: UnitQuaternion<T> = initial_guess.clone();

    let tol_squared = tol * tol;
    let mut res = T::max_value();
    let mut iter = 0;
    while res > tol_squared && iter < max_iter {
        let R = q.to_rotation_matrix();
        let B = R.transpose() * F;

        let B0 = B.column(0);
        let B1 = B.column(1);
        let B2 = B.column(2);

        let gradient = Vector3::new(B2[1] - B1[2], B0[2] - B2[0], B1[0] - B0[1]);

        // compute Hessian, use the fact that it is symmetric
        let h00 = B1[1] + B2[2];
        let h11 = B0[0] + B2[2];
        let h22 = B0[0] + B1[1];
        let h01 = (B1[0] + B0[1]) * 0.5;
        let h02 = (B2[0] + B0[2]) * 0.5;
        let h12 = (B2[1] + B1[2]) * 0.5;

        let detH =
            -(h02 * h02 * h11) + (h01 * h02 * h12) * 2.0 - (h00 * h12 * h12) - (h01 * h01 * h22) + (h00 * h11 * h22);
        let factor = detH.recip() * (-0.25);

        let mut omega = Vector3::zeros();

        // compute symmetric inverse
        omega[0] = (h11 * h22 - h12 * h12) * gradient[0]
            + (h02 * h12 - h01 * h22) * gradient[1]
            + (h01 * h12 - h02 * h11) * gradient[2];
        omega[1] = (h02 * h12 - h01 * h22) * gradient[0]
            + (h00 * h22 - h02 * h02) * gradient[1]
            + (h01 * h02 - h00 * h12) * gradient[2];
        omega[2] = (h01 * h12 - h02 * h11) * gradient[0]
            + (h01 * h02 - h00 * h12) * gradient[1]
            + (h00 * h11 - h01 * h01) * gradient[2];
        omega *= factor;

        // if det(H) = 0 use gradient descent, never happened in our tests, could also be removed
        if detH.abs() < 1.0e-9 {
            omega = -gradient;
        }

        // instead of clamping just use gradient descent. also works fine and does not require the norm
        let useGD = omega.dot(&gradient) > T::zero();
        if useGD {
            omega = &gradient * (-0.125);
        }

        let l_omega2 = omega.norm_squared();

        let w = (1.0 - l_omega2) / (1.0 + l_omega2);
        let vec = omega * (2.0 / (1.0 + l_omega2));

        // no normalization needed because the Cayley map returs a unit quaternion
        q = q * UnitQuaternion::new_unchecked(Quaternion::from_parts(w, vec));

        iter += 1;
        res = l_omega2;
    }

    q
}

pub fn diag_left_mul<T, D1, D2, S>(diag: &Vector<T, D1, S>, matrix: &MatrixMN<T, D1, D2>) -> MatrixMN<T, D1, D2>
where
    T: RealField,
    D1: DimName,
    D2: DimName,
    S: Storage<T, D1>,
    DefaultAllocator: Allocator<T, D1, D2>,
{
    // TODO: This is inefficient
    let mut result = matrix.clone();
    for (i, mut row) in result.row_iter_mut().enumerate() {
        row *= diag[i];
    }
    result
}

/// Creates a mutable column-major slice from the given matrix.
///
/// Panics if the matrix does not have column-major storage.
pub fn coerce_col_major_slice_mut<T, R, C, S, RSlice, CSlice>(
    matrix: &mut Matrix<T, R, C, S>,
    slice_rows: RSlice,
    slice_cols: CSlice,
) -> MatrixSliceMut<T, RSlice, CSlice, U1, RSlice>
where
    T: Scalar,
    R: Dim,
    RSlice: Dim,
    C: Dim,
    CSlice: Dim,
    S: StorageMut<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>,
{
    assert_eq!(slice_rows.value(), matrix.nrows());
    assert_eq!(slice_cols.value(), matrix.ncols());
    let (rstride, cstride) = matrix.strides();
    assert!(
        rstride == 1 && cstride == matrix.nrows(),
        "Matrix must have column-major storage."
    );

    unsafe {
        let data = SliceStorageMut::new_with_strides_unchecked(
            &mut matrix.data,
            (0, 0),
            (slice_rows, slice_cols),
            (U1, slice_rows),
        );
        Matrix::from_data_statically_unchecked(data)
    }
}

pub fn try_transmute_ref<T: 'static, U: 'static>(e: &T) -> Option<&U> {
    use std::any::TypeId;
    use std::mem::transmute;
    if TypeId::of::<T>() == TypeId::of::<U>() {
        Some(unsafe { transmute(e) })
    } else {
        None
    }
}

pub fn try_transmute_ref_mut<T: 'static, U: 'static>(e: &mut T) -> Option<&mut U> {
    use std::any::TypeId;
    use std::mem::transmute;
    if TypeId::of::<T>() == TypeId::of::<U>() {
        Some(unsafe { transmute(e) })
    } else {
        None
    }
}

pub fn cross_product_matrix<T: RealField>(x: &Vector3<T>) -> Matrix3<T> {
    Matrix3::new(T::zero(), -x[2], x[1], x[2], T::zero(), -x[0], -x[1], x[0], T::zero())
}

pub fn dump_matrix_to_file<'a, T: Scalar + Display>(
    path: impl AsRef<Path>,
    matrix: impl Into<DMatrixSlice<'a, T>>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    let matrix = matrix.into();
    for i in 0..matrix.nrows() {
        write!(writer, "{}", matrix[(i, 0)])?;
        for j in 1..matrix.ncols() {
            write!(writer, " {}", matrix[(i, j)])?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;

    Ok(())
}

/// Dumps matrices corresponding to node-node connectivity and element-node connectivity
/// to the Matrix Market sparse storage format.
pub fn dump_mesh_connectivity_matrices<T, D, C>(
    node_path: impl AsRef<Path>,
    element_path: impl AsRef<Path>,
    mesh: &Mesh<T, D, C>,
) -> Result<(), Box<dyn Error>>
where
    T: Scalar + LowerExp,
    D: DimName,
    C: Sync + Connectivity,
    DefaultAllocator: Allocator<T, D>,
    Mesh<T, D, C>: Sync,
{
    let pattern = CsrParAssembler::<usize>::default().assemble_pattern(mesh);
    let nnz = pattern.nnz();
    let node_matrix = CsrMatrix::from_pattern_and_values(Arc::new(pattern), vec![1.0f64; nnz]);

    dump_csr_matrix_to_mm_file(node_path.as_ref(), &node_matrix).map_err(|err| err as Box<dyn Error>)?;

    // Create a rectangular matrix with element index on the rows and
    // node indices as columns
    let mut element_node_matrix = CooMatrix::new(mesh.connectivity().len(), mesh.vertices().len());
    for (i, conn) in mesh.connectivity().iter().enumerate() {
        for &j in conn.vertex_indices() {
            element_node_matrix.push(i, j, 1.0f64);
        }
    }

    dump_csr_matrix_to_mm_file(element_path.as_ref(), &element_node_matrix.to_csr(Add::add))
        .map_err(|err| err as Box<dyn Error>)?;
    Ok(())
}

/// Dumps a CSR matrix to a matrix market file.
///
/// TODO: Support writing integers etc. Probably need a custom trait for this
/// for writing the correct header, as well as for formatting numbers correctly
/// (scientific notation for floating point, integer for integers)
pub fn dump_csr_matrix_to_mm_file<T: Scalar + LowerExp>(
    path: impl AsRef<Path>,
    matrix: &CsrMatrix<T>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "%%MatrixMarket matrix coordinate real general")?;

    // Write dimensions
    writeln!(writer, "{} {} {}", matrix.nrows(), matrix.ncols(), matrix.nnz())?;

    for (i, j, v) in matrix.iter() {
        // Indices have to be stored as 1-based
        writeln!(writer, "{} {} {:.e}", i + 1, j + 1, v)?;
    }
    writer.flush()?;

    Ok(())
}

pub fn flatten_vertically_into<T, R1, C1, S1, R2, C2, S2>(
    output: &mut Matrix<T, R2, C2, S2>,
    matrices: &[Matrix<T, R1, C1, S1>],
) where
    T: Scalar,
    R1: Dim,
    C1: Dim,
    S1: Storage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: StorageMut<T, R2, C2>,
    ShapeConstraint: SameNumberOfColumns<C2, C1> + SameNumberOfRows<Dynamic, R1>,
{
    if let Some(first) = matrices.first() {
        let cols = first.ncols();
        let mut rows = 0;

        for matrix in matrices {
            assert_eq!(matrix.ncols(), cols, "All matrices must have same number of columns.");
            output.rows_mut(rows, matrix.nrows()).copy_from(matrix);
            rows += matrix.nrows();
        }
        assert_eq!(
            rows,
            output.nrows(),
            "Number of rows in output must match number of total rows in input."
        );
    } else {
        assert_eq!(
            output.nrows(),
            0,
            "Can only vertically flatten empty slice of matrices into a matrix with 0 rows."
        );
    }
}

pub fn flatten_vertically<T, R, C, S>(matrices: &[Matrix<T, R, C, S>]) -> Option<MatrixMN<T, Dynamic, C>>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    ShapeConstraint: SameNumberOfRows<Dynamic, R>,
{
    if let Some(first) = matrices.first() {
        let rows = matrices.iter().map(Matrix::nrows).sum();
        let mut output = MatrixMN::zeros_generic(Dynamic::new(rows), first.data.shape().1);
        flatten_vertically_into(&mut output, matrices);
        Some(output)
    } else {
        None
    }
}

pub fn prefix_sum(counts: impl IntoIterator<Item = usize>, x0: usize) -> impl Iterator<Item = usize> {
    counts.into_iter().scan(x0, |sum, x| {
        let current = *sum;
        *sum += x;
        Some(current)
    })
}

pub fn min_eigenvalue_symmetric<T, D, S>(matrix: &SquareMatrix<T, D, S>) -> T
where
    T: RealField,
    D: Dim + DimSub<U1>,
    S: Storage<T, D, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>> + Allocator<T, D> + Allocator<T, DimDiff<D, U1>>,
{
    use std::cmp::Ordering;
    matrix
        .symmetric_eigenvalues()
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .unwrap()
        .to_owned()
}

/// Extracts D-dimensional nodal values from a global vector using a node index list
pub fn extract_by_node_index<T, D>(u: &[T], node_indices: &[usize]) -> DVector<T>
where
    T: Scalar + Copy + Zero,
    D: DimName,
{
    let u = DVectorSlice::from(u);
    let mut extracted = DVector::zeros(D::dim() * node_indices.len());
    for (i_local, &i_global) in node_indices.iter().enumerate() {
        let ui = u.fixed_rows::<D>(D::dim() * i_global);
        extracted
            .fixed_rows_mut::<D>(D::dim() * i_local)
            .copy_from(&ui);
    }
    extracted
}

pub fn min_max_symmetric_eigenvalues<T, D, S>(matrix: &SquareMatrix<T, D, S>) -> (T, T)
where
    T: RealField,
    D: Dim + DimSub<U1>,
    S: Storage<T, D, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>> + Allocator<T, D> + Allocator<T, DimDiff<D, U1>>,
{
    use std::cmp::Ordering;
    matrix
        .symmetric_eigenvalues()
        .iter()
        .minmax_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .into_option()
        .map(|(a, b)| (*a, *b))
        .unwrap()
}

pub fn condition_number_symmetric<T, D, S>(matrix: &SquareMatrix<T, D, S>) -> T
where
    T: RealField,
    D: Dim + DimSub<U1>,
    S: Storage<T, D, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>> + Allocator<T, D> + Allocator<T, DimDiff<D, U1>>,
{
    use std::cmp::Ordering;
    let (min, max) = matrix
        .symmetric_eigenvalues()
        .into_iter()
        .cloned()
        .minmax_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .into_option()
        .expect("Currently don't support empty matrices");

    max.abs() / min.abs()
}

/*
pub fn condition_number_csr<T>(matrix: &CsrMatrix<T>) -> T
where
    T: RealField + mkl_corrode::SupportedScalar,
{
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "Matrix must be square for condition number computation."
    );
    assert!(
        matrix.nrows() > 0,
        "Cannot compute condition number for empty matrix."
    );
    use mkl_corrode::mkl_sys::MKL_INT;
    use mkl_corrode::sparse::{CsrMatrixHandle, MatrixDescription};
    use std::convert::TryFrom;

    let row_offsets: Vec<_> = matrix
        .row_offsets()
        .iter()
        .cloned()
        .map(|idx| MKL_INT::try_from(idx).unwrap())
        .collect();
    let columns: Vec<_> = matrix
        .column_indices()
        .iter()
        .cloned()
        .map(|idx| MKL_INT::try_from(idx).unwrap())
        .collect();

    // TODO: This isn't 100% safe at the moment, because we don't properly enforce
    // the necessary invariants in `Csr` (but we should, it's just a lack of time)
    // TODO: Error handling
    let mkl_csr = unsafe {
        CsrMatrixHandle::from_raw_csr_data(
            matrix.nrows(),
            matrix.ncols(),
            &row_offsets[..matrix.nrows()],
            &row_offsets[1..],
            &columns,
            matrix.values(),
        )
    }
    .unwrap();

    let description = MatrixDescription::default();

    // TODO: Error handling
    let eigenresult_largest = k_largest_eigenvalues(&mkl_csr, &description, 1).unwrap();
    let eigenresult_smallest = k_smallest_eigenvalues(&mkl_csr, &description, 1).unwrap();

    let eig_max = eigenresult_largest.eigenvalues().first().unwrap();
    let eig_min = eigenresult_smallest.eigenvalues().first().unwrap();

    eig_max.abs() / eig_min.abs()
}
*/

#[cfg(feature = "proptest")]
pub mod proptest {
    use crate::sparse::SparsityPattern;
    use crate::util::prefix_sum;
    use crate::CsrMatrix;
    use nalgebra::{DMatrix, Point2, Scalar, Vector2};
    use proptest::collection::{btree_set, vec};
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Reason, TestRunner};
    use std::cmp::min;
    use std::iter::once;
    use std::sync::Arc;

    pub fn point2_f64_strategy() -> impl Strategy<Value = Point2<f64>> {
        vector2_f64_strategy().prop_map(|vector| Point2::from(vector))
    }

    pub fn vector2_f64_strategy() -> impl Strategy<Value = Vector2<f64>> {
        let xrange = prop_oneof![-3.0..3.0, -100.0..100.0];
        let yrange = xrange.clone();
        (xrange, yrange).prop_map(|(x, y)| Vector2::new(x, y))
    }

    /// Simple helper function to produce square shapes for use with matrix strategies.
    pub fn square_shape<S>(dim: S) -> impl Strategy<Value = (usize, usize)>
    where
        S: Strategy<Value = usize>,
    {
        dim.prop_map(|dim| (dim, dim))
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct DMatrixStrategy<ElementStrategy, ShapeStrategy> {
        element_strategy: ElementStrategy,
        shape_strategy: ShapeStrategy,
    }

    impl DMatrixStrategy<(), ()> {
        pub fn new() -> Self {
            Self {
                element_strategy: (),
                shape_strategy: (),
            }
        }
    }

    impl<ElementStrategy, ShapeStrategy> DMatrixStrategy<ElementStrategy, ShapeStrategy> {
        pub fn with_elements<E>(self, element_strategy: E) -> DMatrixStrategy<E, ShapeStrategy>
        where
            E: Strategy,
        {
            DMatrixStrategy {
                element_strategy,
                shape_strategy: self.shape_strategy,
            }
        }

        pub fn with_shapes<S>(self, shape_strategy: S) -> DMatrixStrategy<ElementStrategy, S>
        where
            S: Strategy<Value = (usize, usize)>,
        {
            DMatrixStrategy {
                element_strategy: self.element_strategy,
                shape_strategy,
            }
        }
    }

    impl<ElementStrategy, ShapeStrategy> Strategy for DMatrixStrategy<ElementStrategy, ShapeStrategy>
    where
        ElementStrategy: Clone + 'static + Strategy,
        ElementStrategy::Value: Scalar,
        ShapeStrategy: Clone + 'static + Strategy<Value = (usize, usize)>,
    {
        type Tree = Box<dyn ValueTree<Value = Self::Value>>;
        type Value = DMatrix<ElementStrategy::Value>;

        fn new_tree(&self, runner: &mut TestRunner) -> Result<Self::Tree, Reason> {
            let element_strategy = self.element_strategy.clone();
            self.shape_strategy
                .clone()
                .prop_flat_map(move |(nrows, ncols)| {
                    let num_elements = nrows * ncols;
                    vec(element_strategy.clone(), num_elements)
                        .prop_map(move |elements| DMatrix::from_row_slice(nrows, ncols, &elements))
                })
                .boxed()
                .new_tree(runner)
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct SparsityPatternStrategy<ShapeStrategy, MinorsPerMajorStrategy> {
        shape_strategy: ShapeStrategy,
        minors_per_major: MinorsPerMajorStrategy,
    }

    impl SparsityPatternStrategy<(), ()> {
        pub fn new() -> Self {
            Self {
                shape_strategy: (),
                minors_per_major: (),
            }
        }
    }

    impl<ShapeStrategy, MinorsPerMajorStrategy> SparsityPatternStrategy<ShapeStrategy, MinorsPerMajorStrategy> {
        pub fn with_shapes<S>(self, shape_strategy: S) -> SparsityPatternStrategy<S, MinorsPerMajorStrategy>
        where
            S: Strategy<Value = (usize, usize)>,
        {
            SparsityPatternStrategy {
                shape_strategy,
                minors_per_major: self.minors_per_major,
            }
        }

        pub fn with_num_minors_per_major<N>(self, strategy: N) -> SparsityPatternStrategy<ShapeStrategy, N>
        where
            N: Strategy<Value = usize>,
        {
            SparsityPatternStrategy {
                shape_strategy: self.shape_strategy,
                minors_per_major: strategy,
            }
        }
    }

    impl<ShapeStrategy, MinorsPerMajorStrategy> Strategy for SparsityPatternStrategy<ShapeStrategy, MinorsPerMajorStrategy>
    where
        ShapeStrategy: Clone + 'static + Strategy<Value = (usize, usize)>,
        MinorsPerMajorStrategy: Clone + 'static + Strategy<Value = usize>,
    {
        type Tree = Box<dyn ValueTree<Value = Self::Value>>;
        type Value = SparsityPattern;

        fn new_tree(&self, runner: &mut TestRunner) -> Result<Self::Tree, Reason> {
            let shape_strategy = self.shape_strategy.clone();
            let minors_per_major = self.minors_per_major.clone();
            shape_strategy
                .prop_flat_map(move |(major_dim, minor_dim)| {
                    // Given major_dim and minor_dim, generate a vector of counts,
                    // corresponding to the number of minor indices per major dimension entry
                    let minors_per_major = minors_per_major
                        .clone()
                        .prop_map(move |count| min(count, minor_dim));
                    vec(minors_per_major, major_dim)
                        .prop_flat_map(move |counts| {
                            // Construct offsets from counts
                            let offsets = prefix_sum(counts.iter().cloned().chain(once(0)), 0).collect::<Vec<_>>();

                            // We build one strategy per major entry (i.e. per row in a CSR matrix)
                            let mut major_strategies = Vec::with_capacity(major_dim);
                            for count in counts {
                                if 10 * count <= minor_dim {
                                    // If we require less than approx. 10% of minor_dim,
                                    // every pick is at least 90% likely to not be an index
                                    // we already picked, so we can generate a set
                                    major_strategies.push(
                                        btree_set(0..minor_dim, count)
                                            .prop_map(|indices| indices.into_iter().collect::<Vec<_>>())
                                            .boxed(),
                                    )
                                } else {
                                    // Otherwise, we simply shuffle the integers
                                    // [0, minor_dim) and take the `count` first
                                    let strategy = Just((0..minor_dim).collect::<Vec<_>>())
                                        .prop_shuffle()
                                        .prop_map(move |mut indices| {
                                            let indices = &mut indices[0..count];
                                            indices.sort_unstable();
                                            indices.to_vec()
                                        })
                                        .boxed();
                                    major_strategies.push(strategy);
                                }
                            }
                            (Just(major_dim), Just(minor_dim), Just(offsets), major_strategies)
                        })
                        .prop_map(move |(major_dim, minor_dim, offsets, minor_indices_by_major)| {
                            let minor_indices: Vec<usize> = minor_indices_by_major.into_iter().flatten().collect();
                            SparsityPattern::from_offsets_and_indices(major_dim, minor_dim, offsets, minor_indices)
                        })
                })
                .boxed()
                .new_tree(runner)
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct CsrStrategy<ElementStrategy, ShapeStrategy, ColsPerRowStrategy> {
        pattern_strategy: SparsityPatternStrategy<ShapeStrategy, ColsPerRowStrategy>,
        element_strategy: ElementStrategy,
    }

    impl CsrStrategy<(), (), ()> {
        pub fn new() -> Self {
            Self {
                pattern_strategy: SparsityPatternStrategy::new(),
                element_strategy: (),
            }
        }
    }

    impl<ElementStrategy, ShapeStrategy, ColsPerRowStrategy>
        CsrStrategy<ElementStrategy, ShapeStrategy, ColsPerRowStrategy>
    {
        pub fn with_elements<E>(self, element_strategy: E) -> CsrStrategy<E, ShapeStrategy, ColsPerRowStrategy>
        where
            E: Strategy,
        {
            CsrStrategy {
                pattern_strategy: self.pattern_strategy,
                element_strategy,
            }
        }

        pub fn with_shapes<S>(self, shape_strategy: S) -> CsrStrategy<ElementStrategy, S, ColsPerRowStrategy>
        where
            S: Strategy<Value = (usize, usize)>,
        {
            let pattern = self.pattern_strategy.with_shapes(shape_strategy);
            CsrStrategy {
                pattern_strategy: pattern,
                element_strategy: self.element_strategy,
            }
        }

        pub fn with_cols_per_row<N>(self, cols_per_row_strategy: N) -> CsrStrategy<ElementStrategy, ShapeStrategy, N>
        where
            N: Strategy<Value = usize>,
        {
            let pattern = self
                .pattern_strategy
                .with_num_minors_per_major(cols_per_row_strategy);
            CsrStrategy {
                pattern_strategy: pattern,
                element_strategy: self.element_strategy,
            }
        }
    }

    impl<ElementStrategy, ShapeStrategy, MinorsPerMajorStrategy> Strategy
        for CsrStrategy<ElementStrategy, ShapeStrategy, MinorsPerMajorStrategy>
    where
        ElementStrategy: Clone + 'static + Strategy,
        ShapeStrategy: Clone + 'static + Strategy<Value = (usize, usize)>,
        MinorsPerMajorStrategy: Clone + 'static + Strategy<Value = usize>,
    {
        type Tree = Box<dyn ValueTree<Value = Self::Value>>;
        type Value = CsrMatrix<ElementStrategy::Value>;

        fn new_tree(&self, runner: &mut TestRunner) -> Result<Self::Tree, Reason> {
            let element_strategy = self.element_strategy.clone();
            let pattern_strategy = self.pattern_strategy.clone();
            pattern_strategy
                .prop_flat_map(move |pattern| {
                    let nnz = pattern.nnz();
                    (Just(pattern), vec(element_strategy.clone(), nnz))
                })
                .prop_map(|(pattern, values)| CsrMatrix::from_pattern_and_values(Arc::new(pattern), values))
                .boxed()
                .new_tree(runner)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{CsrStrategy, DMatrixStrategy, SparsityPatternStrategy};
        use itertools::Itertools;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn dmatrix_strategy_respects_strategies(
                matrix in DMatrixStrategy::new()
                            .with_shapes((Just(5), 2usize..=3))
                            .with_elements(0i32 ..= 5))
            {
                prop_assert_eq!(matrix.nrows(), 5);
                prop_assert!(matrix.ncols() >= 2);
                prop_assert!(matrix.ncols() <= 3);
                prop_assert!(matrix.iter().cloned().all(|x| x >= 0 && x <= 5));
            }

            #[test]
            fn sparsity_pattern_strategy_respects_strategies(
                pattern in SparsityPatternStrategy::new()
                            .with_shapes((Just(5), 2usize..=3))
                            .with_num_minors_per_major(1usize ..= 2))
            {
                prop_assert_eq!(pattern.major_dim(), 5);
                prop_assert!(pattern.minor_dim() >= 2);
                prop_assert!(pattern.minor_dim() <= 3);

                let counts: Vec<_> = pattern.major_offsets()
                    .iter()
                    .tuple_windows()
                    .map(|(prev, next)| next - prev)
                    .collect();

                prop_assert!(counts.iter().cloned().all(|c| c >= 1 && c <= 2));
            }

            #[test]
            fn csr_strategy_respects_strategies(
                matrix in CsrStrategy::new()
                            .with_shapes((Just(5), 2usize..=3))
                            .with_cols_per_row(1usize..=2)
                            .with_elements(0i32..5))
            {
                prop_assert_eq!(matrix.nrows(), 5);
                prop_assert!(matrix.ncols() >= 2);
                prop_assert!(matrix.ncols() <= 3);
                prop_assert!(matrix.values().iter().cloned().all(|x| x >= 0 && x <= 5));

                let counts: Vec<_> = matrix.row_offsets()
                    .iter()
                    .tuple_windows()
                    .map(|(prev, next)| next - prev)
                    .collect();

                prop_assert!(counts.iter().cloned().all(|c| c >= 1 && c <= 2));
            }
        }
    }
}
