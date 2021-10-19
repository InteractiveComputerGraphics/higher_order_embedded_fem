use crate::sparse::{CsrMatrixHandle, MatrixDescription, SparseStatusError};
use crate::util::is_same_type;
use crate::SupportedScalar;

use mkl_sys::{mkl_sparse_d_ev, mkl_sparse_d_svd, mkl_sparse_ee_init, MKL_INT};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EigenResult<T> {
    eigenvectors: Vec<T>,
    eigenvalues: Vec<T>,
    residuals: Vec<T>,
}

impl<T> EigenResult<T> {
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &[T] {
        &self.eigenvectors
    }

    pub fn residuals(&self) -> &[T] {
        &self.residuals
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvdResult<T> {
    singular_values: Vec<T>,
    left_vectors: Option<Vec<T>>,
    right_vectors: Option<Vec<T>>,
    residuals: Vec<T>,
}

impl<T> SvdResult<T> {
    pub fn singular_values(&self) -> &[T] {
        &self.singular_values
    }

    pub fn left_vectors(&self) -> Option<&[T]> {
        self.left_vectors.as_ref().map(|v| v.as_slice())
    }

    pub fn right_vectors(&self) -> Option<&[T]> {
        self.right_vectors.as_ref().map(|v| v.as_slice())
    }

    pub fn residuals(&self) -> &[T] {
        &self.residuals
    }
}

/// Decides whether to compute the smallest or largest eigenvalues/singular values.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Which {
    Largest,
    Smallest,
}

impl Which {
    fn integer_representation(&self) -> i8 {
        match self {
            Self::Largest => 'L' as i8,
            Self::Smallest => 'S' as i8,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SingularVectorType {
    Right,
    Left,
}

impl SingularVectorType {
    fn integer_representation(&self) -> i8 {
        match self {
            Self::Right => 'R' as i8,
            Self::Left => 'L' as i8,
        }
    }
}

fn extremal_eigenvalues<T>(
    which: Which,
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar,
{
    let k_in = k as MKL_INT;
    let mut k_out = 0 as MKL_INT;

    if is_same_type::<T, f64>() {
        // TODO: Allow tweaking options
        let mut opts = vec![0 as MKL_INT; 128];
        let code = unsafe { mkl_sparse_ee_init(opts.as_mut_ptr()) };
        SparseStatusError::new_result(code, "mkl_sparse_ee_init")?;

        let mut eigenvalues = vec![T::zero_element(); k];
        let mut eigenvectors = vec![T::zero_element(); k * matrix.cols()];
        let mut residuals = vec![T::zero_element(); k];

        let mut which = which.integer_representation();

        let code = unsafe {
            mkl_sparse_d_ev(
                &mut which,
                opts.as_mut_ptr(),
                matrix.handle,
                description.to_mkl_descr(),
                k_in,
                &mut k_out,
                eigenvalues.as_mut_ptr() as *mut f64,
                eigenvectors.as_mut_ptr() as *mut f64,
                residuals.as_mut_ptr() as *mut f64,
            )
        };
        SparseStatusError::new_result(code, "mkl_sparse_d_ev")?;
        let k_out = k_out as usize;
        eigenvalues.truncate(k_out);
        eigenvectors.truncate(k_out * matrix.cols());
        residuals.truncate(k_out);
        Ok(EigenResult {
            eigenvectors,
            eigenvalues,
            residuals,
        })
    } else {
        panic!("Unsupported type");
    }
}

pub fn sparse_svd<T>(
    which: Which,
    vector_type: SingularVectorType,
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<SvdResult<T>, SparseStatusError>
    where
        T: SupportedScalar,
{
    // TODO: Check if k is not too large?
    let k_in = k as MKL_INT;
    let mut k_out = 0 as MKL_INT;

    if is_same_type::<T, f64>() {
        // TODO: Allow tweaking options
        let mut opts = vec![0 as MKL_INT; 128];
        let code = unsafe { mkl_sparse_ee_init(opts.as_mut_ptr()) };
        SparseStatusError::new_result(code, "mkl_sparse_ee_init")?;

        let mut singular_values = vec![T::zero_element(); k];
        let mut residuals = vec![T::zero_element(); k];
        let mut left_vectors = vec![T::zero_element(); k * matrix.rows()];
        let mut right_vectors = vec![T::zero_element(); k * matrix.cols()];

        let mut which = which.integer_representation();
        let mut int_vector_type = vector_type.integer_representation();

        let code = unsafe {
            mkl_sparse_d_svd(
                &mut which,
                &mut int_vector_type,
                opts.as_mut_ptr(),
                matrix.handle,
                description.to_mkl_descr(),
                k_in,
                &mut k_out,
                singular_values.as_mut_ptr() as *mut f64,
                left_vectors.as_mut_ptr() as *mut f64,
                right_vectors.as_mut_ptr() as *mut f64,
                residuals.as_mut_ptr() as *mut f64,
            )
        };
        SparseStatusError::new_result(code, "mkl_sparse_d_svd")?;
        let k_out = k_out as usize;

        singular_values.truncate(k_out);
        residuals.truncate(k_out);
        let mut result = SvdResult {
            singular_values,
            residuals,
            left_vectors: None,
            right_vectors: None,
        };

        match vector_type {
            SingularVectorType::Left => {
                left_vectors.truncate(k_out * matrix.rows());
                result.left_vectors = Some(left_vectors);
            },
            SingularVectorType::Right => {
                right_vectors.truncate(k_out * matrix.cols());
                result.right_vectors = Some(right_vectors);
            }
        }

        Ok(result)
    } else {
        panic!("Unsupported type");
    }
}

/// Attempts to compute the `k` largest eigenvalues of the given matrix, with the given description.
///
/// Note that the returned number of eigenvalues might be smaller than requested (see MKL
/// docs for details).
pub fn k_largest_eigenvalues<T>(
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar,
{
    extremal_eigenvalues(Which::Largest, matrix, description, k)
}

/// Attempts to compute the `k` smallest eigenvalues of the given matrix, with the given description.
///
/// Note that the returned number of eigenvalues might be smaller than requested (see MKL
/// docs for details).
// TODO: Extend to general sparse matrices, not just CSR
pub fn k_smallest_eigenvalues<T>(
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar,
{
    extremal_eigenvalues(Which::Smallest, matrix, description, k)
}
