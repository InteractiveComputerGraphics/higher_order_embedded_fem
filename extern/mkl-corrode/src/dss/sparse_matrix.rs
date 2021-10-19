use crate::dss::MatrixStructure;
use crate::SupportedScalar;

use mkl_sys::MKL_INT;

use core::fmt;
use std::borrow::Cow;
use std::convert::TryFrom;
use std::fmt::{Debug, Display};

use crate::util::{is_same_type, transmute_identical_slice};

// TODO: We only care about square matrices
#[derive(Debug, PartialEq, Eq)]
pub struct SparseMatrix<'a, T>
where
    T: Clone,
{
    row_offsets: Cow<'a, [MKL_INT]>,
    columns: Cow<'a, [MKL_INT]>,
    values: Cow<'a, [T]>,
    structure: MatrixStructure,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseMatrixDataError {
    NonMonotoneColumns,
    MissingExplicitDiagonal,
    UnexpectedLowerTriangularPart,
    NonMonotoneRowOffsets,
    EmptyRowOffsets,
    InvalidRowOffset,
    InvalidColumnIndex,
    InsufficientIndexSize,
}

impl SparseMatrixDataError {
    fn is_recoverable(&self) -> bool {
        use SparseMatrixDataError::*;
        match self {
            NonMonotoneColumns => false,
            MissingExplicitDiagonal => true,
            UnexpectedLowerTriangularPart => true,
            NonMonotoneRowOffsets => false,
            EmptyRowOffsets => false,
            InvalidRowOffset => false,
            InvalidColumnIndex => false,
            InsufficientIndexSize => false,
        }
    }
}

impl Display for SparseMatrixDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Error in sparse matrix data: {:?}", self)
    }
}

impl std::error::Error for SparseMatrixDataError {}

trait CsrProcessor<T> {
    /// Called when processing of the current row has finished.
    fn row_processed(&mut self) {}
    fn visit_column(&mut self, i: MKL_INT, j: MKL_INT, v: &T) -> Result<(), SparseMatrixDataError>;
    fn visit_missing_diagonal_entry(&mut self, i: MKL_INT) -> Result<(), SparseMatrixDataError>;
}

fn process_csr<'a, T, I>(
    row_offsets: &'a [I],
    columns: &'a [I],
    values: &'a [T],
    structure: MatrixStructure,
    processor: &mut impl CsrProcessor<T>,
) -> Result<(), SparseMatrixDataError>
where
    T: SupportedScalar,
    usize: TryFrom<I>,
    MKL_INT: TryFrom<I>,
    I: Copy,
{
    let needs_explicit_diagonal = match structure {
        MatrixStructure::Symmetric | MatrixStructure::StructurallySymmetric => false,
        MatrixStructure::NonSymmetric => true,
    };

    // Helper conversion functions.
    let offset_as_usize =
        |offset| usize::try_from(offset).map_err(|_| SparseMatrixDataError::InvalidRowOffset);
    let index_as_mkl_int =
        |idx| MKL_INT::try_from(idx).map_err(|_| SparseMatrixDataError::InvalidColumnIndex);
    let usize_as_mkl_int = |idx| {
        <MKL_INT as TryFrom<usize>>::try_from(idx)
            .map_err(|_| SparseMatrixDataError::InsufficientIndexSize)
    };

    let num_rows = row_offsets.len() - 1;
    let num_cols = usize_as_mkl_int(num_rows)?;
    let nnz = values.len();
    // TODO: Assertion or error?
    assert_eq!(nnz, columns.len());

    if row_offsets.is_empty() {
        return Err(SparseMatrixDataError::EmptyRowOffsets);
    }

    if nnz != offset_as_usize(*row_offsets.last().unwrap())? {
        return Err(SparseMatrixDataError::InvalidRowOffset);
    }

    for i in 0..num_rows {
        let current_offset = row_offsets[i];
        let row_begin = offset_as_usize(current_offset)?;
        let row_end = offset_as_usize(row_offsets[i + 1])?;
        let i = usize_as_mkl_int(i)?;

        if row_end < row_begin {
            return Err(SparseMatrixDataError::NonMonotoneRowOffsets);
        }

        // - check that each column is in bounds, if not abort
        // - check that column indices are monotone increasing, if not abort
        // - If (structurally) symmetric: check that the diagonal element exists, if not insert it
        // - If (structurally) symmetric: ignore lower triangular elements

        let columns_for_row = &columns[row_begin..row_end];
        let values_for_row = &values[row_begin..row_end];

        // TODO: Rename to "have_processed"
        let mut have_placed_diagonal = false;
        let mut prev_column = None;
        for (j, v_j) in columns_for_row.iter().zip(values_for_row) {
            let j = index_as_mkl_int(*j)?;

            if j < 0 || j >= num_cols {
                return Err(SparseMatrixDataError::InvalidColumnIndex);
            }

            if let Some(j_prev) = prev_column {
                if j <= j_prev {
                    return Err(SparseMatrixDataError::NonMonotoneColumns);
                }
            }

            if needs_explicit_diagonal {
                if i == j {
                    have_placed_diagonal = true;
                // TODO: Can remove the i < j comparison here!
                } else if i < j && !have_placed_diagonal {
                    processor.visit_missing_diagonal_entry(i)?;
                    have_placed_diagonal = true;
                }
            }

            processor.visit_column(i, j, v_j)?;
            prev_column = Some(j);
        }
        processor.row_processed();
    }
    Ok(())
}

fn rebuild_csr<'a, T, I>(
    row_offsets: &'a [I],
    columns: &'a [I],
    values: &'a [T],
    structure: MatrixStructure,
) -> Result<SparseMatrix<'a, T>, SparseMatrixDataError>
where
    T: SupportedScalar,
    usize: TryFrom<I>,
    MKL_INT: TryFrom<I>,
    I: Copy,
{
    let keep_lower_tri = match structure {
        MatrixStructure::Symmetric | MatrixStructure::StructurallySymmetric => false,
        MatrixStructure::NonSymmetric => true,
    };

    struct CsrRebuilder<X> {
        new_row_offsets: Vec<MKL_INT>,
        new_columns: Vec<MKL_INT>,
        new_values: Vec<X>,
        current_offset: MKL_INT,
        num_cols_in_current_row: MKL_INT,
        keep_lower_tri: bool,
    }

    impl<X> CsrRebuilder<X> {
        fn push_val(&mut self, j: MKL_INT, v_j: X) {
            self.new_columns.push(j);
            self.new_values.push(v_j);
            self.num_cols_in_current_row += 1;
        }
    }

    impl<X: SupportedScalar> CsrProcessor<X> for CsrRebuilder<X> {
        fn row_processed(&mut self) {
            let new_offset = self.current_offset + self.num_cols_in_current_row;
            self.current_offset = new_offset;
            self.num_cols_in_current_row = 0;
            self.new_row_offsets.push(new_offset);
        }

        fn visit_column(&mut self, i: MKL_INT, j: MKL_INT, v_j: &X) -> Result<(), SparseMatrixDataError> {
            let should_push = j >= i || (j < i && self.keep_lower_tri);
            if should_push {
                self.push_val(j, *v_j);
            }
            Ok(())
        }

        fn visit_missing_diagonal_entry(&mut self, i: MKL_INT) -> Result<(), SparseMatrixDataError> {
            self.push_val(i, X::zero_element());
            Ok(())
        }
    }

    let mut rebuilder = CsrRebuilder {
        new_row_offsets: vec![0],
        new_columns: Vec::new(),
        new_values: Vec::new(),
        current_offset: 0,
        num_cols_in_current_row: 0,
        keep_lower_tri,
    };

    process_csr(row_offsets, columns, values, structure, &mut rebuilder)?;

    let matrix = SparseMatrix {
        row_offsets: Cow::Owned(rebuilder.new_row_offsets),
        columns: Cow::Owned(rebuilder.new_columns),
        values: Cow::Owned(rebuilder.new_values),
        structure,
    };
    Ok(matrix)
}

impl<'a, T> SparseMatrix<'a, T>
where
    T: SupportedScalar,
{
    pub fn row_offsets(&self) -> &[MKL_INT] {
        &self.row_offsets
    }

    pub fn columns(&self) -> &[MKL_INT] {
        &self.columns
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn structure(&self) -> MatrixStructure {
        self.structure
    }

    pub fn try_from_csr(
        row_offsets: &'a [MKL_INT],
        columns: &'a [MKL_INT],
        values: &'a [T],
        structure: MatrixStructure,
    ) -> Result<Self, SparseMatrixDataError> {
        let allow_lower_tri = match structure {
            MatrixStructure::Symmetric | MatrixStructure::StructurallySymmetric => false,
            MatrixStructure::NonSymmetric => true,
        };

        struct CsrCheck {
            allow_lower_tri: bool,
        }

        impl<X: SupportedScalar> CsrProcessor<X> for CsrCheck {
            fn visit_column(&mut self, i: MKL_INT, j: MKL_INT, _: &X) -> Result<(), SparseMatrixDataError> {
                if !self.allow_lower_tri && j < i {
                    Err(SparseMatrixDataError::UnexpectedLowerTriangularPart)
                } else {
                    Ok(())
                }
            }

            fn visit_missing_diagonal_entry(
                &mut self,
                _: MKL_INT,
            ) -> Result<(), SparseMatrixDataError> {
                Err(SparseMatrixDataError::MissingExplicitDiagonal)
            }
        }

        let mut checker = CsrCheck { allow_lower_tri };
        process_csr(row_offsets, columns, values, structure, &mut checker)?;

        let matrix = SparseMatrix {
            row_offsets: Cow::Borrowed(row_offsets),
            columns: Cow::Borrowed(columns),
            values: Cow::Borrowed(values),
            structure,
        };
        Ok(matrix)
    }

    pub fn try_convert_from_csr<I>(
        row_offsets: &'a [I],
        columns: &'a [I],
        values: &'a [T],
        structure: MatrixStructure,
    ) -> Result<Self, SparseMatrixDataError>
    where
        I: 'static + Copy,
        MKL_INT: TryFrom<I>,
        usize: TryFrom<I>,
    {
        // If the data already has the right integer type, then try to pass it in to MKL directly.
        // If it fails, it might be that we can recover by rebuilding the matrix data.
        if is_same_type::<I, MKL_INT>() {
            let row_offsets_mkl_int = transmute_identical_slice(row_offsets).unwrap();
            let columns_mkl_int = transmute_identical_slice(columns).unwrap();
            let result =
                Self::try_from_csr(row_offsets_mkl_int, columns_mkl_int, values, structure);
            match result {
                Ok(matrix) => return Ok(matrix),
                Err(error) => {
                    if !error.is_recoverable() {
                        return Err(error);
                    }
                }
            }
        };

        rebuild_csr(row_offsets, columns, values, structure)
    }
}
