use mkl_sys::{MKL_DSS_NON_SYMMETRIC, MKL_DSS_SYMMETRIC, MKL_DSS_SYMMETRIC_STRUCTURE, MKL_INT};

mod solver;
mod sparse_matrix;
pub use solver::*;
pub use sparse_matrix::*;

// TODO: Support complex numbers
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixStructure {
    StructurallySymmetric,
    Symmetric,
    NonSymmetric,
}

impl MatrixStructure {
    fn to_mkl_opt(&self) -> MKL_INT {
        use MatrixStructure::*;
        match self {
            StructurallySymmetric => MKL_DSS_SYMMETRIC_STRUCTURE,
            Symmetric => MKL_DSS_SYMMETRIC,
            NonSymmetric => MKL_DSS_NON_SYMMETRIC,
        }
    }
}
