pub mod allocators;
pub mod assembly;
pub mod cg;
pub mod connectivity;
pub mod element;
pub mod embedding;
pub mod error;
pub mod geometry;
pub mod lp_solvers;
pub mod model;
pub mod quadrature;
pub mod reorder;
pub mod rtree;
pub mod solid;
pub mod space;
pub mod sparse;
pub mod util;

#[cfg(feature = "proptest")]
pub mod proptest;

// TODO: Don't export
pub use sparse::CooMatrix;
pub use sparse::CsrMatrix;

pub mod mesh;

mod mesh_convert;
mod space_impl;

pub extern crate nalgebra;
pub extern crate nested_vec;
pub extern crate vtkio;
