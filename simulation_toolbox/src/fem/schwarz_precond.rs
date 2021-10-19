use fenris::assembly::{CsrParAssembler, ElementAssembler, ElementConnectivityAssembler};
use fenris::connectivity::{CellConnectivity, Connectivity};
use fenris::nalgebra::{DMatrix, DMatrixSliceMut, DefaultAllocator, U3};
use fenris::sparse::SparsityPattern;
use fenris::{CooMatrix, CsrMatrix};
use std::error::Error;
use std::iter::once;
use std::ops::Add;
use std::sync::Arc;

use coarse_prof::profile;
use fenris::allocators::ElementConnectivityAllocator;
use fenris::element::{ElementConnectivity, FiniteElement};
use fenris::embedding::EmbeddedModel3d;
use fenris::geometry::ConvexPolyhedron;
use fenris::nested_vec::NestedVec;
use fenris::quadrature::Quadrature;
use fenris::space::FiniteElementSpace;
use hamilton::storages::VecStorage;
use hamilton::Component;
use log::info;
use paradis::coloring::sequential_greedy_coloring;
use paradis::DisjointSubsets;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SchwarzPreconditionerComponent {
    // TODO: Better names
    /// Node-connectivities for elements that will be treated with the preconditioner
    pub schwarz_connectivity: NestedVec<usize>,
    /// Sorted list of indices of nodes that are not a member of preconditioned
    /// elements. These nodes will be treated with a (possibly block-) Jacobi-type
    /// preconditioner.
    pub untreated_nodes: Vec<usize>,

    #[serde(skip)]
    pub preconditioner: Option<CsrMatrix<f64>>,

    /// Colors must match schwarz_connectivity
    #[serde(skip)]
    pub colors: Option<Vec<DisjointSubsets>>,
}

impl SchwarzPreconditionerComponent {
    pub fn from_space<S>(space: &S) -> Self
    where
        S: FiniteElementSpace<f64>,
        DefaultAllocator: ElementConnectivityAllocator<f64, S::Connectivity>,
    {
        let mut schwarz_connectivity = NestedVec::new();
        for i in 0..space.num_connectivities() {
            let conn = space.get_connectivity(i).unwrap();
            schwarz_connectivity.push(conn.vertex_indices());
        }
        let untreated_nodes = Vec::new();
        let colors = sequential_greedy_coloring(&schwarz_connectivity);
        Self {
            schwarz_connectivity,
            untreated_nodes,
            preconditioner: None,
            colors: Some(colors),
        }
    }

    pub fn from_embedded_model<C>(model: &EmbeddedModel3d<f64, C>, volume_fraction_threshold: f64) -> Self
    where
        C: Connectivity + CellConnectivity<f64, U3>,
        C::Cell: for<'a> ConvexPolyhedron<'a, f64>,
        C: ElementConnectivity<f64, GeometryDim = U3, ReferenceDim = U3>,
        DefaultAllocator: ElementConnectivityAllocator<f64, C>,
    {
        let mut schwarz_connectivity = NestedVec::new();
        for (index, interface_conn) in model.interface_connectivity().iter().enumerate() {
            let bg_cell_volume = interface_conn
                .cell(model.vertices())
                .unwrap()
                .compute_volume();
            // TODO: Don't rely on quadrature for volume? Currently we use the
            // mass quadrature because it's likely to be highest order,
            // however since basically any correct quadrature should have the sum of the weights
            // be equal to the volume, we could also use something else
            let quadrature = &model
                .mass_quadrature()
                .expect(
                    "TODO: Handle missing mass quadrature, or better yet use a different \
                         way to obtain embedded volumes",
                )
                .interface_quadratures()[index];

            let element = interface_conn.element(model.vertices()).unwrap();
            let embedded_volume = quadrature.integrate(|xi| element.reference_jacobian(xi).determinant());

            let volume_fraction = embedded_volume / bg_cell_volume;
            if volume_fraction <= volume_fraction_threshold {
                schwarz_connectivity.push(interface_conn.vertex_indices());
            }
        }

        // Tag nodes that belong to interface connectivity
        let mut is_interface_node = vec![false; model.vertices().len()];
        for node_idx in schwarz_connectivity.iter_array_elements() {
            is_interface_node[*node_idx] = true;
        }

        let untreated_nodes: Vec<usize> = is_interface_node
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_interface)| if is_interface { None } else { Some(idx) })
            .collect();

        let total_num_elements = model.background_mesh().connectivity().len();
        let proportion = schwarz_connectivity.len() as f64 / total_num_elements as f64;
        info!(
            "Preconditioning {} out of {} elements ({:2.1} %) with Schwarz preconditioner.",
            schwarz_connectivity.len(),
            total_num_elements,
            100.0 * proportion
        );

        let colors = sequential_greedy_coloring(&schwarz_connectivity);
        Self {
            schwarz_connectivity,
            untreated_nodes,
            preconditioner: None,
            colors: Some(colors),
        }
    }
}

impl Component for SchwarzPreconditionerComponent {
    type Storage = VecStorage<Self>;
}

pub fn build_preconditioner_pattern(
    num_nodes: usize,
    schwarz_connectivity: &NestedVec<usize>,
    untreated_nodes: &[usize],
    solution_dim: usize,
) -> Arc<SparsityPattern> {
    // TODO: This can be done *much* more efficiently
    let n = solution_dim * num_nodes;
    let mut coo = CooMatrix::new(n, n);

    for node_indices in schwarz_connectivity.iter() {
        for node_i in node_indices {
            for node_j in node_indices {
                for i in 0..solution_dim {
                    for j in 0..solution_dim {
                        coo.push(solution_dim * *node_i + i, solution_dim * *node_j + j, 0.0);
                    }
                }
            }
        }
    }

    // Add block diagonal entries for untreated nodes
    for &node_idx in untreated_nodes {
        for i in 0..solution_dim {
            coo.push(solution_dim * node_idx + i, solution_dim * node_idx + i, 0.0);
            // Below was for old block diagonal preconditioning
            // for j in 0 .. solution_dim {
            //     coo.push(solution_dim * node_idx + i, solution_dim * node_idx + j, 0.0);
            // }
        }
    }

    let csr = coo.to_csr(Add::add);
    csr.sparsity_pattern().clone()
}

pub fn build_schwarz_preconditioner_into(
    preconditioner: &mut CsrMatrix<f64>,
    system_matrix: &CsrMatrix<f64>,
    schwarz_connectivity: &NestedVec<usize>,
    untreated_nodes: &[usize],
    colors: &[DisjointSubsets],
    solution_dim: usize,
) -> Result<(), Box<dyn Error>> {
    let mut csr_assembler = CsrParAssembler::default();
    let element_assembler = SchwarzElementAssembler {
        matrix: system_matrix,
        schwarz_connectivity,
        solution_dim,
    };

    {
        profile!("assemble");
        csr_assembler
            .assemble_into_csr(preconditioner, &colors, &element_assembler)
            .map_err(|err| err as Box<dyn Error>)?;
    }

    // Add remaining diagonal entries
    for &node_idx in untreated_nodes {
        for i in 0..solution_dim {
            let row_index = solution_dim * node_idx + i;
            let mut precond_row = preconditioner.row_mut(row_index);
            assert_eq!(precond_row.nnz(), 1);
            assert_eq!(precond_row.col_at_local_index(0), row_index);

            let matrix_row = system_matrix.row(row_index);
            let entry = matrix_row.get(row_index).unwrap_or(0.0);

            if entry != 0.0 {
                precond_row.values_mut()[0] = entry.recip();
            } else {
                return Err(Box::<dyn Error>::from(
                    "Can not construct Schwarz preconditioner: \
                    Diagonal entry is zero",
                ));
            }
        }
    }

    Ok(())
}

pub fn build_schwarz_preconditioner(
    matrix: &CsrMatrix<f64>,
    schwarz_connectivity: &NestedVec<usize>,
    untreated_nodes: &[usize],
    solution_dim: usize,
) -> Result<CsrMatrix<f64>, Box<dyn Error>> {
    profile!("schwarz preconditioner construction");
    assert_eq!(matrix.nrows(), matrix.ncols());
    assert_eq!(matrix.nrows() % solution_dim, 0);
    let num_nodes = matrix.nrows() / solution_dim;

    let pattern = {
        profile!("build pattern");
        build_preconditioner_pattern(num_nodes, schwarz_connectivity, untreated_nodes, solution_dim)
    };

    let nnz = pattern.nnz();
    let mut result = CsrMatrix::from_pattern_and_values(pattern, vec![0.0; nnz]);

    let colors = sequential_greedy_coloring(schwarz_connectivity);

    build_schwarz_preconditioner_into(
        &mut result,
        matrix,
        schwarz_connectivity,
        untreated_nodes,
        &colors,
        solution_dim,
    )?;
    Ok(result)
}

struct SchwarzElementAssembler<'a> {
    matrix: &'a CsrMatrix<f64>,
    solution_dim: usize,
    schwarz_connectivity: &'a NestedVec<usize>,
}

impl<'a> ElementConnectivityAssembler for SchwarzElementAssembler<'a> {
    fn num_nodes(&self) -> usize {
        assert_eq!(self.matrix.nrows() % self.solution_dim, 0);
        self.matrix.nrows() / self.solution_dim()
    }

    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_elements(&self) -> usize {
        self.schwarz_connectivity.len()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.schwarz_connectivity
            .get(element_index)
            .expect("Element index is assumed to be in bounds")
            .len()
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        let indices = self
            .schwarz_connectivity
            .get(element_index)
            .expect("Element index is assumed to be in bounds");
        assert_eq!(output.len(), indices.len());
        output.copy_from_slice(indices);
    }
}

impl<'a> ElementAssembler<f64> for SchwarzElementAssembler<'a> {
    fn assemble_element_matrix_into(
        &self,
        mut output: DMatrixSliceMut<f64>,
        element_index: usize,
    ) -> Result<(), Box<dyn Send + Error>> {
        let node_indices = self
            .schwarz_connectivity
            .get(element_index)
            .expect("Element index is assumed to be in bounds");
        let sdim = self.solution_dim;

        let element_matrix_dim = sdim * node_indices.len();

        let mut element_block = DMatrix::zeros(element_matrix_dim, element_matrix_dim);

        // TODO: Write this is as a "gather" routine in CSR?
        for (local_node_i, global_node_i) in node_indices.iter().enumerate() {
            for i in 0..sdim {
                let local_row = sdim * local_node_i + i;
                let global_row = sdim * *global_node_i + i;

                let csr_row = self.matrix.row(global_row);
                let mut element_block_row = element_block.row_mut(local_row);

                for (local_node_j, global_node_j) in node_indices.iter().enumerate() {
                    for j in 0..sdim {
                        let local_col = sdim * local_node_j + j;
                        let global_col = sdim * *global_node_j + j;
                        element_block_row[local_col] = csr_row.get(global_col).expect("TODO: Error handling?");
                    }
                }
            }
        }

        let element_block_inverse = invert_element_block(element_block);
        output.copy_from(&element_block_inverse);

        Ok(())
    }
}

fn invert_element_block(element_block_matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut block_eigen = nalgebra_lapack::SymmetricEigen::new(element_block_matrix);
    let max_eval = block_eigen.eigenvalues.amax();
    let lower_threshold = 1e-12 * max_eval;

    for eval in &mut block_eigen.eigenvalues {
        *eval =
            // Note: We flip the sign of negative eigenvalues so as to produce
            // a positive definite preconditioner.
            if eval.abs() >= lower_threshold { 1.0 / eval.abs() }
            else { 1.0 / (lower_threshold) };
    }
    block_eigen.recompose()
}

pub fn element_wise_dense_additive_schwarz_preconditioner<C>(matrix: &DMatrix<f64>, connectivity: &[C]) -> DMatrix<f64>
where
    C: Connectivity,
{
    let mut result = DMatrix::zeros(matrix.nrows(), matrix.ncols());
    for conn in connectivity {
        // Gather the element "patch matrix"
        let indices: Vec<_> = conn
            .vertex_indices()
            .iter()
            // TODO: I hate this: Fix?
            .flat_map(|v_idx| {
                once(3 * v_idx)
                    .chain(once(3 * v_idx + 1))
                    .chain(once(3 * v_idx + 2))
            })
            .collect();
        assert_eq!(indices.len(), 3 * conn.vertex_indices().len());
        let patch_matrix = matrix.select_rows(&indices).select_columns(&indices);
        let patch_inverse = invert_element_block(patch_matrix);

        for (c_local, &c_global) in indices.iter().enumerate() {
            for (r_local, &r_global) in indices.iter().enumerate() {
                result[(r_global, c_global)] += patch_inverse[(r_local, c_local)];
            }
        }
    }
    result
}
