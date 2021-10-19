mod poisson_common;
use poisson_common::*;

use fenris::assembly::{
    apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_rhs, assemble_generalized_stiffness,
    assemble_source_term_into,
};
use fenris::element::ElementConnectivity;
use fenris::error::estimate_element_L2_error_squared;
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;
use fenris::geometry::vtk::write_vtk;
use fenris::mesh::Hex20Mesh;
use fenris::quadrature::hex_quadrature_strength_5;
use fenris::vtkio::model::Attribute;
use fenris::vtkio::IOBuffer;
use mkl_corrode::dss;
use mkl_corrode::dss::Definiteness;
use mkl_corrode::dss::MatrixStructure::Symmetric;
use nalgebra::storage::Storage;
use nalgebra::{DVector, DVectorSlice, DVectorSliceMut, Vector1, Vector3, U1};
use std::error::Error;
use std::ops::Add;
use vtkio::model::DataSet;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    use std::f64::consts::PI;
    let sin = |x| f64::sin(x);

    let u_exact_xyz = |x, y, z| sin(PI * x) * sin(PI * y) * sin(PI * z);
    let u_exact = |x: &Vector3<f64>| -> Vector1<f64> { Vector1::new(u_exact_xyz(x.x, x.y, x.z)) };
    let f_xyz = { |x, y, z| 3.0 * PI * PI * u_exact_xyz(x, y, z) };
    let f = |x: &Vector3<f64>| Vector1::new(f_xyz(x.x, x.y, x.z));
    let resolutions = vec![1, 2, 4, 8, 16];

    for res in resolutions {
        let mesh = create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, res);
        // let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate()?)?;
        // let mesh = Tet20Mesh::from(&mesh);
        // let mesh = Tet10Mesh::from(&mesh);
        let mesh = Hex20Mesh::from(&mesh);
        let quadrature = hex_quadrature_strength_5();
        // let quadrature = tet_quadrature_strength_5();
        let ndof = mesh.vertices().len();

        // Assemble system matrix and right hand side
        let qtable = |_| &quadrature;
        let u = DVector::zeros(ndof);
        let operator = PoissonEllipticOperator;
        let mut A = assemble_generalized_stiffness(mesh.vertices(), mesh.connectivity(), &operator, &u, &qtable)
            .to_csr(Add::add);
        let boundary_vertices: Vec<_> = mesh
            .vertices()
            .iter()
            .enumerate()
            .filter(|(_, v)| (v.coords - Vector3::new(0.5, 0.5, 0.5)).amax() >= 0.499)
            .map(|(i, _)| i)
            .collect();

        let mut rhs = DVector::zeros(ndof);
        assemble_source_term_into(DVectorSliceMut::from(&mut rhs), &mesh, &f, &qtable);

        apply_homogeneous_dirichlet_bc_csr::<_, U1>(&mut A, &boundary_vertices);
        apply_homogeneous_dirichlet_bc_rhs(&mut rhs, &boundary_vertices, 1);

        let A_dss =
            dss::SparseMatrix::try_convert_from_csr(A.row_offsets(), A.column_indices(), A.values(), Symmetric)?;
        let options = dss::SolverOptions::default().parallel_reorder(true);
        let mut solver = dss::Solver::try_factor_with_opts(&A_dss, Definiteness::PositiveDefinite, &options)?;
        let u_solution = solver.solve(rhs.data.as_slice()).unwrap();

        {
            let mut dataset = DataSet::from(&mesh);
            if let DataSet::UnstructuredGrid { ref mut data, .. } = dataset {
                let u_buffer = IOBuffer::from_slice(u_solution.as_slice());
                let attribute = Attribute::Scalars {
                    num_comp: 1,
                    lookup_table: None,
                    data: u_buffer,
                };
                data.point.push((format!("u"), attribute));
            } else {
                panic!("Unexpected data");
            }

            write_vtk(dataset, format!("poisson_sol_{}.vtk", res), "Poisson solution")?;
        }

        // Write nodal interpolated data
        {
            let mut dataset = DataSet::from(&mesh);
            let u_nodal_interpolation: Vec<_> = mesh
                .vertices()
                .iter()
                .map(|v| u_exact(&v.coords).x)
                .collect();
            if let DataSet::UnstructuredGrid { ref mut data, .. } = dataset {
                let u_buffer = IOBuffer::from_slice(u_nodal_interpolation.as_slice());
                let attribute = Attribute::Scalars {
                    num_comp: 1,
                    lookup_table: None,
                    data: u_buffer,
                };
                data.point.push((format!("u"), attribute));
            } else {
                panic!("Unexpected data");
            }

            write_vtk(
                dataset,
                format!("poisson_nodal_interpolation_{}.vtk", res),
                "Poisson solution",
            )?;
        }

        let l2_error_squared = mesh
            .connectivity()
            .iter()
            .map(|conn| {
                let u_weights =
                    conn.element_variables(DVectorSlice::from_slice(u_solution.as_slice(), u_solution.len()));
                let element = conn.element(mesh.vertices()).unwrap();
                estimate_element_L2_error_squared(&element, |x, _| u_exact(&x.coords), &u_weights, &quadrature)
            })
            .sum::<f64>();
        let l2_error = l2_error_squared.sqrt();

        println!("L2 error: {:3.3e}", l2_error);
    }

    Ok(())
}
