use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use fenris::allocators::{ElementConnectivityAllocator, FiniteElementMatrixAllocator, VolumeFiniteElementAllocator};
use fenris::assembly::{
    apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_rhs, assemble_source_term_into, QuadratureTable,
};
use fenris::connectivity::{
    CellConnectivity, Connectivity, Hex20Connectivity, Hex27Connectivity, Hex8Connectivity, Tet20Connectivity,
};
use fenris::element::{map_physical_coordinates, ConnectivityNodalDim, ElementConnectivity, ReferenceFiniteElement};
use fenris::embedding::{
    embed_mesh_3d, embed_mesh_3d_with_opts, embed_quadrature_3d, embed_quadrature_3d_with_opts, EmbedOptions,
    EmbeddedModel3d, EmbeddedModelBuilder, Embedding, QuadratureOptions,
};
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;
use fenris::geometry::vtk::{write_vtk, VtkCellConnectivity};
use fenris::mesh::{Hex20Mesh, Hex27Mesh, HexMesh, Mesh, Tet20Mesh, Tet4Mesh};
use fenris::model::{FiniteElementInterpolator, MakeInterpolator, NodalModel, NodalModel3d};
use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::storage::Storage;
use fenris::nalgebra::{
    DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName, DimNameMul, MatrixMN, Point, Point3, RealField,
    Vector3, U10, U3,
};
use fenris::quadrature::{
    hex_quadrature_strength_11, hex_quadrature_strength_3, hex_quadrature_strength_5, tet_quadrature_strength_10,
    tet_quadrature_strength_3, tet_quadrature_strength_5, QuadraturePair3d,
};
use fenris::rtree::GeometryCollectionAccelerator;
use fenris::solid::materials::{LinearElasticMaterial, YoungPoisson};
use fenris::solid::{ElasticMaterialModel, ElasticityModel, ElasticityModelParallel};
use fenris::space::{FiniteElementSpace, GeometricFiniteElementSpace};
use fenris::util::flatten_vertically;
use fenris::vtkio::model::{Attribute, DataSet};
use fenris::vtkio::IOBuffer;
use hamilton2::calculus::{DifferentiableVectorFunction, VectorFunction};
use hamilton2::newton::{newton_line_search, BacktrackingLineSearch, NewtonSettings};
use mkl_corrode::dss;
use mkl_corrode::dss::Definiteness;
use mkl_corrode::dss::MatrixStructure::Symmetric;
use simulation_toolbox::io::msh::{try_mesh_from_bytes, TryConnectivityFromMshElement, TryVertexFromMshNode};
use std::ops::Add;
use structopt::StructOpt;

use rayon::prelude::*;

use fenris::cg::{ConjugateGradient, LinearOperator, RelativeResidualCriterion};
use fenris::error::estimate_element_L2_error_squared;
use fenris::geometry::{BoundedGeometry, ConvexPolyhedron, Distance, DistanceQuery};
use fenris::lp_solvers::GlopSolver;
use fenris::CsrMatrix;
use mkl_corrode::sparse::{CsrMatrixHandle, MatrixDescription, SparseOperation};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::convert::TryFrom;

/// Timestamped println
macro_rules! tprintln {
    ($($arg:tt)*) => {
        {
            use chrono::offset::Local;
            let print_str = format!($($arg)*);
            let now = Local::now();
            println!("[{}] {}",
                now.format("%H:%M:%S"),
                print_str);
        }
    }
}

#[derive(Debug, StructOpt)]
struct CommandlineArgs {
    #[structopt(short = "-r", long = "--resolutions", help = "Resolution")]
    resolutions: Option<Vec<usize>>,

    #[structopt(long = "--reference-mesh", help = "Reference mesh")]
    reference_mesh: Option<String>,

    #[structopt(
        long,
        default_value = "data",
        parse(from_os_str),
        help = "Base directory for output files"
    )]
    output_dir: PathBuf,
    #[structopt(
        long,
        default_value = "assets",
        parse(from_os_str),
        help = "Base directory for asset files"
    )]
    asset_dir: PathBuf,
    #[structopt(
        long,
        parse(from_os_str),
        help = "Path for the logfile relative to 'output-dir/scene-name'"
    )]
    log_file: Option<PathBuf>,
}

fn load_mesh<T, D, C>(asset_dir: impl AsRef<Path>, filename: &str) -> Result<Mesh<T, D, C>, Box<dyn Error>>
where
    T: RealField,
    D: DimName,
    C: Connectivity + TryConnectivityFromMshElement<C>,
    Point<T, D>: TryVertexFromMshNode<T, D, f64>,
    DefaultAllocator: Allocator<T, D>,
{
    let msh_bytes = {
        let file_path = asset_dir.as_ref().join("fcm_convergence/").join(filename);
        let mut file = File::open(file_path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        bytes
    };
    try_mesh_from_bytes(&msh_bytes)
}

#[allow(dead_code)]
fn bump_function(r: f64) -> f64 {
    if r.abs() < 1.0 {
        (-(1.0 / (1.0 - r * r))).exp()
    } else {
        0.0
    }
}

fn body_force(x: &Point3<f64>) -> Vector3<f64> {
    // let x0 = Point3::new(0.0, 1.0, 0.0);
    // let d = x - x0;
    // let r = (d.x * d.x + d.y * d.y).sqrt();
    // let r = (x - x0).magnitude();
    let scale = 500000.0;
    // let magnitude = scale * bump_function(3.0 * r);
    // let direction = - Vector3::y(); //(x - x0) / (r + 1e-12);

    // TODO: Temporarily using this simpler body force. Should switch back later!
    // - 5000.0 * Vector3::new(-0.0, -9.81, 0.0)
    // -5000.0 * Vector3::new(0.0, 10.0 * _x.y, 0.0)
    // -5000.0 * Vector3::new(0.0, 10.0 * x.y, 0.0)

    // Note: At one point this configuration seems to have worked reasonably well
    // let r = (x - x0).magnitude();
    // let scale = 50000.0;
    // let magnitude = scale * bump_function(3.0 * r);
    // let direction = -Vector3::y(); //(x - x0) / (r + 1e-12);

    // magnitude * direction
    // use std::f64;
    // use std::f64::consts::PI;
    // let cos = |x| f64::cos(x);
    let y = x.y;
    let z = x.z;
    let x = x.x;
    let r = (x * x + z * z).sqrt();
    let magnitude = scale * bump_function(3.0 * r) * y;
    let direction = -Vector3::y();
    magnitude * direction
}

// TODO: Remove u_exact and def_grad_exact once we're fully commited to the new approach
// (i.e. not MMS)
// fn u_exact(x: &Point3<f64>) -> Vector3<f64> {
//     let cos = |x| f64::cos(x);
//     let z = x.z;
//     let y = x.y;
//     let x = x.x;
//     (1.0 - cos(y)) * Vector3::new(0.0, cos(2.0 * PI*x)*cos(2.0 * PI*z) - y, 0.0)
// }
//
// fn deformation_gradient_exact(x: &Point3<f64>) -> Matrix3<f64> {
//     // Construct individual columns of du/dx (du/dx, du/dy, du/dz)
//     // let du_dx = Vector3::new(0.0, 0.0, 0.0);
//     // let du_dy = x.y.sin() * Vector3::new(f64::cos(2.0 * PI * x.z), 1.0, 1.0);
//     // let du_dz = (1.0 - x.y.cos()) * Vector3::new(-2.0 * PI * f64::sin(2.0 * PI * x.z), 0.0, 0.0);
//
//     // let u_jacobian = Matrix3::from_columns(&[du_dx, du_dy, du_dz]);
//
//     // TODO: Find exact deformation gradient. The above is old
//
//     // Note: We keep this in case we want to experiment with
//     // other deformations without having to analytically re-evaluate the derivatives
//     let mut point_vec = DVector::zeros(3);
//     point_vec.copy_from(&x.coords);
//     let mut vector_function = VectorFunctionBuilder::with_dimension(3)
//         .with_function(|u, x| u.copy_from(&u_exact(&Point3::from(Vector3::from_column_slice(x.as_slice())))));
//     let u_jacobian = approximate_jacobian(&mut vector_function, &point_vec, &1e-6);
//
//     let f = Matrix3::identity() + u_jacobian;
//     f
// }

fn create_dataset_with_displacements<C>(mesh: &Mesh<f64, U3, C>, displacements: &[Vector3<f64>]) -> DataSet
where
    C: VtkCellConnectivity,
{
    let u_vector = flatten_vertically(&displacements).unwrap_or(DVector::zeros(0));
    let mut dataset = DataSet::from(mesh);
    if let DataSet::UnstructuredGrid { ref mut data, .. } = dataset {
        let displacement_buffer = IOBuffer::from_slice(u_vector.as_slice());
        let attribute = Attribute::Vectors {
            data: displacement_buffer,
        };
        data.point.push((format!("displacement"), attribute));
    } else {
        panic!("Unexpected data");
    }
    dataset
}

// TODO: Remove
// fn save_exact_solution(mesh: &Tet4Mesh<f64>, args: &CommandlineArgs)
//     -> Result<(), Box<dyn Error>> {
//     let displacements: Vec<_> = mesh.vertices()
//         .iter()
//         .map(u_exact)
//         .collect();
//     let dataset = create_dataset_with_displacements(&mesh, &displacements);
//     let vtk_output_file = args.output_dir.join("fcm_convergence/").join("hemisphere.vtk");
//     write_vtk(dataset, &vtk_output_file, "FCM convergence")?;
//     Ok(())
// }

fn dump_embedded_solution(
    embedded_mesh: &Tet4Mesh<f64>,
    interpolator: &FiniteElementInterpolator<f64>,
    u_solution: &DVector<f64>,
    args: &CommandlineArgs,
    filename: impl AsRef<Path>,
    vtk_title: &str,
) -> Result<(), Box<dyn Error>> {
    let embedded_displacements = interpolator.interpolate(u_solution);
    let dataset = create_dataset_with_displacements(&embedded_mesh, &embedded_displacements);
    let vtk_output_file = args
        .output_dir
        .join("fcm_convergence/")
        .join(filename.as_ref());
    write_vtk(dataset, &vtk_output_file, vtk_title)?;
    Ok(())
}

fn create_fcm_hex8_model(
    bg_mesh: &HexMesh<f64>,
    embedding: Embedding<f64>,
) -> Result<EmbeddedModel3d<f64, Hex8Connectivity>, Box<dyn Error>> {
    let quadrature_opts = QuadratureOptions {
        stabilization: None, //Some(StabilizationOptions {
                             //     stabilization_factor: 1e-8,
                             //     stabilization_quadrature: hex_quadrature_strength_3()
                             // })
    };

    tprintln!("Construct stiffness quadrature");
    let stiffness_quadrature = embed_quadrature_3d_with_opts(
        &bg_mesh,
        &embedding,
        hex_quadrature_strength_3(),
        tet_quadrature_strength_3(),
        &quadrature_opts,
    )?;
    tprintln!("Simplifying stiffness quadrature...");
    // TODO: Use simplification in final sim
    let stiffness_quadrature = stiffness_quadrature.simplified(3, &GlopSolver::new())?;
    tprintln!("Finished stiffness quadrature simplification.");
    let elliptic_quadrature = stiffness_quadrature.clone();
    let fe_model = EmbeddedModelBuilder::from_embedding(&bg_mesh, embedding)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();
    Ok(fe_model)
}

fn create_fcm_hex20_model(
    bg_mesh: &HexMesh<f64>,
    embedding: Embedding<f64>,
) -> Result<EmbeddedModel3d<f64, Hex20Connectivity>, Box<dyn Error>> {
    let quadrature_opts = QuadratureOptions {
        stabilization: None, //Some(StabilizationOptions {
                             //     stabilization_factor: 1e-8,
                             //     stabilization_quadrature: hex_quadrature_strength_3()
                             // })
    };

    let bg_mesh = Hex20Mesh::from(bg_mesh);

    tprintln!("Construct stiffness quadrature");
    let stiffness_quadrature = embed_quadrature_3d_with_opts(
        &bg_mesh,
        &embedding,
        hex_quadrature_strength_5(),
        tet_quadrature_strength_5(),
        &quadrature_opts,
    )?;
    tprintln!("Simplifying stiffness quadrature...");
    // TODO: Use simplification in final sim
    let stiffness_quadrature = stiffness_quadrature.simplified(5, &GlopSolver::new())?;
    tprintln!("Finished stiffness quadrature simplification.");
    let elliptic_quadrature = stiffness_quadrature.clone();
    let fe_model = EmbeddedModelBuilder::from_embedding(&bg_mesh, embedding)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();
    Ok(fe_model)
}

fn create_fcm_hex27_model(
    bg_mesh: &HexMesh<f64>,
    embedding: Embedding<f64>,
) -> Result<EmbeddedModel3d<f64, Hex27Connectivity>, Box<dyn Error>> {
    let quadrature_opts = QuadratureOptions {
        stabilization: None, //Some(StabilizationOptions {
                             //     stabilization_factor: 1e-8,
                             //     stabilization_quadrature: hex_quadrature_strength_3()
                             // })
    };

    let bg_mesh = Hex27Mesh::from(bg_mesh);

    tprintln!("Construct stiffness quadrature");
    let stiffness_quadrature = embed_quadrature_3d_with_opts(
        &bg_mesh,
        &embedding,
        hex_quadrature_strength_5(),
        tet_quadrature_strength_5(),
        &quadrature_opts,
    )?;
    tprintln!("Simplifying stiffness quadrature...");
    let stiffness_quadrature = stiffness_quadrature; //.simplified(5, &GlopSolver::new())?;
    tprintln!("Finished stiffness quadrature simplification.");
    let elliptic_quadrature = stiffness_quadrature.clone();
    let fe_model = EmbeddedModelBuilder::from_embedding(&bg_mesh, embedding)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();
    Ok(fe_model)
}

#[derive(Serialize, Deserialize)]
pub struct ExperimentResult {
    pub resolution: usize,
    pub mesh_size: f64,
    pub l2_error: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ExperimentResults {
    pub methods: BTreeMap<String, Vec<ExperimentResult>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommandlineArgs::from_args();

    // let transform_mesh = |mut mesh: Tet4Mesh<f64>| {
    //     for v in mesh.vertices_mut() {
    //         let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::x()), PI);
    //         *v = rotation * *v;
    //     }
    //     mesh
    // };

    // let mesh: Tet4Mesh<f64> = load_mesh(&args.asset_dir, "halfcylinder/halfcylinder_50.msh")?;
    let mesh: Tet4Mesh<f64> = load_mesh(&args.asset_dir, "hemisphere_50_uniform.msh")?;
    // let mesh = transform_mesh(mesh);
    // let mesh = {
    //     let mut mesh = create_rectangular_uniform_hex_mesh(0.5, 2, 1, 2, 16);
    //     mesh.translate(&Vector3::new(-0.51, 0.0, -0.51));
    //     mesh.transform_vertices(|v| *v = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), PI/4.0) * *v);
    //     let polymesh = PolyMesh3d::from(&mesh);
    //     Tet4Mesh::try_from(&polymesh.triangulate()?)?
    // };
    let embedded_mesh = PolyMesh3d::from(&mesh);

    tprintln!(
        "Loaded embedded mesh with {} elements, {} vertices.",
        mesh.connectivity().len(),
        mesh.vertices().len()
    );

    let reference = {
        // let reference_mesh = mesh.clone();
        // let reference_mesh: Tet4Mesh<f64> = load_mesh(&args.asset_dir, "halfcylinder/halfcylinder_50_uniform_refined1.msh")?;
        let ref_mesh_filename = args
            .reference_mesh
            .as_ref()
            .cloned()
            .unwrap_or("hemisphere_50_uniform_refined1.msh".to_string());
        let reference_mesh: Tet4Mesh<f64> = load_mesh(&args.asset_dir, &ref_mesh_filename)?;
        // let reference_mesh = transform_mesh(reference_mesh);
        // let reference_mesh = {
        //     let mut mesh = create_rectangular_uniform_hex_mesh(0.93333, 2, 1, 2, 8);
        //     mesh.translate(&Vector3::new(-1.0, 0.0, -1.0));
        //     let polymesh = PolyMesh3d::from(&mesh);
        //     Tet4Mesh::try_from(&polymesh.triangulate()?)?
        // };
        simulate_reference_solution(&reference_mesh, &args)?
    };

    let resolutions = args
        .resolutions
        .as_ref()
        .cloned()
        // Pick some sensible (small) defaults
        .unwrap_or(vec![1, 2, 4, 8, 16]);

    let mut results = ExperimentResults {
        methods: Default::default(),
    };

    for res in resolutions {
        tprintln!("==================================");
        tprintln!("Resolution {}...", res);

        let bg_mesh = {
            let mut mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 1, 2, res);
            mesh.translate(&Vector3::new(-1.0, 0.0, -1.0));
            mesh
        };
        let h = 1.0 / (res as f64);

        let mut push_result = |name: &str, l2_error| {
            results
                .methods
                .entry(name.to_string())
                .or_insert_with(|| Vec::new())
                .push(ExperimentResult {
                    resolution: res,
                    mesh_size: h,
                    l2_error,
                });
        };

        tprintln!("Simulating FCM Hex8...");
        {
            let l2_error = simulate_fcm_hex8(&bg_mesh, &mesh, &embedded_mesh, res, &args, &reference)?;
            push_result("fcm_hex8", l2_error);
        }
        tprintln!("----------------------------------");
        tprintln!("Simulating embedded FEM Hex8...");
        {
            let l2_error = simulate_fem_hex8(&bg_mesh, &mesh, &embedded_mesh, res, &args, &reference)?;
            push_result("fem_hex8", l2_error);
        }
        tprintln!("----------------------------------");
        tprintln!("Simulating FCM Hex20...");
        {
            let l2_error = simulate_fcm_hex20(&bg_mesh, &mesh, &embedded_mesh, res, &args, &reference)?;
            push_result("fcm_hex20", l2_error);
        }
        tprintln!("----------------------------------");
        tprintln!("Simulating embedded FEM Hex20...");
        {
            let l2_error = simulate_fem_hex20(&bg_mesh, &mesh, &embedded_mesh, res, &args, &reference)?;
            push_result("fem_hex20", l2_error);
        }

        // tprintln!("Simulating FCM Hex27...");
        // {
        //     let l2_error = simulate_fcm_hex27(&bg_mesh, &mesh, &embedded_mesh, res, &args, &reference)?;
        //     push_result("fcm_hex27", l2_error);
        // }

        {
            let json_filename = args.output_dir.join("fcm_convergence/hex_results.json");
            let mut file = File::create(json_filename)?;
            serde_json::to_writer_pretty(&mut file, &results)?;
        }
    }

    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct ReferenceSolution {
    mesh: Tet20Mesh<f64>,
    displacement: DVector<f64>,
}

fn simulate_reference_solution(
    mesh: &Tet4Mesh<f64>,
    args: &CommandlineArgs,
) -> Result<ReferenceSolution, Box<dyn Error>> {
    // Use quadratic tets for the reference solution
    let mesh = Tet20Mesh::from(mesh);
    tprintln!(
        "Mesh for reference solution: {} elements, {} nodes.",
        mesh.connectivity().len(),
        mesh.vertices().len()
    );
    let quadrature = tet_quadrature_strength_5();
    let model = NodalModel::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let rhs_cell_quadrature = tet_quadrature_strength_10();
    let rhs_quadrature = |_| &rhs_cell_quadrature;

    let u_inital = DVector::zeros(model.ndof());
    let u_sol = solve_steady_state(&model, &rhs_quadrature, &u_inital, true)?;

    assert_eq!(u_sol.len() % 3, 0);
    let displacements: Vec<_> = u_sol
        .as_slice()
        .chunks_exact(3)
        .map(|u_i| Vector3::from_column_slice(u_i))
        .collect();
    let dataset = create_dataset_with_displacements(&mesh, &displacements);

    let filename = "reference_solution.vtk";
    let vtk_output_file = args.output_dir.join("fcm_convergence/").join(filename);
    write_vtk(dataset, vtk_output_file, "Reference solution")?;
    Ok(ReferenceSolution {
        mesh,
        displacement: u_sol,
    })
}

fn construct_bg_mesh_from_embedding(initial_bg_mesh: &HexMesh<f64>, embedding: &Embedding<f64>) -> HexMesh<f64> {
    let keep_cells = {
        let mut cells = embedding.interface_cells.clone();
        cells.extend(&embedding.interior_cells);
        cells.sort_unstable();
        cells
    };
    initial_bg_mesh.keep_cells(&keep_cells)
}

#[allow(unused)]
fn simulate_fem_hex8(
    bg_mesh: &HexMesh<f64>,
    embedded_tet_mesh: &Tet4Mesh<f64>,
    embedded_mesh: &PolyMesh3d<f64>,
    res: usize,
    args: &CommandlineArgs,
    reference: &ReferenceSolution,
) -> Result<f64, Box<dyn Error>> {
    // Embed the mesh in order to determine the "real" background mesh (i.e. cutting off exterior
    // cells).
    tprintln!("Embedding mesh...");
    let embedding = embed_mesh_3d(&bg_mesh, &embedded_mesh);
    tprintln!(
        "Number of background cells: {}",
        embedding.interface_cells.len() + embedding.interior_cells.len()
    );

    let mesh = construct_bg_mesh_from_embedding(&bg_mesh, &embedding);

    let quadrature = hex_quadrature_strength_3();
    let fe_model = NodalModel3d::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let rhs_cell_quadrature = hex_quadrature_strength_11();
    let rhs_quadrature = |_| &rhs_cell_quadrature;

    let u_inital = DVector::zeros(fe_model.ndof());
    tprintln!("Solve...");
    let u_sol = solve_steady_state(&fe_model, &rhs_quadrature, &u_inital, false)?;

    tprintln!("Create interpolator...");
    let accelerator = GeometryCollectionAccelerator::new(&fe_model);
    let interpolator = accelerator.make_interpolator(&embedded_tet_mesh.vertices())?;

    tprintln!("Estimate L2 error...");
    let bg_poly_mesh = PolyMesh3d::from(&mesh);
    let l2_error = estimate_l2_error(
        "fem_hex8",
        res,
        args,
        reference,
        &bg_poly_mesh,
        &accelerator,
        DVectorSlice::from(&u_sol),
    )?;
    tprintln!("L2 error: {:3.3e}", l2_error);

    dump_embedded_solution(
        &embedded_tet_mesh,
        &interpolator,
        &u_sol,
        &args,
        format!("hemisphere_solved_fem_hex8_{}.vtk", res),
        "FEM solved Hex8",
    )?;

    Ok(l2_error)
}

#[allow(unused)]
fn simulate_fem_hex20(
    bg_mesh: &HexMesh<f64>,
    embedded_tet_mesh: &Tet4Mesh<f64>,
    embedded_mesh: &PolyMesh3d<f64>,
    res: usize,
    args: &CommandlineArgs,
    reference: &ReferenceSolution,
) -> Result<f64, Box<dyn Error>> {
    // Embed the mesh in order to determine the "real" background mesh (i.e. cutting off exterior
    // cells).
    tprintln!("Embedding mesh...");
    let embedding = embed_mesh_3d(&bg_mesh, &embedded_mesh);
    tprintln!(
        "Number of background cells: {}",
        embedding.interface_cells.len() + embedding.interior_cells.len()
    );

    let hex8_mesh = construct_bg_mesh_from_embedding(&bg_mesh, &embedding);
    let mesh = Hex20Mesh::from(&hex8_mesh);

    let quadrature = hex_quadrature_strength_5();
    let fe_model = NodalModel3d::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let rhs_cell_quadrature = hex_quadrature_strength_11();
    let rhs_quadrature = |_| &rhs_cell_quadrature;

    let u_inital = DVector::zeros(fe_model.ndof());
    tprintln!("Solve...");
    let u_sol = solve_steady_state(&fe_model, &rhs_quadrature, &u_inital, false)?;

    tprintln!("Create interpolator...");
    let accelerator = GeometryCollectionAccelerator::new(&fe_model);
    let interpolator = accelerator.make_interpolator(&embedded_tet_mesh.vertices())?;

    tprintln!("Estimate L2 error...");
    let bg_poly_mesh = PolyMesh3d::from(&hex8_mesh);
    let l2_error = estimate_l2_error(
        "fem_hex20",
        res,
        args,
        reference,
        &bg_poly_mesh,
        &accelerator,
        DVectorSlice::from(&u_sol),
    )?;
    tprintln!("L2 error: {:3.3e}", l2_error);

    dump_embedded_solution(
        &embedded_tet_mesh,
        &interpolator,
        &u_sol,
        &args,
        format!("hemisphere_solved_fem_hex20_{}.vtk", res),
        "FEM solved Hex20",
    )?;

    Ok(l2_error)
}

fn interpolate_displacement<'a, S>(
    space: &'a S,
    displacements: DVectorSlice<f64>,
    x_material: &Point3<f64>,
) -> Result<Vector3<f64>, Box<dyn Error>>
where
    S: GeometricFiniteElementSpace<'a, f64> + DistanceQuery<'a, Point3<f64>>,
    S::Connectivity: ElementConnectivity<f64, GeometryDim = U3, ReferenceDim = U3>,
    DefaultAllocator: ElementConnectivityAllocator<f64, S::Connectivity>
        + Allocator<f64, U3, <S::Connectivity as ElementConnectivity<f64>>::NodalDim>
        + VolumeFiniteElementAllocator<f64, U3, <S::Connectivity as ElementConnectivity<f64>>::NodalDim>,
{
    let element_index = space
        .nearest(x_material)
        .ok_or_else(|| Box::<dyn Error>::from("Failed to find nearest element."))?;

    let connectivity = space.get_connectivity(element_index).unwrap();
    let element = connectivity.element(space.vertices()).unwrap();
    // TODO: It shouldn't be necessary to specify the type here, but there looks as if
    // there might be a bug in type inference?
    let u_element: MatrixMN<f64, U3, _> = connectivity.element_variables(&displacements);
    let xi = map_physical_coordinates(&element, x_material)?;
    let phi = element.evaluate_basis(&xi.coords);
    let u: Vector3<f64> = u_element * phi.transpose();

    Ok(u)
}

fn solve_and_estimate_fcm_error<C>(
    method_name: &str,
    res: usize,
    fe_model: &EmbeddedModel3d<f64, C>,
    bg_mesh: &HexMesh<f64>,
    embedded_tet_mesh: &Tet4Mesh<f64>,
    embedding: &Embedding<f64>,
    reference: &ReferenceSolution,
    args: &CommandlineArgs,
) -> Result<f64, Box<dyn Error>>
where
    C: Sync + Connectivity,
    C: CellConnectivity<f64, U3>,
    C::Cell: Sync + BoundedGeometry<f64, Dimension = U3> + Distance<f64, Point3<f64>>,
    C: ElementConnectivity<f64, GeometryDim = U3, ReferenceDim = U3>,
    U3: DimNameMul<C::NodalDim>,
    C::NodalDim: DimNameMul<U3>,
    DefaultAllocator: ElementConnectivityAllocator<f64, C> + FiniteElementMatrixAllocator<f64, U3, U3, C::NodalDim>,

    // TODO: This shouldn't be necessary, is due to a bug in rustc. Report...?
    DefaultAllocator: ElementConnectivityAllocator<f64, Hex8Connectivity>,
{
    tprintln!("Create interpolator...");
    let accelerator = GeometryCollectionAccelerator::new(fe_model);
    let interpolator = accelerator.make_interpolator(&embedded_tet_mesh.vertices())?;

    // Quadrature used for error measurement
    // TODO: Increase accuracy?
    let rhs_interior_quadrature = hex_quadrature_strength_11();
    tprintln!("Set up error/rhs quadrature");
    let rhs_quadrature = embed_quadrature_3d_with_opts(
        bg_mesh,
        embedding,
        rhs_interior_quadrature.clone(),
        tet_quadrature_strength_10(),
        &QuadratureOptions::default(),
    )?;

    let num_interior = fe_model.interior_connectivity().len();
    let rhs_quadrature_table = |i| -> &QuadraturePair3d<f64> {
        if i < num_interior {
            rhs_quadrature.interior_quadrature()
        } else {
            rhs_quadrature
                .interface_quadratures()
                .get(i - num_interior)
                .unwrap()
        }
    };

    tprintln!("Solve...");
    let u_initial = DVector::zeros(fe_model.ndof());
    let u_sol = solve_steady_state(fe_model, &rhs_quadrature_table, &u_initial, false)?;

    tprintln!("Estimate L2 error...");
    let bg_integration_mesh = construct_bg_mesh_from_embedding(bg_mesh, embedding);
    let bg_integration_mesh = PolyMesh3d::from(&bg_integration_mesh);
    let l2_error = estimate_l2_error(
        method_name,
        res,
        args,
        reference,
        &bg_integration_mesh,
        &accelerator,
        DVectorSlice::from(&u_sol),
    )?;
    tprintln!("L2 error: {:3.3e}", l2_error);

    dump_embedded_solution(
        embedded_tet_mesh,
        &interpolator,
        &u_sol,
        &args,
        format!("hemisphere_{}_solved_{}.vtk", method_name, res),
        "FCM solved",
    )?;

    Ok(l2_error)
}

fn simulate_fcm_hex8(
    bg_mesh: &HexMesh<f64>,
    mesh: &Tet4Mesh<f64>,
    embedded_mesh: &PolyMesh3d<f64>,
    res: usize,
    args: &CommandlineArgs,
    reference: &ReferenceSolution,
) -> Result<f64, Box<dyn Error>> {
    tprintln!("Embedding mesh...");
    let embed_opts = EmbedOptions {
        upper_volume_threshold: 0.999999,
        lower_volume_threshold: 1e-9,
    };
    let embedding = embed_mesh_3d_with_opts(&bg_mesh, &embedded_mesh, &embed_opts);
    tprintln!(
        "Number of background cells: {}",
        embedding.interface_cells.len() + embedding.interior_cells.len()
    );

    tprintln!("Creating embedded model");
    let fe_model = create_fcm_hex8_model(&bg_mesh, embedding.clone())?;

    let filename = format!("bgmesh_{}.vtk", res);
    let vtk_bg_output_file = args.output_dir.join("fcm_convergence/").join(filename);
    write_vtk(
        fe_model.background_mesh(),
        vtk_bg_output_file,
        &format!("Background mesh (res {})", res),
    )?;
    tprintln!(
        "ndofs: {}, interior cells: {}, interface cells: {}",
        fe_model.ndof(),
        fe_model.interior_connectivity().len(),
        fe_model.interface_connectivity().len()
    );

    solve_and_estimate_fcm_error("fcm_hex8", res, &fe_model, &bg_mesh, &mesh, &embedding, reference, args)
}

fn simulate_fcm_hex20(
    bg_mesh: &HexMesh<f64>,
    mesh: &Tet4Mesh<f64>,
    embedded_mesh: &PolyMesh3d<f64>,
    res: usize,
    args: &CommandlineArgs,
    reference: &ReferenceSolution,
) -> Result<f64, Box<dyn Error>> {
    tprintln!("Embedding mesh...");
    let embed_opts = EmbedOptions {
        upper_volume_threshold: 0.999999,
        lower_volume_threshold: 1e-9,
    };
    let embedding = embed_mesh_3d_with_opts(&bg_mesh, &embedded_mesh, &embed_opts);
    tprintln!(
        "Number of background cells: {}",
        embedding.interface_cells.len() + embedding.interior_cells.len()
    );

    tprintln!("Creating embedded model");
    let fe_model = create_fcm_hex20_model(&bg_mesh, embedding.clone())?;

    let filename = format!("bgmesh_{}.vtk", res);
    let vtk_bg_output_file = args.output_dir.join("fcm_convergence/").join(filename);
    write_vtk(
        fe_model.background_mesh(),
        vtk_bg_output_file,
        &format!("Background mesh (res {})", res),
    )?;

    tprintln!(
        "ndofs: {}, interior cells: {}, interface cells: {}",
        fe_model.ndof(),
        fe_model.interior_connectivity().len(),
        fe_model.interface_connectivity().len()
    );

    solve_and_estimate_fcm_error(
        "fcm_hex20",
        res,
        &fe_model,
        &bg_mesh,
        &mesh,
        &embedding,
        reference,
        args,
    )
}

#[allow(dead_code)]
fn simulate_fcm_hex27(
    bg_mesh: &HexMesh<f64>,
    mesh: &Tet4Mesh<f64>,
    embedded_mesh: &PolyMesh3d<f64>,
    res: usize,
    args: &CommandlineArgs,
    reference: &ReferenceSolution,
) -> Result<f64, Box<dyn Error>> {
    tprintln!("Embedding mesh...");
    let embedding = embed_mesh_3d(&bg_mesh, &embedded_mesh);
    tprintln!(
        "Number of background cells: {}",
        embedding.interface_cells.len() + embedding.interior_cells.len()
    );

    tprintln!("Creating embedded model");
    let fe_model = create_fcm_hex27_model(&bg_mesh, embedding.clone())?;

    let filename = format!("bgmesh_{}.vtk", res);
    let vtk_bg_output_file = args.output_dir.join("fcm_convergence/").join(filename);
    write_vtk(
        fe_model.background_mesh(),
        vtk_bg_output_file,
        &format!("Background mesh (res {})", res),
    )?;

    tprintln!(
        "ndofs: {}, interior cells: {}, interface cells: {}",
        fe_model.ndof(),
        fe_model.interior_connectivity().len(),
        fe_model.interface_connectivity().len()
    );

    solve_and_estimate_fcm_error(
        "fcm_hex27",
        res,
        &fe_model,
        &bg_mesh,
        &mesh,
        &embedding,
        reference,
        args,
    )
}

fn estimate_l2_error<'a, S>(
    method_name: &str,
    res: usize,
    args: &CommandlineArgs,
    reference: &ReferenceSolution,
    approx_poly_mesh: &PolyMesh3d<f64>,
    approx_space: &'a S,
    approx_displacements: DVectorSlice<f64>,
) -> Result<f64, Box<dyn Error>>
where
    S: Sync + GeometricFiniteElementSpace<'a, f64> + DistanceQuery<'a, Point3<f64>>,
    S::Connectivity: ElementConnectivity<f64, GeometryDim = U3, ReferenceDim = U3>,
    DefaultAllocator: ElementConnectivityAllocator<f64, S::Connectivity>
        + Allocator<f64, U3, ConnectivityNodalDim<f64, S::Connectivity>>
        + VolumeFiniteElementAllocator<f64, U3, <S::Connectivity as ElementConnectivity<f64>>::NodalDim>
        // TODO: This seems to be a rustc bug. Obviously the below bound is satisfied even without
        // requiring it, since all the types are concrete. However, rust will confuse the types
        // with the ones belonging to S otherwise. Probably some normalization issue?
        // Maybe report this once we have more time
        + VolumeFiniteElementAllocator<f64, U3, U10>
        + ElementConnectivityAllocator<f64, Tet20Connectivity>,
{
    let options = EmbedOptions {
        // Ensure that cells don't get labeled "interior".
        upper_volume_threshold: 10000.0,
        lower_volume_threshold: 0.0,
    };
    let embedding = embed_mesh_3d_with_opts(&reference.mesh, approx_poly_mesh, &options);
    assert_eq!(embedding.interior_cells.len(), 0);
    assert_eq!(embedding.interface_cells.len(), reference.mesh.connectivity().len());
    //
    let quadrature = tet_quadrature_strength_10();
    let embedded_quadrature = embed_quadrature_3d(&reference.mesh, &embedding, quadrature.clone(), &quadrature)?;

    let per_element_squared_errors: Vec<_> = reference
        .mesh
        .connectivity()
        .par_iter()
        .enumerate()
        .map(|(i, conn)| {
            let element = conn.element(reference.mesh.vertices()).unwrap();
            let u_element = conn.element_variables(&reference.displacement);

            let u_h = |x: &Point3<_>, _| {
                interpolate_displacement(approx_space, approx_displacements, x).expect("Interpolation failed")
            };

            let error_quadrature = &embedded_quadrature.interface_quadratures()[i];

            let l2_squared = estimate_element_L2_error_squared(&element, u_h, &u_element, &error_quadrature);
            l2_squared
        })
        .collect();

    let l2_error_squared = per_element_squared_errors.iter().sum::<f64>();
    let per_element_errors: Vec<_> = per_element_squared_errors
        .iter()
        .zip(reference.mesh.cell_iter())
        .map(|(err, cell)| err.sqrt() / cell.compute_volume())
        .collect();

    let mut dataset = DataSet::from(&reference.mesh);
    if let DataSet::UnstructuredGrid { ref mut data, .. } = dataset {
        let l2_errors = IOBuffer::from_slice(per_element_errors.as_slice());
        let attribute = Attribute::Scalars {
            num_comp: 1,
            lookup_table: None,
            data: l2_errors,
        };

        data.cell.push((format!("l2_error"), attribute));
    } else {
        panic!("Unexpected data");
    }

    let filename = format!("l2_errors_{}_res_{}.vtk", method_name, res);
    let vtk_output_file = args.output_dir.join("fcm_convergence/").join(&filename);
    write_vtk(dataset, &vtk_output_file, "L2 errors")?;

    Ok(l2_error_squared.sqrt())
}

fn solve_steady_state<M>(
    fe_model: &M,
    rhs_quadrature_table: impl QuadratureTable<f64, U3>,
    u_initial: &DVector<f64>,
    use_iterative: bool,
) -> Result<DVector<f64>, Box<dyn Error>>
where
    M: ElasticityModelParallel<f64, U3>,
    M: FiniteElementSpace<f64>,
    M::Connectivity: ElementConnectivity<f64, GeometryDim = U3, ReferenceDim = U3>,
    U3: DimNameMul<<M::Connectivity as ElementConnectivity<f64>>::NodalDim>,
    DefaultAllocator: ElementConnectivityAllocator<f64, M::Connectivity>
        + Allocator<f64, U3, <M::Connectivity as ElementConnectivity<f64>>::NodalDim>
        // TODO: Obviously these bounds shouldn't be necessary, but without them
        // rustc somehow confuses the generic types
        + Allocator<f64, U3, U3>
        + Allocator<f64, U3>,
{
    let dirichlet_nodes: Vec<_> = fe_model
        .vertices()
        .iter()
        .enumerate()
        .filter(|(_, v)| v.y < 1e-6)
        .map(|(i, _)| i)
        .collect();

    tprintln!("Num Dirichlet nodes: {}", dirichlet_nodes.len());

    let material = LinearElasticMaterial::from(YoungPoisson {
        young: 5e4,
        poisson: 0.00,
    });

    let rhs = {
        let mut rhs = DVector::zeros(fe_model.ndof());
        assemble_source_term_into(
            DVectorSliceMut::from(&mut rhs),
            fe_model,
            &|x| -> Vector3<f64> { body_force(&Point3::from(*x)) },
            &rhs_quadrature_table,
        );
        rhs
    };

    let function = SteadyStateFiniteElementFunction {
        model: fe_model,
        material_model: &material,
        rhs: &rhs,
        dirichlet_nodes,
        use_iterative,
    };

    // Warmstart with the nodal projection from the analytic solution
    let mut u_sol = u_initial.clone();
    // let mut u_sol = DVector::zeros(rhs.len());

    let mut f = DVector::zeros(u_sol.len());
    let mut dx = f.clone();

    let settings = NewtonSettings {
        max_iterations: None,
        tolerance: 1e-9 * rhs.norm(),
    };

    let newton_result = newton_line_search(
        function,
        &mut u_sol,
        &mut f,
        &mut dx,
        settings,
        &mut BacktrackingLineSearch,
    );

    match newton_result {
        Ok(iter) => {
            tprintln!("Newton iters: {}", iter)
        }
        Err(err) => {
            tprintln!("Newton error: {}", err)
        }
    };

    Ok(u_sol)
}

/// Vector function defining the (discrete) equations for static equilibrium (steady state).
pub struct SteadyStateFiniteElementFunction<'a, M, Material> {
    model: &'a M,
    material_model: &'a Material,
    rhs: &'a DVector<f64>,
    dirichlet_nodes: Vec<usize>,
    use_iterative: bool,
}

impl<'a, M, Material> VectorFunction<f64> for SteadyStateFiniteElementFunction<'a, M, Material>
where
    M: ElasticityModelParallel<f64, U3>,
    Material: Sync + ElasticMaterialModel<f64, U3>,
{
    fn dimension(&self) -> usize {
        self.model.ndof()
    }

    fn eval_into(&mut self, f: &mut DVectorSliceMut<f64>, x: &DVectorSlice<f64>) {
        // We solve the non-linear equation
        //  - f_int(u) - rhs = 0
        // so the left-hand side becomes the vector function F(u)

        // Is this necessary...?
        f.fill(0.0);
        self.model.assemble_elastic_pseudo_forces_into_par(
            DVectorSliceMut::from(&mut *f),
            DVectorSlice::from(x),
            self.material_model,
        );
        *f *= -1.0;
        *f -= self.rhs;
        apply_homogeneous_dirichlet_bc_rhs(f, &self.dirichlet_nodes, 3);
    }
}

impl<'a, M, Material> DifferentiableVectorFunction<f64> for SteadyStateFiniteElementFunction<'a, M, Material>
where
    M: ElasticityModelParallel<f64, U3>,
    Material: Sync + ElasticMaterialModel<f64, U3>,
{
    #[allow(non_snake_case)]
    fn solve_jacobian_system(
        &mut self,
        sol: &mut DVectorSliceMut<f64>,
        x: &DVectorSlice<f64>,
        rhs: &DVectorSlice<f64>,
    ) -> Result<(), Box<dyn Error>> {
        // TODO: Reuse sparsity pattern etc.
        tprintln!("Assembling stiffness matrix...");
        let mut A = self
            .model
            .assemble_stiffness_par(&x.clone_owned(), self.material_model)
            .to_csr(Add::add);
        apply_homogeneous_dirichlet_bc_csr::<_, U3>(&mut A, &self.dirichlet_nodes);
        tprintln!(
            "Assembled stiffness matrix: {} nnz, ({} x {})",
            A.nnz(),
            A.nrows(),
            A.ncols()
        );

        if self.use_iterative {
            let jacobi_precond_elements: Vec<_> = A.diag_iter().map(|a_ii| a_ii.recip()).collect();
            let p = CsrMatrix::from_diagonal(DVectorSlice::from_slice(
                jacobi_precond_elements.as_slice(),
                jacobi_precond_elements.len(),
            ));

            use mkl_corrode::mkl_sys::MKL_INT;
            let convert_to_mkl_int = |indices: &[usize]| {
                indices
                    .iter()
                    .map(|idx| MKL_INT::try_from(*idx).unwrap())
                    .collect::<Vec<_>>()
            };

            let p_row_offsets = convert_to_mkl_int(p.row_offsets());
            let p_col_indices = convert_to_mkl_int(p.column_indices());
            let p_mkl = CsrMatrixHandle::from_csr_data(
                p.nrows(),
                p.ncols(),
                &p_row_offsets[..p.nrows()],
                &p_row_offsets[1..],
                &p_col_indices,
                p.values(),
            )
            .expect("Sparse matrix construction should never fail");
            p_mkl
                .set_mv_hint(SparseOperation::NonTranspose, &MatrixDescription::default(), 2000)
                .map_err(|_| Box::<dyn Error>::from("MKL error during set_mv_hint"))?;
            p_mkl
                .optimize()
                .map_err(|_| Box::<dyn Error>::from("MKL error during optimize"))?;

            let a_row_offsets = convert_to_mkl_int(A.row_offsets());
            let a_col_indices = convert_to_mkl_int(A.column_indices());
            let a_mkl = CsrMatrixHandle::from_csr_data(
                A.nrows(),
                A.ncols(),
                &a_row_offsets[..A.nrows()],
                &a_row_offsets[1..],
                &a_col_indices,
                A.values(),
            )
            .expect("Sparse matrix construction should never fail");
            a_mkl
                .set_mv_hint(SparseOperation::NonTranspose, &MatrixDescription::default(), 2000)
                .map_err(|_| Box::<dyn Error>::from("MKL error during set_mv_hint"))?;
            a_mkl
                .optimize()
                .map_err(|_| Box::<dyn Error>::from("MKL error during optimize"))?;
            p_mkl
                .set_mv_hint(SparseOperation::NonTranspose, &MatrixDescription::default(), 2000)
                .map_err(|_| Box::<dyn Error>::from("MKL error during set_mv_hint"))?;
            p_mkl
                .optimize()
                .map_err(|_| Box::<dyn Error>::from("MKL error during optimize"))?;

            let p_mkl_op = MklCsrLinearOperator(&p_mkl);
            let a_mkl_op = MklCsrLinearOperator(&a_mkl);

            tprintln!("Solving with CG...");
            let cg_result = ConjugateGradient::new()
                .with_operator(DebugOperator::new(a_mkl_op))
                .with_preconditioner(&p_mkl_op)
                .with_stopping_criterion(RelativeResidualCriterion::new(1e-9))
                .solve_with_guess(rhs, DVectorSliceMut::from(&mut *sol))?;

            tprintln!("CG iterations: {}", cg_result.num_iterations);
        } else {
            let A_dss =
                dss::SparseMatrix::try_convert_from_csr(A.row_offsets(), A.column_indices(), A.values(), Symmetric)?;
            let options = dss::SolverOptions::default().parallel_reorder(true);
            tprintln!("Factoring system...");
            let mut solver = dss::Solver::try_factor_with_opts(&A_dss, Definiteness::PositiveDefinite, &options)?;
            tprintln!("Done factoring.");
            let solution = solver.solve(rhs.data.as_slice())?;
            sol.copy_from_slice(&solution);
        }
        Ok(())
    }
}

struct DebugOperator<Op> {
    op: Op,
    iters: Cell<usize>,
}

impl<Op> DebugOperator<Op> {
    pub fn new(op: Op) -> Self {
        Self {
            op,
            iters: Cell::new(0),
        }
    }
}

impl<Op> LinearOperator<f64> for DebugOperator<Op>
where
    Op: LinearOperator<f64>,
{
    fn apply(&self, y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        self.op.apply(y, x)?;
        self.iters.replace(self.iters.get() + 1);
        if self.iters.get() % 100 == 0 {
            tprintln!("Finished {} iterations...", self.iters.get());
        }
        Ok(())
    }
}

struct MklCsrLinearOperator<'a>(&'a CsrMatrixHandle<'a, f64>);

impl<'a> LinearOperator<f64> for MklCsrLinearOperator<'a> {
    fn apply(&self, mut y: DVectorSliceMut<f64>, x: DVectorSlice<f64>) -> Result<(), Box<dyn Error>> {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.0.rows());
        assert_eq!(x.len(), self.0.cols());

        let description = MatrixDescription::default();
        mkl_corrode::sparse::spmv_csr(
            SparseOperation::NonTranspose,
            1.0,
            self.0,
            &description,
            x.as_slice(),
            0.0,
            y.as_mut_slice(),
        )
        .map_err(|_| Box::<dyn Error>::from("MKL error during sparse spmv"))?;
        Ok(())
    }
}
