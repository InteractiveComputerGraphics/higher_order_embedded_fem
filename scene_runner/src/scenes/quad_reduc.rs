use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;

use fenris::element::ElementConnectivity;
use fenris::embedding::{compute_element_embedded_quadrature, embed_mesh_3d, optimize_quadrature, QuadratureOptions};
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;
use fenris::geometry::vtk::write_vtk;
use fenris::geometry::ConvexPolyhedron;
use fenris::lp_solvers::GlopSolver;
use fenris::nalgebra::{Point3, Rotation3, Unit, Vector3};
use fenris::quadrature::{
    tet_quadrature_strength_1, tet_quadrature_strength_10, tet_quadrature_strength_2, tet_quadrature_strength_3,
    tet_quadrature_strength_5, Quadrature,
};
use fenris::vtkio::model::DataSet;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use simulation_toolbox::io::json_helper::serde_json;

use crate::scenes::helpers::PointHelper;
use crate::scenes::{Scene, SceneConstructor, SceneParameters};

pub fn scenes() -> Vec<SceneConstructor> {
    vec![
        SceneConstructor {
            name: "quad_reduc_monomials".to_string(),
            constructor: build_quad_reduc_monomials,
        },
        SceneConstructor {
            name: "quad_reduc_box".to_string(),
            constructor: build_quad_reduc_box,
        },
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MonomialResultSet {
    strength: usize,
    results: Vec<MonomialResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MonomialResult {
    exponents: [u16; 3],
    result: QuadratureReduction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuadratureReduction {
    original_strength: usize,
    reduced_strength: usize,
    original_points: usize,
    reduced_points: usize,
    exact_integral: f64,
    original_integral: f64,
    original_abs_error: f64,
    original_rel_error: f64,
    reduced_integral: f64,
    reduced_abs_error: f64,
    reduced_rel_error: f64,
}

#[derive(Clone, Debug)]
struct PolynomialTerm {
    exponents: Vector3<i32>,
    coefficient: f64,
}

impl PolynomialTerm {
    fn new_axyz(a: f64, x_exp: u16, y_exp: u16, z_exp: u16) -> Self {
        PolynomialTerm {
            exponents: Vector3::new(x_exp as i32, y_exp as i32, z_exp as i32),
            coefficient: a,
        }
    }

    fn evaluate(&self, x: &Vector3<f64>) -> f64 {
        let mut y = self.coefficient;
        for i in 0..3 {
            y *= x[i].powi(self.exponents[i] as i32);
        }
        y
    }

    fn indefinite_integral(&self, x: &Vector3<f64>) -> f64 {
        let e = &self.exponents;
        let numerator = self.coefficient * x.x.powi(e.x + 1) * x.y.powi(e.y + 1) * x.z.powi(e.z + 1);
        let denominator = ((e.z + 1) * (e.x * e.y + e.x + e.y + 1)) as f64;
        numerator / denominator
    }

    #[rustfmt::skip]
    fn definite_integral(&self, x_min: &Vector3<f64>, x_max: &Vector3<f64>) -> f64 {
        let f = |x| self.indefinite_integral(x);

        let points = [
            Vector3::new(x_max.x, x_max.y, x_max.z),
            Vector3::new(x_min.x, x_max.y, x_max.z),
            Vector3::new(x_max.x, x_min.y, x_max.z),
            Vector3::new(x_max.x, x_max.y, x_min.z),
            Vector3::new(x_min.x, x_min.y, x_max.z),
            Vector3::new(x_min.x, x_max.y, x_min.z),
            Vector3::new(x_max.x, x_min.y, x_min.z),
            Vector3::new(x_min.x, x_min.y, x_min.z),
        ];

        let result =
              f(&points[0]) - f(&points[1])
            - f(&points[2]) - f(&points[3])
            + f(&points[4]) + f(&points[5])
            + f(&points[6]) - f(&points[7]);

        result
    }
}

#[derive(Clone, Debug)]
struct Polynomial {
    terms: Vec<PolynomialTerm>,
}

impl Polynomial {
    fn evaluate(&self, x: &Vector3<f64>) -> f64 {
        let mut y = 0.0;
        for term in &self.terms {
            y += term.evaluate(x);
        }
        y
    }

    fn definite_integral(&self, x_min: &Vector3<f64>, x_max: &Vector3<f64>) -> f64 {
        let mut y = 0.0;
        for term in &self.terms {
            y += term.definite_integral(x_min, x_max);
        }
        y
    }
}

pub fn rotate_deg(points: &mut [Point3<f64>], angle_in_deg: f64, axis: &Unit<Vector3<f64>>) {
    let angle = angle_in_deg * 180.0 * std::f64::consts::PI.recip();
    let rot = Rotation3::from_axis_angle(axis, angle);
    for p in points {
        *p = rot * p.clone();
    }
}

fn write_mesh_to_vtk<P: AsRef<Path>, S: AsRef<str>, D: Into<DataSet>>(
    base_path: P,
    name: S,
    dataset: D,
) -> Result<(), Box<dyn Error>> {
    let filename = base_path
        .as_ref()
        .join(format!("{name}.vtk", name = name.as_ref()));
    Ok(write_vtk(dataset.into(), filename, name.as_ref())?)
}

fn build_quad_reduc_monomials(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let scene = Scene {
        initial_state: Default::default(),
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: 0.0,
        name: String::from("quad_reduc_monomials"),
    };

    let embedded_mesh_resolution = 1;
    let embedded_box_size = 0.5;
    let embedded_box_rotation_angle_deg = 32.0;
    let embedded_box_rotation_axis = Unit::new_normalize(Vector3::new(1.0, 0.8, 0.9));

    let embedded_box_rotation_angle_rad = embedded_box_rotation_angle_deg * 180.0 * std::f64::consts::PI.recip();

    let mut embedded_mesh = create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, embedded_mesh_resolution);
    PointHelper::scale_max_extent_to(embedded_mesh.vertices_mut(), embedded_box_size);
    PointHelper::center_to_origin(embedded_mesh.vertices_mut());

    let embedded_bb = PointHelper::bb(embedded_mesh.vertices()).unwrap();
    let embedded_volume = embedded_mesh
        .cell_iter()
        .map(|tet| tet.compute_volume())
        .fold(0.0, std::ops::Add::add);
    println!("Volume of embedded box: {}", embedded_volume);

    // Rotate the mesh
    rotate_deg(
        embedded_mesh.vertices_mut(),
        embedded_box_rotation_angle_deg,
        &embedded_box_rotation_axis,
    );

    let embedded_displacement = {
        // Find min point of mesh
        let aabb = PointHelper::bb(embedded_mesh.vertices()).unwrap();
        let min = aabb.min();

        let safety = Vector3::repeat(1e-1 * embedded_box_size);
        let embedded_displacement = -(*min - safety);

        // Displace it into first octant
        for v in embedded_mesh.vertices_mut() {
            v.coords += embedded_displacement;
        }

        embedded_displacement
    };

    let mut background_mesh = create_rectangular_uniform_hex_mesh(2.0, 1, 1, 1, 1);
    PointHelper::center_to_origin(background_mesh.vertices_mut());

    write_mesh_to_vtk(&params.output_dir, "embedded_mesh", &embedded_mesh)?;
    write_mesh_to_vtk(&params.output_dir, "background_mesh", &background_mesh)?;

    let embedded_mesh = PolyMesh3d::from(&embedded_mesh);
    let embedding = embed_mesh_3d(&background_mesh, &embedded_mesh);

    // Define function f to integrate, computed by rotating points into the coordinate
    // system of the embedded box.
    let construct_f = |poly: &Polynomial| {
        let p = poly.clone();
        move |x: &Vector3<f64>| -> f64 {
            let x = x - &embedded_displacement;
            let rot = Rotation3::from_axis_angle(&embedded_box_rotation_axis, -embedded_box_rotation_angle_rad);
            let x_transformed = rot * x + &embedded_displacement;
            p.evaluate(&x_transformed)
        }
    };

    // Quadrature strengths and rules
    let tet_quadratures = vec![
        (1, tet_quadrature_strength_1(), 1),
        (2, tet_quadrature_strength_2(), 2),
        (3, tet_quadrature_strength_3(), 3),
        (4, tet_quadrature_strength_5(), 5),
        (5, tet_quadrature_strength_5(), 5),
        (6, tet_quadrature_strength_10(), 10),
        (7, tet_quadrature_strength_10(), 10),
        (8, tet_quadrature_strength_10(), 10),
        (9, tet_quadrature_strength_10(), 10),
        (10, tet_quadrature_strength_10(), 10),
    ];

    let construct_monomials_exponents = |order: usize| -> Vec<[u16; 3]> {
        std::iter::repeat(0..=order)
            // We want a polynomial in three dimensions, so take 3x [0..N]
            .take(3)
            .multi_cartesian_product()
            // Filter out all terms with lower order
            .filter(|exponents| exponents.iter().sum::<usize>() <= order)
            .map(|exponents| {
                assert!(exponents.len() == 3);
                [exponents[0] as u16, exponents[1] as u16, exponents[2] as u16]
            })
            .collect()
    };

    let mut results = Vec::new();
    for (strength, quadrature, original_strength) in tet_quadratures {
        println!(
            "Original strength: {}, reduced strength: {}",
            original_strength, strength
        );

        let quadratures: Vec<_> = embedding
            .interface_cells
            .iter()
            .zip(embedding.interface_cell_embeddings.iter())
            .map(|(bg_cell_idx, embedded_intersection)| {
                let element = background_mesh
                    .connectivity()
                    .get(*bg_cell_idx)
                    .unwrap()
                    .element(background_mesh.vertices())
                    .unwrap();
                compute_element_embedded_quadrature(
                    &element,
                    embedded_intersection,
                    &quadrature,
                    &QuadratureOptions::default(),
                )
                .unwrap()
            })
            .collect();

        assert_eq!(quadratures.len(), 1);
        println!("Computed quadrature. Starting optimization...");

        let quadrature = quadratures.first().unwrap();
        let quadrature_opt = optimize_quadrature(&quadrature, strength, &GlopSolver::new()).unwrap();

        println!(
            "Num quadrature points before optimization: {}",
            quadrature.points().len()
        );
        println!(
            "Num quadrature points after optimization: {}",
            quadrature_opt.points().len()
        );

        let exponent_sets = construct_monomials_exponents(strength);
        let mut monomial_results = Vec::with_capacity(exponent_sets.len());

        println!("Monomial exponent sets for strength {}:", strength);
        for exponents in exponent_sets.iter() {
            println!("{:?}", exponents);
        }

        println!("Computing integrals...");
        for exponents in exponent_sets.iter() {
            let monomial = Polynomial {
                terms: vec![PolynomialTerm::new_axyz(1.0, exponents[0], exponents[1], exponents[2])],
            };

            let f = construct_f(&monomial);

            let exact_integral: f64 = monomial.definite_integral(
                &(embedded_bb.min() + embedded_displacement),
                &(embedded_bb.max() + embedded_displacement),
            );
            //monomial.definite_integral(embedded_bb.min(), embedded_bb.max());
            let original_integral: f64 = quadrature.integrate(|x| f(x));
            let optimized_integral: f64 = quadrature_opt.integrate(|x| f(x));

            let original_absdiff = (exact_integral - original_integral).abs();
            let optimized_absdiff = (exact_integral - optimized_integral).abs();

            let (original_reldiff, optimized_reldiff) = {
                let original_reldiff = original_absdiff / exact_integral.abs();
                let optimized_reldiff = optimized_absdiff / exact_integral.abs();

                //assert!(original_reldiff.is_finite());
                //assert!(optimized_reldiff.is_finite());

                (original_reldiff, optimized_reldiff)
            };

            println!("Exact integral     : {:.15e}", exact_integral);
            println!("Original integral  : {:.15e}", original_integral);
            println!("Original abs error : {:.3e}", original_absdiff);
            println!("Original rel error : {:.3e}", original_reldiff);
            println!("Optimized integral : {:.15e}", optimized_integral);
            println!("Optimized abs error: {:.3e}", optimized_absdiff);
            println!("Optimized rel error: {:.3e}", optimized_reldiff);

            monomial_results.push(MonomialResult {
                exponents: exponents.clone(),
                result: QuadratureReduction {
                    original_strength,
                    reduced_strength: strength,
                    original_points: quadrature.points().len(),
                    reduced_points: quadrature_opt.points().len(),
                    exact_integral: exact_integral,
                    original_integral: original_integral,
                    original_abs_error: original_absdiff,
                    original_rel_error: original_reldiff,
                    reduced_integral: optimized_integral,
                    reduced_abs_error: optimized_absdiff,
                    reduced_rel_error: optimized_reldiff,
                },
            });

            println!("");
        }

        results.push(MonomialResultSet {
            strength,
            results: monomial_results,
        })
    }

    let output_json = serde_json::to_string_pretty(&results)?;
    let mut json_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(r#"U:\Documents\Programming\rust\femproto2\notebooks\quad_plots\quad_reduc_results_monomials.json"#)?;
    json_file.write_all(output_json.as_bytes())?;

    Ok(scene)
}

fn build_quad_reduc_box(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let scene = Scene {
        initial_state: Default::default(),
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: 0.0,
        name: String::from("quad_reduc_box"),
    };

    let embedded_mesh_resolution = 1;
    let embedded_box_size = 1.1;
    let embedded_box_rotation_angle_deg = 32.0;
    let embedded_box_rotation_axis = Unit::new_normalize(Vector3::new(1.0, 0.8, 0.9));
    let embedded_box_rotation_angle_rad = embedded_box_rotation_angle_deg * 180.0 * std::f64::consts::PI.recip();

    let mut embedded_mesh = create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, embedded_mesh_resolution);
    PointHelper::scale_max_extent_to(embedded_mesh.vertices_mut(), embedded_box_size);
    PointHelper::center_to_origin(embedded_mesh.vertices_mut());

    let embedded_bb = PointHelper::bb(embedded_mesh.vertices()).unwrap();
    let embedded_volume = embedded_mesh
        .cell_iter()
        .map(|tet| tet.compute_volume())
        .fold(0.0, std::ops::Add::add);
    println!("Volume of embedded box: {}", embedded_volume);

    rotate_deg(
        embedded_mesh.vertices_mut(),
        embedded_box_rotation_angle_deg,
        &embedded_box_rotation_axis,
    );

    let mut background_mesh = create_rectangular_uniform_hex_mesh(2.0, 1, 1, 1, 1);
    PointHelper::center_to_origin(background_mesh.vertices_mut());

    write_mesh_to_vtk(&params.output_dir, "embedded_mesh", &embedded_mesh)?;
    write_mesh_to_vtk(&params.output_dir, "background_mesh", &background_mesh)?;

    let embedded_mesh = PolyMesh3d::from(&embedded_mesh);
    let embedding = embed_mesh_3d(&background_mesh, &embedded_mesh);

    // Polynomial defined in non-rotated embedded object
    let poly = Polynomial {
        terms: vec![
            PolynomialTerm::new_axyz(0.05, 0, 0, 0),
            PolynomialTerm::new_axyz(0.1, 0, 1, 0),
            PolynomialTerm::new_axyz(1.0, 0, 0, 1),
            PolynomialTerm::new_axyz(1.0, 2, 0, 0),
            PolynomialTerm::new_axyz(1.0, 0, 2, 0),
            PolynomialTerm::new_axyz(10.0, 2, 2, 2),
            PolynomialTerm::new_axyz(3.0, 4, 2, 1),
            PolynomialTerm::new_axyz(20.5, 6, 0, 4),
            PolynomialTerm::new_axyz(-1.5, 2, 2, 1),
            PolynomialTerm::new_axyz(30.0, 4, 2, 2),
            PolynomialTerm::new_axyz(200.0, 2, 5, 3),
            PolynomialTerm::new_axyz(3000.0, 7, 2, 1),
            PolynomialTerm::new_axyz(100.0, 9, 0, 0),
            PolynomialTerm::new_axyz(1.0, 0, 0, 9),
        ],
    };

    // Define function f to integrate, computed by rotating points into the coordinate
    // system of the embedded box.
    let f = {
        let p = poly.clone();
        move |x: &Vector3<f64>| -> f64 {
            let rot = Rotation3::from_axis_angle(&embedded_box_rotation_axis, -embedded_box_rotation_angle_rad);
            let x_transformed = rot * x;
            p.evaluate(&x_transformed)
        }
    };

    // Quadrature strengths and rules
    let tet_quadratures = vec![
        (1, tet_quadrature_strength_1(), 1),
        (2, tet_quadrature_strength_2(), 2),
        (3, tet_quadrature_strength_3(), 3),
        (4, tet_quadrature_strength_5(), 5),
        (5, tet_quadrature_strength_5(), 5),
        (6, tet_quadrature_strength_10(), 10),
        (7, tet_quadrature_strength_10(), 10),
        (8, tet_quadrature_strength_10(), 10),
        (9, tet_quadrature_strength_10(), 10),
        (10, tet_quadrature_strength_10(), 10),
    ];

    let mut output = Vec::new();
    for (strength, quadrature, original_strength) in tet_quadratures {
        println!(
            "Original strength: {}, reduced strength: {}",
            original_strength, strength
        );

        let quadratures: Vec<_> = embedding
            .interface_cells
            .iter()
            .zip(embedding.interface_cell_embeddings.iter())
            .map(|(bg_cell_idx, embedded_intersection)| {
                let element = background_mesh
                    .connectivity()
                    .get(*bg_cell_idx)
                    .unwrap()
                    .element(background_mesh.vertices())
                    .unwrap();
                compute_element_embedded_quadrature(
                    &element,
                    embedded_intersection,
                    &quadrature,
                    &QuadratureOptions::default(),
                )
                .unwrap()
            })
            .collect();

        assert_eq!(quadratures.len(), 1);
        println!("Computed quadrature. Starting optimization...");

        let quadrature = quadratures.first().unwrap();
        let quadrature_opt = optimize_quadrature(&quadrature, strength, &GlopSolver::new()).unwrap();

        println!(
            "Num quadrature points before optimization: {}",
            quadrature.points().len()
        );
        println!(
            "Num quadrature points after optimization: {}",
            quadrature_opt.points().len()
        );

        let exact_integral: f64 = poly.definite_integral(embedded_bb.min(), embedded_bb.max());
        let original_integral: f64 = quadrature.integrate(|x| f(x));
        let optimized_integral: f64 = quadrature_opt.integrate(|x| f(x));

        let original_absdiff = (exact_integral - original_integral).abs();
        let optimized_absdiff = (exact_integral - optimized_integral).abs();
        let original_reldiff = original_absdiff / exact_integral.abs();
        let optimized_reldiff = optimized_absdiff / exact_integral.abs();

        println!("Exact integral     : {:.15e}", exact_integral);
        println!("Original integral  : {:.15e}", original_integral);
        println!("Original abs error : {:.3e}", original_absdiff);
        println!("Original rel error : {:.3e}", original_reldiff);
        println!("Optimized integral : {:.15e}", optimized_integral);
        println!("Optimized abs error: {:.3e}", optimized_absdiff);
        println!("Optimized rel error: {:.3e}", optimized_reldiff);

        output.push(QuadratureReduction {
            original_strength,
            reduced_strength: strength,
            original_points: quadrature.points().len(),
            reduced_points: quadrature_opt.points().len(),
            exact_integral: exact_integral,
            original_integral: original_integral,
            original_abs_error: original_absdiff,
            original_rel_error: original_reldiff,
            reduced_integral: optimized_integral,
            reduced_abs_error: optimized_absdiff,
            reduced_rel_error: optimized_reldiff,
        })
    }

    let output_json = serde_json::to_string_pretty(&output)?;
    let mut json_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(r#"U:\Documents\Programming\rust\femproto2\notebooks\quad_plots\quad_reduc_results.json"#)?;
    json_file.write_all(output_json.as_bytes())?;

    Ok(scene)
}
