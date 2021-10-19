use fenris::element::{Hex8Element, Quad4d2Element};
use fenris::embedding::{
    compute_element_embedded_quadrature, construct_embedded_quadrature_for_element_2d, embed_cell_2d,
    optimize_quadrature, QuadratureOptions,
};
use fenris::geometry::procedural::{approximate_triangle_mesh_for_sdf_2d, create_simple_stupid_sphere};
use fenris::geometry::sdf::SdfCircle;
use fenris::geometry::{ConvexPolygon, Hexahedron, Quad2d};
use fenris::lp_solvers::GlopSolver;
use fenris::quadrature::{tet_quadrature_strength_10, tri_quadrature_strength_11, Quadrature};
use matrixcompare::assert_scalar_eq;
use nalgebra::{Point2, Point3, Vector2, Vector3};

use std::convert::TryFrom;

#[test]
fn optimize_quadrature_2d() {
    let quad = Quad2d([
        Point2::new(0.0, 2.0),
        Point2::new(2.0, 2.0),
        Point2::new(2.0, 3.0),
        Point2::new(0.0, 3.0),
    ]);

    let circle = SdfCircle {
        radius: 1.0,
        center: Vector2::new(2.0, 3.0),
    };
    let quad_polygon = ConvexPolygon::try_from(quad).unwrap();

    let element = Quad4d2Element::from(quad);
    let triangle_mesh = approximate_triangle_mesh_for_sdf_2d(&circle, 0.05);
    let embedded_polygons = embed_cell_2d(&quad_polygon, &triangle_mesh).unwrap();
    let quadrature =
        construct_embedded_quadrature_for_element_2d(&element, &embedded_polygons, tri_quadrature_strength_11());
    let quadrature_opt = optimize_quadrature(&quadrature, 9, &GlopSolver::new()).unwrap();

    println!(
        "Num quadrature points before optimization: {}",
        quadrature.points().len()
    );
    println!(
        "Num quadrature points after optimization: {}",
        quadrature_opt.points().len()
    );

    let f = |x: f64, y: f64| {
        -1.0 * x.powi(5) * y.powi(4) + 2.0 * x * y - 5.0 * x.powi(4) * y.powi(5)
            + 2.5 * x.powi(1) * y.powi(8)
            + x * y
            + 2.0
    };

    let f = |p: &Vector2<f64>| f(p.x, p.y);

    let original_integral: f64 = quadrature.integrate(f);
    let optimized_integral: f64 = quadrature_opt.integrate(f);

    assert_scalar_eq!(original_integral, optimized_integral, comp = abs, tol = 1e-9);
}

#[test]
fn optimize_quadrature_3d() {
    let element = Hex8Element::from_vertices([
        Point3::new(0.0, 2.0, 0.0),
        Point3::new(2.0, 2.0, 0.0),
        Point3::new(2.0, 3.0, 0.0),
        Point3::new(0.0, 3.0, 0.0),
        Point3::new(0.0, 2.0, 2.0),
        Point3::new(2.0, 2.0, 2.0),
        Point3::new(2.0, 3.0, 2.0),
        Point3::new(0.0, 3.0, 2.0),
    ]);
    let hex = Hexahedron::from_vertices(element.vertices().clone());
    let embedded_mesh =
        create_simple_stupid_sphere(&Point3::new(2.0, 3.0, 2.0), 1.0, 5).intersect_convex_polyhedron(&hex);
    let quadrature = compute_element_embedded_quadrature(
        &element,
        &embedded_mesh,
        tet_quadrature_strength_10(),
        &QuadratureOptions::default(),
    )
    .unwrap();
    let quadrature_opt = optimize_quadrature(&quadrature, 8, &GlopSolver::new()).unwrap();

    println!(
        "Num quadrature points before optimization: {}",
        quadrature.points().len()
    );
    println!(
        "Num quadrature points after optimization: {}",
        quadrature_opt.points().len()
    );

    let f = |x: f64, y: f64, z: f64| {
        -1.0 * x.powi(4) * y.powi(4) + 2.0 * x * y * z.powi(3) - 5.0 * x.powi(4) * y.powi(4)
            + 2.5 * x.powi(1) * y.powi(7)
            + 3.5 * z.powi(8)
            - 9.0 * x.powi(3) * y.powi(3) * z.powi(2)
            + z
            + x * y
            + 2.0
    };

    let f = |p: &Vector3<f64>| f(p.x, p.y, p.z);

    let original_integral: f64 = quadrature.integrate(f);
    let optimized_integral: f64 = quadrature_opt.integrate(f);

    assert_scalar_eq!(original_integral, optimized_integral, comp = abs, tol = 1e-9);
}

#[test]
fn zeroth_order_simplification() {
    // Points are arbitrary
    let points = vec![
        Vector3::new(-0.5, 0.3, 0.5),
        Vector3::new(0.3, 0.2, -0.2),
        Vector3::new(0.4, 0.1, -0.5),
        Vector3::new(0.5, -0.2, 0.1),
    ];
    let weights = vec![0.1, 0.5, 0.8, 3.0];
    let quadrature = (weights.clone(), points.clone());
    let quadrature_opt = optimize_quadrature(&quadrature, 0, &GlopSolver::new()).unwrap();
    let opt_weights = quadrature_opt.0;
    let opt_points = quadrature_opt.1;

    assert_eq!(opt_weights.len(), 1);
    assert_eq!(opt_points.len(), 1);
    let expected_weight: f64 = weights.iter().sum();
    assert_scalar_eq!(opt_weights[0], expected_weight, comp = abs, tol = 1e-9);

    assert!(
        opt_points[0] == points[0]
            || opt_points[0] == points[1]
            || opt_points[0] == points[2]
            || opt_points[0] == points[3]
    )
}
