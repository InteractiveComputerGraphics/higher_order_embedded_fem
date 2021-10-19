use fenris::mesh::TriangleMesh2d;

use fenris::element::{FiniteElement, Hex8Element, Quad4d2Element};
use fenris::embedding::{
    compute_element_embedded_quadrature, construct_embedded_quadrature, embed_mesh_2d, QuadratureOptions,
};
use fenris::geometry::procedural::{create_rectangular_uniform_hex_mesh, create_rectangular_uniform_quad_mesh_2d};
use fenris::quadrature::{
    hex_quadrature_strength_11, tet_quadrature_strength_10, tri_quadrature_strength_5_f64, Quadrature,
};
use nalgebra::{Point2, Point3, Vector2, Vector3};

use fenris::connectivity::{CellConnectivity, Tri3d2Connectivity};
use fenris::geometry::polymesh::PolyMesh3d;
use matrixcompare::assert_scalar_eq;

#[test]
fn embedded_trimesh_integrates_to_correct_area() {
    let a = Point2::new(1.0, 1.0);
    let b = Point2::new(2.0, 3.0);
    let c = Point2::new(3.0, 0.0);
    let d = Point2::new(2.0, -2.0);
    let e = Point2::new(5.0, -1.0);

    let trimesh = {
        let vertices = vec![a, b, c, d, e];
        let cells = vec![
            Tri3d2Connectivity([0, 2, 1]),
            Tri3d2Connectivity([3, 2, 0]),
            Tri3d2Connectivity([3, 4, 2]),
        ];
        TriangleMesh2d::from_vertices_and_connectivity(vertices, cells)
    };

    let top_left = Vector2::new(0.0, 4.0);
    let background_mesh = create_rectangular_uniform_quad_mesh_2d(3.0, 2, 2, 1, &top_left);

    let embedded = embed_mesh_2d(&background_mesh, &trimesh).unwrap();
    let interface_element_embedding = embedded.clone().into_iter().map(|(cell_index, polygons)| {
        let cell = background_mesh.connectivity()[cell_index]
            .cell(background_mesh.vertices())
            .unwrap();
        let quad_element = Quad4d2Element::from(cell);
        (quad_element, polygons)
    });

    // TODO: Remove this scope after debugging is complete
    //    {
    //        use fenris::geometry::vtk::{create_vtk_data_set_from_polygons, write_vtk};
    //        use vtkio::model::DataSet;
    //        let embedded_data_set = DataSet::from(&trimesh);
    //        write_vtk(
    //            embedded_data_set,
    //            "data/test_trimesh.vtk",
    //            "test trimesh",
    //        ).unwrap();
    //
    //        let background_data_set = DataSet::from(&background_mesh);
    //        write_vtk(
    //            background_data_set,
    //            "data/test_background_mesh.vtk",
    //            "test background mesh",
    //        ).unwrap();
    //
    //        let polygons = embedded.clone()
    //            .into_iter()
    //            .map(|(_, polygons)| polygons)
    //            .flatten()
    //            .collect::<Vec<_>>();
    //        let embedded_data_set = create_vtk_data_set_from_polygons(&polygons);
    //        write_vtk(
    //            embedded_data_set,
    //            "data/test_embedded_polygons.vtk",
    //            " test embedded polygons",
    //        ).unwrap();
    //    }

    let quadratures =
        construct_embedded_quadrature(interface_element_embedding.clone(), tri_quadrature_strength_5_f64());
    assert_eq!(quadratures.len(), embedded.len());

    let area: f64 = interface_element_embedding
        .zip(quadratures)
        .map(|((quad_element, _), quadrature)| {
            quadrature.integrate(|p| quad_element.reference_jacobian(p).determinant())
        })
        .sum();

    assert_scalar_eq!(area, 7.5, comp = abs, tol = 1e-12);
}

#[test]
fn compute_element_embedded_quadrature_test() {
    // Resolution of embedded mesh
    let resolutions = [1, 2];

    for res in &resolutions {
        // We consider a Hexahedron shape [0, 0.5]^3 represented by several smaller hexahedra,
        // embedded in a single larger Hexahedron [-1, 1]^3 element.

        let hex_quadrature = hex_quadrature_strength_11::<f64>();
        // Since the mapping from the reference element to our hexahedral element is linear,
        // and the mapping from the reference tet to each individual tet is linear,
        // we should be able to integrate polynomials of at least degree 8 (10 - 2) given a
        // 10-strength quadrature rule for our tets
        // TODO: Maybe use a bg element that is not the reference element for better generality
        let tet_quadrature = tet_quadrature_strength_10::<f64>();
        let bg_element = Hex8Element::<f64>::reference();

        // We use an element with the exact shape of the intersection to use as a
        // "ground truth" solution
        let ground_truth_element = Hex8Element::from_vertices([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.0, 0.0),
            Point3::new(0.5, 0.5, 0.0),
            Point3::new(0.0, 0.5, 0.0),
            Point3::new(0.0, 0.0, 0.5),
            Point3::new(0.5, 0.0, 0.5),
            Point3::new(0.5, 0.5, 0.5),
            Point3::new(0.0, 0.5, 0.5),
        ]);

        let f_separate_coords =
            |x: f64, y: f64, z: f64| x.powi(5) * z.powi(3) + x.powi(3) * y.powi(3) * z + x + y + z + 5.0;
        let f = |x: &Vector3<f64>| f_separate_coords(x.x, x.y, x.z);

        let embedded_intersection = PolyMesh3d::from(&create_rectangular_uniform_hex_mesh(0.5, 1, 1, 1, *res));

        let (weights, points) = compute_element_embedded_quadrature(
            &bg_element,
            &embedded_intersection,
            tet_quadrature,
            &QuadratureOptions::default(),
        )
        .unwrap();

        // Since our element is identical to the reference element, the sum of the quadrature
        // weights must be equal to the volume.
        assert_scalar_eq!(weights.iter().sum::<f64>(), 0.125, comp = abs, tol = 1e-12);

        let quadrature = (weights, points);
        let integral: f64 = quadrature.integrate(f);

        let expected_integral: f64 = hex_quadrature.integrate(|xi| {
            let x = ground_truth_element.map_reference_coords(xi);
            let j = ground_truth_element.reference_jacobian(xi);
            j.determinant().abs() * f(&x)
        });

        assert_scalar_eq!(integral, expected_integral, comp = abs, tol = 1e-12);
    }
}
