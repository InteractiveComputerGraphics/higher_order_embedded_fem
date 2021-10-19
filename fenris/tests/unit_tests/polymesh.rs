use fenris::geometry::polymesh::{PolyMesh, PolyMesh3d};
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;
use nested_vec::NestedVec;

use itertools::iproduct;
use matrixcompare::assert_scalar_eq;
use nalgebra::Point3;

fn create_single_tetrahedron_polymesh() -> PolyMesh3d<f64> {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];
    let faces = NestedVec::from(&vec![
        // TODO: This isn't consistent with respect to winding order etc.
        // We need to introduce the concept of half faces or something similar to
        // make this stuff consistent per cell
        vec![0, 1, 2],
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
    ]);
    let cells = NestedVec::from(&vec![vec![0, 1, 2, 3]]);
    PolyMesh::from_poly_data(vertices, faces, cells)
}

#[test]
fn triangulate_single_tetrahedron_is_unchanged() {
    let mesh = create_single_tetrahedron_polymesh();

    let triangulated = mesh.triangulate().unwrap();

    assert_eq!(triangulated.num_cells(), 1);
    assert_eq!(triangulated.num_faces(), 4);

    // TODO: Further tests!
}

#[test]
fn compute_volume() {
    {
        // Single cube, multiple resolutions
        let unit_lengths = [1.0, 0.5, 1.5];
        let nx = [1, 2, 3];
        let ny = [1, 2, 3];
        let nz = [1, 2, 3];
        let resolutions = [1, 2];

        for (u, nx, ny, nz, res) in iproduct!(&unit_lengths, &nx, &ny, &nz, &resolutions) {
            let cube = create_rectangular_uniform_hex_mesh(*u, *nx, *ny, *nz, *res);
            let cube = PolyMesh3d::from(&cube);
            let expected_volume: f64 = u * u * u * (nx * ny * nz) as f64;
            dbg!(u, nx, ny, nz, res);
            assert_scalar_eq!(cube.compute_volume(), expected_volume, comp = abs, tol = 1e-12);
        }
    }
}

#[test]
fn keep_cells() {
    {
        // Single tetrahedron
        let mesh = create_single_tetrahedron_polymesh();

        // Keep no cells, should give empty mesh
        {
            let kept = mesh.keep_cells(&[]);
            assert_eq!(kept.vertices().len(), 0);
            assert_eq!(kept.num_faces(), 0);
            assert_eq!(kept.num_cells(), 0);
        }

        // Keep cell 0, should give unchanged mesh back
        {
            let kept = mesh.keep_cells(&[0]);
            assert_eq!(mesh, kept);
        }
    }
}
