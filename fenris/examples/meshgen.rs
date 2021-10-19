use fenris::geometry::procedural::{
    approximate_quad_mesh_for_sdf_2d, approximate_triangle_mesh_for_sdf_2d, voxelize_sdf_2d,
};
use fenris::geometry::sdf::SdfCircle;
use fenris::geometry::vtk::{
    create_vtk_data_set_from_polygons, create_vtk_data_set_from_quad_mesh, create_vtk_data_set_from_triangle_mesh,
    write_vtk,
};

use fenris::embedding::embed_mesh_2d;
use nalgebra::Vector2;
use vtkio::Error;

pub fn main() -> Result<(), Error> {
    let voxelize_resolution = 0.23;
    let fitted_resolution = 0.1;
    let sdf = SdfCircle {
        radius: 1.0,
        center: Vector2::zeros(),
    };

    let voxelized_mesh = voxelize_sdf_2d(&sdf, voxelize_resolution);
    let voxelized_data_set = create_vtk_data_set_from_quad_mesh(&voxelized_mesh);
    write_vtk(
        voxelized_data_set,
        "data/circle_mesh_voxelized.vtk",
        "data/voxelized circle",
    )?;

    let fitted_quad_mesh = approximate_quad_mesh_for_sdf_2d(&sdf, fitted_resolution);
    let fitted_quad_mesh_data_set = create_vtk_data_set_from_quad_mesh(&fitted_quad_mesh);
    write_vtk(
        fitted_quad_mesh_data_set,
        "data/circle_mesh_fitted.vtk",
        "data/fitted circle quad mesh",
    )?;

    let fitted_triangle_mesh = approximate_triangle_mesh_for_sdf_2d(&sdf, fitted_resolution);
    let fitted_triangle_mesh_data_set = create_vtk_data_set_from_triangle_mesh(&fitted_triangle_mesh);
    write_vtk(
        fitted_triangle_mesh_data_set,
        "data/circle_mesh_triangle_fitted.vtk",
        "data/fitted circle triangle mesh",
    )?;

    let embedding = embed_mesh_2d(&voxelized_mesh, &fitted_triangle_mesh).unwrap();
    let polygons = embedding
        .into_iter()
        .map(|(_, polygons)| polygons)
        .flatten()
        .collect::<Vec<_>>();
    let embedded_data_set = create_vtk_data_set_from_polygons(&polygons);
    write_vtk(
        embedded_data_set,
        "data/embedded_mesh.vtk",
        "data/triangle mesh embedded in quad mesh",
    )?;

    Ok(())
}
