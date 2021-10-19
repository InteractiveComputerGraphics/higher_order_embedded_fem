use fenris::embedding::embed_mesh_3d;
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;

use fenris::geometry::vtk::write_vtk;
use nalgebra::{Rotation3, Unit, Vector3};

use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let bg_mesh = create_rectangular_uniform_hex_mesh(2.0, 2, 1, 1, 8);

    let embed_mesh = {
        let mut embed_mesh = create_rectangular_uniform_hex_mesh(0.5, 1, 1, 1, 4);
        let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0)), 0.45);
        let t = Vector3::new(0.50, 0.50, 0.50);
        embed_mesh.transform_vertices(|v| *v = rotation * v.clone() + t);
        PolyMesh3d::from(&embed_mesh)
    };

    println!(
        "Embedded mesh: {} cells, {} vertices.",
        embed_mesh.num_cells(),
        embed_mesh.vertices().len()
    );
    println!(
        "Background mesh: {} cells, {} vertices.",
        bg_mesh.connectivity().len(),
        bg_mesh.vertices().len()
    );

    write_vtk(&bg_mesh, "data/embed_mesh_3d/bg_mesh.vtk", "bg mesh")?;
    write_vtk(&embed_mesh, "data/embed_mesh_3d/embed_mesh.vtk", "embedded mesh")?;

    println!("Embedding...");
    let now = Instant::now();
    let embedding = embed_mesh_3d(&bg_mesh, &embed_mesh);
    let elapsed = now.elapsed().as_secs_f64();
    println!("Completed embedding in {:2.2} seconds.", elapsed);

    let exterior_mesh = bg_mesh.keep_cells(&embedding.exterior_cells);
    let interior_mesh = bg_mesh.keep_cells(&embedding.interior_cells);
    let interface_mesh = bg_mesh.keep_cells(&embedding.interface_cells);

    let mut keep_cells = embedding.interior_cells.clone();
    keep_cells.extend(embedding.interface_cells.iter().copied());
    keep_cells.sort_unstable();
    let adapted_bg_mesh = bg_mesh.keep_cells(&keep_cells);
    println!(
        "Adapted bg mesh: {} cells, {} vertices.",
        adapted_bg_mesh.connectivity().len(),
        adapted_bg_mesh.vertices().len()
    );

    write_vtk(&exterior_mesh, "data/embed_mesh_3d/exterior.vtk", "exterior mesh")?;
    write_vtk(&interior_mesh, "data/embed_mesh_3d/interior.vtk", "interior mesh")?;
    write_vtk(&interface_mesh, "data/embed_mesh_3d/interface.vtk", "interface mesh")?;

    let aggregate_interface_mesh = PolyMesh3d::concatenate(&embedding.interface_cell_embeddings);
    write_vtk(
        &aggregate_interface_mesh,
        "data/embed_mesh_3d/aggregate_interface.vtk",
        "aggregate interface",
    )?;

    Ok(())
}
