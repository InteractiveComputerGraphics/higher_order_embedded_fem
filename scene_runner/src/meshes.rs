use std::error::Error;
use std::fs::OpenOptions;
use std::io::{BufReader, Read};
use std::path::Path;

use fenris::connectivity::Connectivity;
use fenris::mesh::{Mesh, TriangleMesh2d};
use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DefaultAllocator, DimName, Point};
use once_cell::sync::Lazy;
use simulation_toolbox::io::msh::{try_mesh_from_bytes, TryConnectivityFromMshElement, TryVertexFromMshNode};

fn read_bytes<P>(path: P) -> Result<Vec<u8>, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let file = OpenOptions::new()
        .read(true)
        .write(false)
        .create(false)
        .open(path)?;
    let mut buf_reader = BufReader::new(file);

    let mut data = Vec::new();
    buf_reader.read_to_end(&mut data)?;
    Ok(data)
}

fn load_mesh_from_file_internal<P, D, C>(asset_dir: P, file_name: &str) -> Result<Mesh<f64, D, C>, Box<dyn Error>>
where
    P: AsRef<Path>,
    D: DimName,
    C: Connectivity + TryConnectivityFromMshElement<C>,
    Point<f64, D>: TryVertexFromMshNode<f64, D, f64>,
    DefaultAllocator: Allocator<f64, D>,
{
    let file_path = asset_dir.as_ref().join(file_name);
    let msh_bytes = read_bytes(file_path)?;
    try_mesh_from_bytes(&msh_bytes)
}

pub fn load_mesh_from_file<P, D, C>(asset_dir: P, file_name: &str) -> Result<Mesh<f64, D, C>, Box<dyn Error>>
where
    P: AsRef<Path>,
    D: DimName,
    C: Connectivity + TryConnectivityFromMshElement<C>,
    Point<f64, D>: TryVertexFromMshNode<f64, D, f64>,
    DefaultAllocator: Allocator<f64, D>,
{
    load_mesh_from_file_internal(asset_dir, file_name)
        .map_err(|e| Box::from(format!("Error occured during mesh loading of `{}`: {}", file_name, e)))
}

pub(crate) static BIKE_TRI2D_MESH_COARSE: Lazy<TriangleMesh2d<f64>> = Lazy::new(|| {
    load_mesh_from_file("assets", "meshes/bike_coarse.obj_linear.msh").expect("Unable to load 'BIKE_MSH_COARSE'")
});

pub(crate) static BIKE_TRI2D_MESH_FINE: Lazy<TriangleMesh2d<f64>> = Lazy::new(|| {
    load_mesh_from_file("assets", "meshes/bike_fine.obj_linear.msh").expect("Unable to load 'BIKE_MSH_FINE'")
});

pub(crate) static ELEPHANT_TRI2D_MESH_COARSE: Lazy<TriangleMesh2d<f64>> = Lazy::new(|| {
    load_mesh_from_file("assets", "meshes/elephant_base.obj_coarse.msh").expect("Unable to load 'ELEPHANT_MSH_COARSE'")
});

pub(crate) static ELEPHANT_TRI2D_MESH_FINE: Lazy<TriangleMesh2d<f64>> = Lazy::new(|| {
    load_mesh_from_file("assets", "meshes/elephant_base.obj_fine.msh").expect("Unable to load 'ELEPHANT_MSH_FINE'")
});

pub(crate) static ELEPHANT_TRI2D_MESH_SUPER_FINE: Lazy<TriangleMesh2d<f64>> = Lazy::new(|| {
    load_mesh_from_file("assets", "meshes/elephant_base.obj_super_fine.msh")
        .expect("Unable to load 'ELEPHANT_MSH_SUPER_FINE'")
});

pub(crate) static ELEPHANT_CAGE_TRI2D_MESH_FINE: Lazy<TriangleMesh2d<f64>> = Lazy::new(|| {
    load_mesh_from_file("assets", "meshes/elephant_cage.obj_coarse.msh").expect("Unable to load 'ELEPHANT_MSH_CAGE'")
});
