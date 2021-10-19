use std::error::Error;
use std::fmt;
use std::fmt::Write as FmtWrite;
use std::fs::{create_dir_all, File};
use std::io;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use fenris::connectivity::{Connectivity, Quad4d2Connectivity, Tri3d2Connectivity};
use fenris::geometry::polymesh::{PolyMesh, PolyMesh3d};
use fenris::mesh::{Mesh, Mesh2d, TriangleMesh2d};
use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DefaultAllocator, DimName, Point2, RealField, Scalar, U2, U3};
use fenris::nested_vec::NestedVec;
use hamilton::{StorageContainer, System};
use itertools::Itertools;
use num::ToPrimitive;

use crate::components::{
    get_export_sequence_index, Name, PointInterpolator, PolyMesh2dCollection, PolyMesh3dCollection, VolumeMesh2d,
};
use crate::fem::{FiniteElementElasticModel2d, FiniteElementElasticModel3d, FiniteElementModel2d};

/// A system that writes named 2D volume meshes to PLY files.
#[derive(Debug)]
pub struct PlyVolumeMesh2dOutput {
    pub base_path: PathBuf,
}

/// A system that writes named 2D FEM meshes to PLY files.
#[derive(Debug)]
pub struct PlyFem2dOutput {
    pub base_path: PathBuf,
}

/// A system that writes named from PointInterpolator components to PLY files (as degenerate triangle soups).
#[derive(Debug)]
pub struct PlyInterpolatedPoints2dOutput {
    pub base_path: PathBuf,
}

/*
/// A system that writes named 3D FEM meshes to PLY files.
#[derive(Debug)]
pub struct PlyFem3dOutput {
    pub base_path: PathBuf,
}
*/

/// A system that writes the faces of named PolyMesh2D meshes to PLY files.
#[derive(Debug)]
pub struct PlyPolyMesh2dOutput {
    pub base_path: PathBuf,
}

/// A system that writes the faces of named Polymesh3D meshes to PLY files.
#[derive(Debug)]
pub struct PlyPolyMesh3dOutput {
    pub base_path: PathBuf,
}

impl fmt::Display for PlyVolumeMesh2dOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlyVolumeMesh2dOutput")
    }
}

impl fmt::Display for PlyFem2dOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlyFem2dOutput")
    }
}

impl fmt::Display for PlyInterpolatedPoints2dOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlyInterpolatedPoints2dOutput")
    }
}

/*
impl fmt::Display for PlyFem3dOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlyFem3dOutput")
    }
}
*/

impl fmt::Display for PlyPolyMesh2dOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlyPolyMesh2dOutput")
    }
}

impl fmt::Display for PlyPolyMesh3dOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlyPolyMesh3dOutput")
    }
}

/// Writes the faces stored in a polymesh to a Writer in PLY ASCII format
fn write_polymesh_faces_to_ply_ascii<W, T, D>(writer: &mut W, polymesh: &PolyMesh<T, D>) -> Result<(), Box<dyn Error>>
where
    W: io::Write,
    T: Scalar + ToPrimitive,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    write!(
        writer,
        // TODO: Re-enable export of normals and vertex colors once we properly
        // support this in e.g. PolyMesh3d. The old format is commented out below
        "\
ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_index
end_header
",
        //         "\
        // ply
        // format ascii 1.0
        // element vertex {}
        // property float x
        // property float y
        // property float z
        // property float nx
        // property float ny
        // property float nz
        // property uchar red
        // property uchar green
        // property uchar blue
        // element face {}
        // property list uchar int vertex_index
        // end_header
        // ",
        polymesh.vertices().len(),
        polymesh.num_faces()
    )?;

    // Vertices
    match D::dim() {
        1 => {
            for v in polymesh.vertices() {
                let coords = &v.coords;
                write!(
                    writer,
                    "{} 0.0 0.0\n",
                    // "{} 0.0 0.0 0 0 1 150 0 0\n",
                    coords[0].to_f64().unwrap() as f32
                )?;
            }
        }
        2 => {
            for v in polymesh.vertices() {
                let coords = &v.coords;
                write!(
                    writer,
                    "{} {} 0.0\n",
                    // "{} {} 0.0 0 0 1 150 0 0\n",
                    coords[0].to_f64().unwrap() as f32,
                    coords[1].to_f64().unwrap() as f32
                )?;
            }
        }
        3 => {
            for v in polymesh.vertices() {
                let coords = &v.coords;
                write!(
                    writer,
                    "{} {} {}\n",
                    // "{} {} {} 0 0 1 150 0 0\n",
                    coords[0].to_f64().unwrap() as f32,
                    coords[1].to_f64().unwrap() as f32,
                    coords[2].to_f64().unwrap() as f32
                )?;
            }
        }
        _ => {
            return Err(Box::from(format!(
                "Cannot output vertices of dimension {} into PLY",
                D::dim()
            )));
        }
    }

    // Faces
    for face in polymesh.face_connectivity_iter() {
        let indices = face.iter().map(|v| v.to_string()).join(" ");
        write!(writer, "{} ", face.len())?;
        write!(writer, "{}", indices)?;
        write!(writer, "\n")?;
    }

    Ok(())
}

pub fn dump_polymesh_faces_ply(
    mesh: &PolyMesh3d<f64>,
    base_path: impl AsRef<Path>,
    filename: impl Into<String>,
) -> Result<(), Box<dyn Error>> {
    let filename = filename.into();
    let ply_file_path = base_path.as_ref().join(&filename);

    // Create any necessary parent directories (otherwise creating file will fail)
    if let Some(dir) = ply_file_path.parent() {
        create_dir_all(dir)?;
    }

    let file = File::create(ply_file_path)?;
    let mut ply_writer = io::BufWriter::new(file);
    write_polymesh_faces_to_ply_ascii(&mut ply_writer, mesh)
}

/// Creates a file with the specified filename, appends sequence index, opens it
fn prepare_ply_file_for_writing<P: AsRef<Path>, S: Into<String>>(
    data: &StorageContainer,
    base_path: P,
    filename: S,
) -> Result<File, Box<dyn Error>> {
    // If there is a `StepIndex` component, append the index to the output filename
    let step_index = get_export_sequence_index(data).ok();

    let mut filename = filename.into();
    if let Some(index) = step_index {
        write!(filename, "_{}", index)?;
    }
    write!(filename, ".ply")?;
    let ply_file_path = base_path.as_ref().join(&filename);

    // Create any necessary parent directories (otherwise creating file will fail)
    if let Some(dir) = ply_file_path.parent() {
        create_dir_all(dir)?;
    }

    Ok(File::create(ply_file_path)?)
}

impl System for PlyVolumeMesh2dOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let mesh_storage = data.get_component_storage::<VolumeMesh2d>().borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, mesh) in mesh_storage.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                let ply_file = prepare_ply_file_for_writing(data, &self.base_path, format!("{}_volume2d", name))?;
                let mut ply_writer = io::BufWriter::new(ply_file);

                let polymesh = build_polymesh_from_2d_volume_mesh(mesh)?;
                write_polymesh_faces_to_ply_ascii(&mut ply_writer, &polymesh)?;

                ply_writer.flush()?;
            }
        }

        Ok(())
    }
}

impl System for PlyFem2dOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let fe_models_2d = data
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, fe_model) in fe_models_2d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                let ply_file = prepare_ply_file_for_writing(data, &self.base_path, format!("{}_fe_mesh2d", name))?;
                let mut ply_writer = io::BufWriter::new(ply_file);

                let mut polymesh = build_polymesh_from_2d_model(&fe_model)?;
                interpolate_points_from_2d_model(polymesh.vertices_mut(), &fe_model)?;
                write_polymesh_faces_to_ply_ascii(&mut ply_writer, &polymesh)?;

                ply_writer.flush()?;
            }
        }

        Ok(())
    }
}

impl System for PlyInterpolatedPoints2dOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let interpolators_2d = data.get_component_storage::<PointInterpolator>().borrow();
        let fe_models_2d = data
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, pi) in interpolators_2d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                if let Some(model) = fe_models_2d.get_component(*id) {
                    // Interpolate points with deformed mesh
                    let interpolated_points: Vec<Point2<_>> = pi
                        .interpolator
                        .interpolate::<U2>(&model.u)
                        .into_iter()
                        .zip(pi.reference_points.iter())
                        .map(|(u, p0)| (p0.coords + u).into())
                        .collect();

                    let ply_file = prepare_ply_file_for_writing(data, &self.base_path, format!("{}_points", name))?;
                    let mut ply_writer = io::BufWriter::new(ply_file);

                    let pointcloud_mesh = build_degenerate_pointcloud_mesh_2d(interpolated_points.as_slice());
                    let polymesh = Mesh::try_clone_face_soup_from_mesh(&pointcloud_mesh)?;
                    write_polymesh_faces_to_ply_ascii(&mut ply_writer, &polymesh)?;

                    ply_writer.flush()?;
                }
            }
        }

        Ok(())
    }
}

/*
impl System for PlyFem3dOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let fe_models_3d = data
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, fe_model) in fe_models_3d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                let ply_file = prepare_ply_file_for_writing(
                    data,
                    &self.base_path,
                    format!("{}_fe_mesh3d", name),
                )?;
                let mut ply_writer = io::BufWriter::new(ply_file);

                let mut polymesh = build_polymesh_from_3d_model(&fe_model)?;
                interpolate_points_from_3d_model(polymesh.vertices_mut(), &fe_model)?;
                write_polymesh_faces_to_ply_ascii(&mut ply_writer, &polymesh)?;

                ply_writer.flush()?;
            }
        }

        Ok(())
    }
}
*/

impl System for PlyPolyMesh2dOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let polymeshes_2d = data
            .get_component_storage::<PolyMesh2dCollection>()
            .borrow();
        let fe_models_2d = data
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, p2d_collection) in polymeshes_2d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                for p2d_component in p2d_collection.iter() {
                    // Optionally add subfolder to base output path
                    let mesh_path = if let Some(subfolder) = &p2d_component.subfolder {
                        self.base_path.join(subfolder)
                    } else {
                        self.base_path.clone()
                    };

                    // Recursively create folders and open file for writing
                    let ply_file = prepare_ply_file_for_writing(
                        data,
                        &mesh_path,
                        format!("{}_{}_polymesh", name, p2d_component.mesh_name),
                    )?;

                    if let Some(interpolator) = &p2d_component.interpolator {
                        let mut polymesh = p2d_component.mesh.clone();

                        // Try to get FE model and interpolate vertices of mesh
                        if let Some(model) = fe_models_2d.get_component(*id) {
                            let u_interpolated = interpolator.interpolate::<U2>(&model.u);

                            // Apply deformation to mesh
                            for (v, u) in polymesh
                                .vertices_mut()
                                .iter_mut()
                                .zip(u_interpolated.iter())
                            {
                                v.coords += u;
                            }
                        };

                        let mut ply_writer = io::BufWriter::new(ply_file);
                        write_polymesh_faces_to_ply_ascii(&mut ply_writer, &polymesh)?;
                        ply_writer.flush()?;
                    } else {
                        let mut ply_writer = io::BufWriter::new(ply_file);
                        write_polymesh_faces_to_ply_ascii(&mut ply_writer, &p2d_component.mesh)?;
                        ply_writer.flush()?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl System for PlyPolyMesh3dOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let polymeshes_3d = data
            .get_component_storage::<PolyMesh3dCollection>()
            .borrow();
        let fe_models_3d = data
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, p3d_collection) in polymeshes_3d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                for p3d_component in p3d_collection.iter() {
                    // Optionally add subfolder to base output path
                    let mesh_path = if let Some(subfolder) = &p3d_component.subfolder {
                        self.base_path.join(subfolder)
                    } else {
                        self.base_path.clone()
                    };

                    // Recursively create folders and open file for writing
                    let ply_file = prepare_ply_file_for_writing(
                        data,
                        &mesh_path,
                        format!("{}_{}_polymesh", name, p3d_component.mesh_name),
                    )?;

                    if let Some(interpolator) = &p3d_component.interpolator {
                        let mut polymesh = p3d_component.mesh.clone();

                        // Try to get FE model and interpolate vertices of mesh
                        if let Some(model) = fe_models_3d.get_component(*id) {
                            let u_interpolated = interpolator.interpolate::<U3>(&model.u);

                            // Apply deformation to mesh
                            for (v, u) in polymesh
                                .vertices_mut()
                                .iter_mut()
                                .zip(u_interpolated.iter())
                            {
                                v.coords += u;
                            }
                        };

                        let mut ply_writer = io::BufWriter::new(ply_file);
                        write_polymesh_faces_to_ply_ascii(&mut ply_writer, &polymesh)?;
                        ply_writer.flush()?;
                    } else {
                        let mut ply_writer = io::BufWriter::new(ply_file);
                        write_polymesh_faces_to_ply_ascii(&mut ply_writer, &p3d_component.mesh)?;
                        ply_writer.flush()?;
                    }
                }
            }
        }

        Ok(())
    }
}

trait TryPolygonalizeFaceSoupFromMesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    fn try_polygonalize_face_soup_from_mesh(mesh: &Mesh<T, D, C>) -> Result<PolyMesh<T, D>, Box<dyn Error>> {
        Ok(PolyMesh::from_surface_mesh(mesh))
    }
}

impl<T, D, C> TryPolygonalizeFaceSoupFromMesh<T, D, C> for Mesh<T, D, C>
where
    T: RealField,
    D: DimName,
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
}

trait TryCloneFaceSoupFromMesh<T, D, C>
where
    T: RealField,
    D: DimName,
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    fn try_clone_face_soup_from_mesh(mesh: &Mesh<T, D, C>) -> Result<PolyMesh<T, D>, Box<dyn Error>> {
        let vertices = mesh.vertices().to_vec();
        let mut faces = NestedVec::new();
        let cells = NestedVec::new();

        for cell in mesh.connectivity() {
            faces.push(cell.vertex_indices());
        }

        Ok(PolyMesh::from_poly_data(vertices, faces, cells))
    }
}

impl<T: RealField> TryCloneFaceSoupFromMesh<T, U2, Tri3d2Connectivity> for Mesh2d<T, Tri3d2Connectivity> {}

impl<T: RealField> TryCloneFaceSoupFromMesh<T, U2, Quad4d2Connectivity> for Mesh2d<T, Quad4d2Connectivity> {}

/// Creates a PolyMesh from the FEM mesh of a FiniteElementElasticModel2d
#[rustfmt::skip]
fn build_polymesh_from_2d_model(
    elastic_model: &FiniteElementElasticModel2d,
) -> Result<PolyMesh<f64, U2>, Box<dyn Error>> {
    use FiniteElementModel2d::*;
    match elastic_model.model {
        Tri3d2NodalModel(ref model) => Mesh::try_clone_face_soup_from_mesh(model.mesh()),
        Tri6d2NodalModel(ref model) => Mesh::try_polygonalize_face_soup_from_mesh(model.mesh()),
        Quad4NodalModel(ref model) => Mesh::try_clone_face_soup_from_mesh(model.mesh()),
        Quad9NodalModel(ref model) => Mesh::try_polygonalize_face_soup_from_mesh(model.mesh()),
        EmbeddedTri3d2Model(ref model) => Mesh::try_clone_face_soup_from_mesh(model.background_mesh()),
        EmbeddedTri6d2Model(ref model) => Mesh::try_polygonalize_face_soup_from_mesh(model.background_mesh()),
        EmbeddedQuad4Model(ref model) => Mesh::try_clone_face_soup_from_mesh(model.background_mesh()),
        EmbeddedQuad9Model(ref model) => Mesh::try_polygonalize_face_soup_from_mesh(model.background_mesh()),
    }
}

/// Creates a PolyMesh from a VolumeMesh2d
#[rustfmt::skip]
fn build_polymesh_from_2d_volume_mesh(
    volume_mesh: &VolumeMesh2d,
) -> Result<PolyMesh<f64, U2>, Box<dyn Error>> {
    match volume_mesh {
        VolumeMesh2d::QuadMesh(mesh) => Mesh::try_clone_face_soup_from_mesh(mesh),
        VolumeMesh2d::TriMesh(mesh) => Mesh::try_clone_face_soup_from_mesh(mesh),
    }
}

/// Interpolate deformation of a FiniteElementElasticModel2d onto a list of points
fn interpolate_points_from_2d_model(
    vertices: &mut [Point2<f64>],
    elastic_model: &FiniteElementElasticModel2d,
) -> Result<(), Box<dyn Error>> {
    let interpolator = match_on_finite_element_model_2d!(elastic_model.model, model => {
        model.make_interpolator(vertices)?
    });
    let u_interpolated = interpolator.interpolate::<U2>(&elastic_model.u);
    for (v, u) in vertices.iter_mut().zip(u_interpolated.iter()) {
        v.coords += u;
    }

    Ok(())
}

/// Creates a degenerate triangle soup from a point cloud
fn build_degenerate_pointcloud_mesh_2d(points: &[Point2<f64>]) -> TriangleMesh2d<f64> {
    let mut vertices = Vec::with_capacity(points.len() * 3);
    for p in points {
        vertices.push(p.clone());
        vertices.push(p.clone());
        vertices.push(p.clone());
    }

    let mut connectivity = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        connectivity.push(Tri3d2Connectivity([3 * i + 0, 3 * i + 1, 3 * i + 2]));
    }

    TriangleMesh2d::from_vertices_and_connectivity(vertices, connectivity)
}

/*
/// Creates a PolyMesh from the FEM mesh of a FiniteElementElasticModel2d
#[rustfmt::skip]
fn build_polymesh_from_3d_model(
    elastic_model: &FiniteElementElasticModel3d,
) -> Result<PolyMesh<f64, U3>, Box<dyn Error>> {
    use FiniteElementModel3d::*;
    Ok(match elastic_model.model {
        Hex8NodalModel(ref model) => PolyMesh::from(model.mesh()),
        Tet4NodalModel(ref model) => PolyMesh::from(model.mesh()),
        Tet10NodalModel(ref model) => PolyMesh::from(model.mesh()),
        EmbeddedHex8Model(ref model) => PolyMesh::from(model.background_mesh()),
        EmbeddedTet4Model(ref model) => PolyMesh::from(model.background_mesh()),
        EmbeddedTet10Model(ref model) => PolyMesh::from(model.background_mesh()),
    })
}
*/

/*
/// Interpolate deformation of a FiniteElementElasticModel3d onto a list of points
fn interpolate_points_from_3d_model(
    vertices: &mut [Point3<f64>],
    elastic_model: &FiniteElementElasticModel3d,
) -> Result<(), Box<dyn Error>> {
    let interpolator = match_on_finite_element_model_3d!(elastic_model.model, model => {
        model.make_interpolator(vertices)?
    });
    let u_interpolated = interpolator.interpolate::<U3>(&elastic_model.u);
    for (v, u) in vertices.iter_mut().zip(u_interpolated.iter()) {
        v.coords += u;
    }

    Ok(())
}
*/
