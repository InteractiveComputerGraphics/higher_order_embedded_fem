use crate::components::{get_export_sequence_index, Name, SurfaceMesh2d, VolumeMesh2d, VolumeMesh3d};
use crate::fem::{
    FiniteElementElasticModel2d, FiniteElementElasticModel3d, FiniteElementModel2d, FiniteElementModel3d,
};
use fenris::connectivity::Segment2d2Connectivity;
use fenris::geometry::vtk::write_vtk;
use fenris::mesh::Mesh2d;
use fenris::nalgebra::DVector;
use fenris::solid::ElasticityModel;
use fenris::vtkio::model::{Attribute, DataSet};
use fenris::vtkio::IOBuffer;
use hamilton::{StorageContainer, System};
use log::warn;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::path::PathBuf;

/// A system that writes named volume meshes to VTK files.
#[derive(Debug)]
pub struct VtkVolumeMeshOutput {
    pub base_path: PathBuf,
}

/// A system that writes named surface meshes to VTK files.
#[derive(Debug)]
pub struct VtkSurfaceMeshOutput {
    pub base_path: PathBuf,
}

#[derive(Debug)]
pub struct VtkFemOutput {
    pub base_path: PathBuf,
}

impl Display for VtkVolumeMeshOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtkVolumeMeshOutput")
    }
}

impl Display for VtkSurfaceMeshOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtkSurfaceMeshOutput")
    }
}

impl Display for VtkFemOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtkFemOutput")
    }
}

impl System for VtkVolumeMeshOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        // If there is a `StepIndex` component, append the index to the output filename
        let step_index = get_export_sequence_index(data).ok();
        let mesh2d_storage = data.get_component_storage::<VolumeMesh2d>().borrow();
        let mesh3d_storage = data.get_component_storage::<VolumeMesh3d>().borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, mesh) in mesh2d_storage.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                let filename = create_vtk_filename(&name.0, "volume_mesh", step_index);
                let vtk_file_path = self.base_path.join(filename);

                let dataset = match mesh {
                    VolumeMesh2d::QuadMesh(ref mesh) => DataSet::from(mesh),
                    VolumeMesh2d::TriMesh(ref mesh) => DataSet::from(mesh),
                };

                write_vtk(dataset, vtk_file_path, &name.0)?;
            }
        }

        for (id, mesh) in mesh3d_storage.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                if mesh2d_storage.get_component(*id).is_some() {
                    warn!(
                        "Component has both 2D and 3D volume mesh.\
                         2D volume mesh output will be overwritten."
                    );
                }

                let filename = create_vtk_filename(&name.0, "volume_mesh", step_index);
                let vtk_file_path = self.base_path.join(filename);

                let dataset = DataSet::from(&mesh.0);
                write_vtk(dataset, vtk_file_path, &name.0)?;
            }
        }

        Ok(())
    }
}

/// Creates a mesh containing line segments corresponding to the normals of the surface mesh,
/// with normals placed at the midpoint of the segments.
fn create_normals_mesh(surface_mesh: &Mesh2d<f64, Segment2d2Connectivity>) -> Mesh2d<f64, Segment2d2Connectivity> {
    let mut vertices = Vec::new();
    let mut connectivity = Vec::new();

    for cell in surface_mesh.cell_iter() {
        let begin_idx = vertices.len();
        let end_idx = begin_idx + 1;
        let midpoint = cell.point_from_parameter(0.5);

        let normal_begin = midpoint;
        let normal_end = midpoint + cell.normal_dir().normalize();
        vertices.push(normal_begin);
        vertices.push(normal_end);
        connectivity.push(Segment2d2Connectivity([begin_idx, end_idx]));
    }

    Mesh2d::from_vertices_and_connectivity(vertices, connectivity)
}

fn create_vtk_filename(object_name: &str, tag: &str, step_index: Option<isize>) -> String {
    format!(
        "{obj}_{tag}{step_index}.vtk",
        obj = object_name,
        tag = tag,
        step_index = step_index
            .map(|i| format!("_{}", i))
            .unwrap_or(String::new())
    )
}

impl System for VtkSurfaceMeshOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        // If there is a `StepIndex` component, append the index to the output filename
        let step_index = get_export_sequence_index(data).ok();
        let mesh_storage = data.get_component_storage::<SurfaceMesh2d>().borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, mesh) in mesh_storage.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                // Surface
                let surface_filename = create_vtk_filename(&name.0, "surface_mesh", step_index);
                let surface_vtk_file_path = self.base_path.join(surface_filename);
                let surface_dataset = DataSet::from(&mesh.0);
                write_vtk(
                    surface_dataset,
                    surface_vtk_file_path,
                    &format!("{} surface mesh", &name.0),
                )?;

                // Export normals
                let normals_filename = create_vtk_filename(&name.0, "normals", step_index);
                let normals_vtk_file_path = self.base_path.join(normals_filename);
                let normals_dataset = DataSet::from(&create_normals_mesh(&mesh.0));
                write_vtk(normals_dataset, normals_vtk_file_path, &format!("{} normals", &name.0))?;
            }
        }

        Ok(())
    }
}

fn create_fem_dataset_2d(model: &FiniteElementModel2d) -> DataSet {
    use FiniteElementModel2d::*;
    match model {
        Tri3d2NodalModel(ref model) => DataSet::from(model.mesh()),
        Tri6d2NodalModel(ref model) => DataSet::from(model.mesh()),
        Quad4NodalModel(ref model) => DataSet::from(model.mesh()),
        Quad9NodalModel(ref model) => DataSet::from(model.mesh()),
        EmbeddedTri3d2Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedTri6d2Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedQuad4Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedQuad9Model(ref model) => DataSet::from(model.background_mesh()),
    }
}

fn create_fem_dataset_3d(model: &FiniteElementModel3d) -> DataSet {
    use FiniteElementModel3d::*;
    match model {
        Hex8NodalModel(ref model) => DataSet::from(model.mesh()),
        Hex20NodalModel(ref model) => DataSet::from(model.mesh()),
        Hex27NodalModel(ref model) => DataSet::from(model.mesh()),
        Tet4NodalModel(ref model) => DataSet::from(model.mesh()),
        Tet10NodalModel(ref model) => DataSet::from(model.mesh()),
        EmbeddedHex8Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedHex20Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedHex27Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedTet4Model(ref model) => DataSet::from(model.background_mesh()),
        EmbeddedTet10Model(ref model) => DataSet::from(model.background_mesh()),
    }
}

impl System for VtkFemOutput {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        // If there is a `StepIndex` component, append the index to the output filename
        let step_index = get_export_sequence_index(data).ok();
        let fe_models_2d = data
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow();
        let fe_models_3d = data
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow();
        let name_storage = data.get_component_storage::<Name>().borrow();

        for (id, fe_model) in fe_models_2d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                let mesh_filename = create_vtk_filename(&name.0, "fe_mesh", step_index);
                let mesh_vtk_file_path = self.base_path.join(mesh_filename);
                let mut mesh_dataset = create_fem_dataset_2d(&fe_model.model);

                if let DataSet::UnstructuredGrid { ref mut data, .. } = mesh_dataset {
                    // Displacements are 2d, but Paraview needs 3d points
                    let num_dofs_3d = 3 * fe_model.model.ndof() / 2;
                    let displacements_3d = DVector::from_fn(num_dofs_3d, |i, _| {
                        let node_index = i / 3;
                        let local_index = i % 3;
                        if local_index < 2 {
                            fe_model.u[2 * node_index + local_index]
                        } else {
                            0.0
                        }
                    });

                    let displacement_buffer = IOBuffer::from_slice(displacements_3d.as_slice());
                    let attribute = Attribute::Vectors {
                        data: displacement_buffer,
                    };
                    data.point.push((format!("displacement"), attribute));
                }

                write_vtk(
                    mesh_dataset,
                    mesh_vtk_file_path,
                    &format!("{} Finite Element mesh", &name.0),
                )?;
            }
        }

        for (id, fe_model) in fe_models_3d.entity_component_iter() {
            if let Some(name) = name_storage.get_component(*id) {
                if let Some(_) = fe_models_2d.get_component(*id) {
                    warn!("Entity has both 2d and 3d FEM model. Naming conflict for output.");
                }

                let mesh_filename = create_vtk_filename(&name.0, "fe_mesh", step_index);
                let mesh_vtk_file_path = self.base_path.join(mesh_filename);
                let mut mesh_dataset = create_fem_dataset_3d(&fe_model.model);

                if let DataSet::UnstructuredGrid { ref mut data, .. } = mesh_dataset {
                    let displacement_buffer = IOBuffer::from_slice(fe_model.u.as_slice());
                    let attribute = Attribute::Vectors {
                        data: displacement_buffer,
                    };
                    data.point.push((format!("displacement"), attribute));
                }

                write_vtk(
                    mesh_dataset,
                    mesh_vtk_file_path,
                    &format!("{} Finite Element mesh", &name.0),
                )?;
            }
        }

        Ok(())
    }
}
