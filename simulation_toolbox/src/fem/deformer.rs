use crate::components::{SurfaceMesh2d, VolumeMesh2d, VolumeMesh3d};
use crate::fem::{FiniteElementElasticModel2d, FiniteElementElasticModel3d};
use crate::util::apply_displacements;
use fenris::nalgebra::{U2, U3};
use fenris::space::FiniteElementSpace;
use hamilton::{StorageContainer, System};
use std::error::Error;
use std::fmt;
use std::fmt::Display;

#[derive(Debug)]
pub struct FiniteElementMeshDeformer;

impl Display for FiniteElementMeshDeformer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FiniteElementMeshDeformer")
    }
}

impl System for FiniteElementMeshDeformer {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        // 2D
        {
            let fe_models = data
                .get_component_storage::<FiniteElementElasticModel2d>()
                .borrow();
            let mut volume_meshes = data.get_component_storage::<VolumeMesh2d>().borrow_mut();
            let mut surface_meshes = data.get_component_storage::<SurfaceMesh2d>().borrow_mut();

            for (id, model) in fe_models.entity_component_iter() {
                // Volume mesh
                if let Some(mesh) = volume_meshes.get_component_mut(*id) {
                    let interpolator = &model.material_volume_interpolator;
                    let n_vertices = model.material_volume_mesh.vertices().len();

                    if n_vertices == mesh.vertices().len() {
                        // TODO: Reuse vector instead of re-allocating on every `run` invocation
                        let displacements = interpolator.interpolate::<U2>(&model.u);
                        apply_displacements(
                            mesh.vertices_mut(),
                            model.material_volume_mesh.vertices(),
                            &displacements,
                        );
                    } else {
                        return Err(Box::from(
                            "Reference mesh and deformed mesh have incompatible vertex counts.",
                        ));
                    }
                }

                // Surface mesh
                if let Some(surface_mesh) = surface_meshes.get_component_mut(*id) {
                    let ref_surface = &model.material_surface;
                    let interpolator = &model.material_surface_interpolator;

                    if let (Some(ref_surface), Some(interpolator)) = (ref_surface, interpolator) {
                        let n_vertices = FiniteElementSpace::vertices(ref_surface).len();
                        if n_vertices == surface_mesh.vertices().len() {
                            // TODO: Reuse vector instead of re-allocating on every `run` invocation
                            let displacements = interpolator.interpolate::<U2>(&model.u);
                            surface_mesh.transform_all_vertices(|vertices_mut| {
                                apply_displacements(
                                    vertices_mut,
                                    FiniteElementSpace::vertices(ref_surface),
                                    &displacements,
                                );
                            });
                        } else {
                            return Err(Box::from(
                                "Reference and deformed surface meshes have incompatible vertex counts",
                            ));
                        }
                    }
                }
            }
        }

        // 3D
        {
            let fe_models = data
                .get_component_storage::<FiniteElementElasticModel3d>()
                .borrow();
            let mut volume_meshes = data.get_component_storage::<VolumeMesh3d>().borrow_mut();

            for (id, model) in fe_models.entity_component_iter() {
                // Volume mesh
                if let Some(mesh) = volume_meshes.get_component_mut(*id) {
                    let interpolator = &model.material_volume_interpolator;
                    let n_vertices = model.material_volume_mesh.vertices().len();

                    if n_vertices == mesh.vertices().len() {
                        // TODO: Reuse vector instead of re-allocating on every `run` invocation
                        let displacements = interpolator.interpolate::<U3>(&model.u);
                        apply_displacements(
                            mesh.vertices_mut(),
                            model.material_volume_mesh.vertices(),
                            &displacements,
                        );
                    } else {
                        return Err(Box::from(
                            "Reference mesh and deformed mesh have incompatible vertex counts.",
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}
