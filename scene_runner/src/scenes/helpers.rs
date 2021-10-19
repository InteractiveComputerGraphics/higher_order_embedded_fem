use std::error::Error;
use std::mem;
use std::path::{Path, PathBuf};

use fenris::allocators::ElementConnectivityAllocator;
use fenris::element::ElementConnectivity;
use fenris::embedding::{find_background_cell_indices_2d, EmbeddedModel};
use fenris::geometry::procedural::{create_rectangular_uniform_quad_mesh_2d, voxelize_bounding_box_2d};
use fenris::geometry::vtk::{create_vtk_data_set_from_quadratures, write_vtk};
use fenris::geometry::{AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, BoundedGeometry};
use fenris::mesh::{ClosedSurfaceMesh2d, QuadMesh2d, TriangleMesh2d};
use fenris::model::FiniteElementInterpolator;
use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{
    convert, try_convert, DVector, DefaultAllocator, DimMin, DimName, Point, RealField, Vector2, Vector3, VectorN,
};
use fenris::rtree::GeometryCollectionAccelerator;
use fenris::solid::ElasticityModel;
use fenris::util::flatten_vertically;
use hamilton::{register_component, BijectiveStorageMut, Entity, StorageContainer};
use numeric_literals::replace_float_literals;
use simulation_toolbox::components::{
    Name, PointInterpolator, PolyMesh3dCollection, SimulationTime, StepIndex, SurfaceMesh2d, TimeStep, VolumeMesh2d,
    VolumeMesh3d,
};
use simulation_toolbox::fem::bcs::{Empty, Homogeneous, Union};
use simulation_toolbox::fem::{
    DirichletBoundaryConditionComponent, DirichletBoundaryConditions, FiniteElementElasticModel2d,
    FiniteElementElasticModel3d, FiniteElementModel2d, FiniteElementModel3d, IntegrationMethod, Material,
};
use simulation_toolbox::{match_on_finite_element_model_2d, match_on_finite_element_model_3d};

pub fn register_known_components() -> Result<(), Box<dyn Error>> {
    register_component::<Name>()?;
    register_component::<FiniteElementElasticModel2d>()?;
    register_component::<VolumeMesh2d>()?;
    register_component::<VolumeMesh3d>()?;
    register_component::<SurfaceMesh2d>()?;
    register_component::<PointInterpolator>()?;
    register_component::<PolyMesh3dCollection>()?;
    register_component::<TimeStep>()?;
    register_component::<StepIndex>()?;
    register_component::<SimulationTime>()?;
    register_component::<FiniteElementElasticModel3d>()?;
    register_component::<DirichletBoundaryConditionComponent>()?;

    Ok(())
}

/// Provides utility functions for point clouds
pub struct PointHelper<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    phantom: std::marker::PhantomData<D>,
}

impl<D> PointHelper<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    pub fn bb(points: &[Point<f64, D>]) -> Option<AxisAlignedBoundingBox<f64, D>> {
        AxisAlignedBoundingBox::from_points(points)
    }

    /// Uniformly scales the point cloud with the given scaling factor
    pub fn scale(points: &mut [Point<f64, D>], scaling: f64) {
        for p in points.iter_mut() {
            p.coords *= scaling;
        }
    }

    /// Uniformly scales the point cloud such that the max extent of its bounding box corresponds to the specified value, returns scaling factor
    pub fn scale_max_extent_to(points: &mut [Point<f64, D>], target_extent: f64) -> Option<f64> {
        Self::bb(points).map(|bb| {
            let max_extent = bb.max_extent();
            let scaling = target_extent / max_extent;
            Self::scale(points, scaling);
            scaling
        })
    }

    /// Translates the point cloud by the vector
    pub fn translate(points: &mut [Point<f64, D>], translation: &VectorN<f64, D>) {
        for p in points.iter_mut() {
            p.coords += translation;
        }
    }

    /// Translates the point cloud's bounding box center to the origin, returns the translation
    pub fn center_to_origin(points: &mut [Point<f64, D>]) -> Option<VectorN<f64, D>> {
        Self::bb(points).map(|bb| {
            let c = bb.center();
            let dx = -c.coords;
            Self::translate(points, &dx);
            dx
        })
    }
}

/// Generates a 2d quad background mesh to embed a 2d triangle mesh
pub fn generate_background_mesh_for_tri2d<T>(
    triangle_mesh: &TriangleMesh2d<T>,
    background_resolution: T,
) -> Result<QuadMesh2d<T>, Box<dyn std::error::Error>>
where
    T: RealField,
{
    let voxel_mesh = voxelize_bounding_box_2d(&triangle_mesh.bounding_box(), background_resolution);

    // Make sure to remove background cells that don't intersect the embedded mesh at all
    let indices = find_background_cell_indices_2d(&voxel_mesh, &triangle_mesh)?;
    Ok(voxel_mesh.keep_cells(&indices))
}

/// Generates a 2d quad background mesh to embed a 2d triangle mesh with just one cell
#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn generate_single_cell_background_mesh_for_tri2d<T>(
    triangle_mesh: &TriangleMesh2d<T>,
) -> Result<QuadMesh2d<T>, Box<dyn std::error::Error>>
where
    T: RealField,
{
    let bounds = triangle_mesh.bounding_box();
    let extents = bounds.extents();
    let enlarged_bounds = AxisAlignedBoundingBox2d::new(bounds.min() - extents * 0.01, bounds.max() + extents * 0.01);
    let enlarged_extents = enlarged_bounds.extents();

    let enlarged_extents_f64: Vector2<f64> = try_convert(enlarged_extents).expect("Must be able to fit extents in f64");

    let cell_size_f64 = enlarged_extents_f64.x.max(enlarged_extents_f64.y);
    let cell_size: T = convert(cell_size_f64);

    let center = bounds.center();
    let top_left = Vector2::new(center.x - cell_size / 2.0, center.y + cell_size / 2.0);

    Ok(create_rectangular_uniform_quad_mesh_2d(
        convert(cell_size),
        1,
        1,
        1,
        &top_left,
    ))
}

#[allow(dead_code)]
pub fn dump_embedded_interface_quadrature<T, D, C>(
    model: &EmbeddedModel<T, D, C>,
    path: impl AsRef<Path>,
    title: impl AsRef<str>,
) -> Result<(), Box<dyn Error>>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    C: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    DefaultAllocator: Allocator<T, D> + ElementConnectivityAllocator<T, C>,
{
    let path: &Path = path.as_ref();
    let stem = path
        .file_stem()
        .ok_or_else(|| Box::<dyn Error>::from("Path must be a file path"))?;

    let dump_rules = |name, quadrature| -> Result<(), Box<dyn Error>> {
        let dataset =
            create_vtk_data_set_from_quadratures(model.vertices(), model.interface_connectivity(), quadrature);
        let mut out_file_name = stem.to_os_string();
        out_file_name.push("_");
        out_file_name.push(name);
        if let Some(extension) = path.extension() {
            out_file_name.push(".");
            out_file_name.push(extension);
        }
        let out_path = if let Some(parent) = path.parent() {
            parent.join(&out_file_name)
        } else {
            PathBuf::from(out_file_name)
        };
        write_vtk(dataset, &out_path, title.as_ref())?;
        Ok(())
    };

    if let Some(mass_quadrature) = model.mass_quadrature() {
        dump_rules("mass", mass_quadrature.interface_quadratures())?;
    }

    if let Some(stiffness_quadrature) = model.stiffness_quadrature() {
        dump_rules("stiffness", stiffness_quadrature.interface_quadratures())?;
    }

    if let Some(elliptic_quadrature) = model.elliptic_quadrature() {
        dump_rules("elliptic", elliptic_quadrature.interface_quadratures())?;
    }

    Ok(())
}

/// Helper struct to reduce boilerplate when initializing bodies.
pub struct BodyInitializer2d<'a> {
    state: &'a StorageContainer,
    entity: Entity,
}

#[allow(dead_code)]
impl<'a> BodyInitializer2d<'a> {
    pub fn initialize_in_state(state: &'a StorageContainer) -> Self {
        // Register components as a convenience. Double-registering is unproblematic.
        register_known_components().expect("Failed to register components. Should not happen");
        Self {
            state,
            entity: Entity::new(),
        }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Sets the displacement components for all dofs of the attached finite element model
    pub fn set_displacement(&self, displacement: &DVector<f64>) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            if displacement.len() == model.u.len() {
                model.u = displacement.clone();
            } else {
                return Err(Box::from(
                    "Cannot set displacement: Supplied displacement vector does not have the right dimensions.",
                ));
            }

            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set displacement: Entity does not have a finite element model yet.",
            ))
        }
    }

    /// Sets the velocity components for all dofs of the attached finite element model
    pub fn set_velocity(&self, velocity: &DVector<f64>) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            if velocity.len() == model.v.len() {
                model.v = velocity.clone();
            } else {
                return Err(Box::from(
                    "Cannot set velocity: Supplied velocity vector does not have the right dimensions.",
                ));
            }

            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set velocity: Entity does not have a finite element model yet.",
            ))
        }
    }

    /// Adds an elastic finite element model and a volume mesh component to the entity.
    pub fn add_finite_element_model(
        &self,
        fe_model: impl Into<FiniteElementModel2d>,
        material_volume_mesh: impl Into<VolumeMesh2d>,
    ) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();
        let mut volume_meshes = self
            .state
            .get_component_storage::<VolumeMesh2d>()
            .borrow_mut();

        if fe_models.get_component(self.entity).is_some() {
            panic!("Cannot add finite elment model: Entity already has finite element model.");
        } else {
            let fe_model = fe_model.into();
            let material_volume_mesh = material_volume_mesh.into();
            let ndof = fe_model.ndof();

            let mesh_interpolator = match_on_finite_element_model_2d!(fe_model, fe_model => {
                fe_model.make_interpolator(material_volume_mesh.vertices())?
            });

            let model = FiniteElementElasticModel2d {
                model: fe_model,
                u: DVector::zeros(ndof),
                v: DVector::zeros(ndof),
                factorization: None,
                material_volume_mesh: material_volume_mesh.clone(),
                material_volume_interpolator: mesh_interpolator,
                material_surface: None,
                material_surface_interpolator: None,
                material: Material::default(),
                integrator: Default::default(),
                gravity_enabled: true,
                model_matrix_storage: None,
            };

            fe_models.insert(self.entity, model);
            volume_meshes.insert(self.entity, material_volume_mesh);

            Ok(self)
        }
    }

    pub fn add_boundary_conditions(&self, new_bc: Box<dyn DirichletBoundaryConditions>) -> &Self {
        let mut storage = self
            .state
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow_mut();

        let component = if let Some(dirichlet_bc) = storage.get_component_mut(self.entity) {
            let current_bc = mem::replace(&mut dirichlet_bc.bc, Box::new(Empty));
            let union = Union::try_new(vec![current_bc, new_bc]).expect("We expect BCs to be disjoint at construction");
            DirichletBoundaryConditionComponent::from(union)
        } else {
            DirichletBoundaryConditionComponent::from(new_bc)
        };
        storage.insert_component(self.entity, component);
        self
    }

    /// Set the integrator used for time integration of the model
    pub fn set_integrator(&self, integrator: IntegrationMethod) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            model.integrator = integrator;

            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set integrator: Entity does not have a finite element model yet.",
            ))
        }
    }

    pub fn set_gravity_enabled(&self, enabled: bool) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            model.gravity_enabled = enabled;
            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set gravity: Entity does not have a finite element model yet.",
            ))
        }
    }

    pub fn add_material_surface(&self, material_surface: impl Into<SurfaceMesh2d>) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();
        let mut surface_meshes = self
            .state
            .get_component_storage::<SurfaceMesh2d>()
            .borrow_mut();

        if let Some(mut fe_model) = fe_models.get_component_mut(self.entity) {
            let material_surface = material_surface.into();
            let interpolator = match_on_finite_element_model_2d!(fe_model.model, model => {
                model.make_interpolator(material_surface.vertices())?
            });

            if surface_meshes.get_component(self.entity).is_some() {
                panic!("Cannot add surface mesh: Entity already has a surface mesh attached.");
            }

            let closed_surface_mesh = ClosedSurfaceMesh2d::from_mesh(material_surface.0.clone())?;

            fe_model.material_surface = Some(closed_surface_mesh);
            fe_model.material_surface_interpolator = Some(interpolator);

            surface_meshes.insert(self.entity, material_surface);

            Ok(self)
        } else {
            panic!(
                "No finite element model found for entity. Cannot add material surface.\
                 Please add a finite element model to the entity first."
            );
        }
    }

    /// Sets the boundary conditions of this entity to homogeneous Dirichlet conditions over the given nodes
    pub fn set_static_nodes(&self, static_nodes: Vec<usize>) -> &Self {
        let mut bc_storage = self
            .state
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow_mut();

        bc_storage.insert(self.entity, Homogeneous::new_2d(static_nodes.as_slice()).into());

        self
    }

    pub fn add_name(&self, name: impl Into<Name>) -> &Self {
        let mut names = self.state.get_component_storage::<Name>().borrow_mut();

        if names.get_component(self.entity).is_some() {
            panic!("Cannot add name: Entity already has name.");
        } else {
            names.insert(self.entity, name.into());
            self
        }
    }

    pub fn set_material(&self, material: impl Into<Material>) -> &Self {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel2d>()
            .borrow_mut();

        if let Some(mut fe_model) = fe_models.get_component_mut(self.entity) {
            fe_model.material = material.into();
            self
        } else {
            panic!(
                "No finite element model found for entity. Cannot add material.\
                 Please add a finite element model to the entity first."
            );
        }
    }
}

/// Helper struct to reduce boilerplate when initializing bodies.
pub struct BodyInitializer3d<'a> {
    state: &'a StorageContainer,
    entity: Entity,
}

#[allow(dead_code)]
impl<'a> BodyInitializer3d<'a> {
    pub fn from_entity(entity: Entity, state: &'a StorageContainer) -> Self {
        Self { state, entity }
    }

    pub fn initialize_in_state(state: &'a StorageContainer) -> Self {
        // Register components as a convenience. Double-registering is unproblematic.
        register_known_components().expect("Failed to register components. Should not happen");
        Self {
            state,
            entity: Entity::new(),
        }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Sets the displacement components for all dofs of the attached finite element model
    pub fn set_displacement(&self, displacement: &DVector<f64>) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            if displacement.len() == model.u.len() {
                model.u = displacement.clone();
            } else {
                return Err(Box::from(
                    "Cannot set displacement: Supplied displacement vector does not have the right dimensions.",
                ));
            }

            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set displacement: Entity does not have a finite element model yet.",
            ))
        }
    }

    /// Sets the displacements of all nodes of the finite element model to the same value.
    pub fn set_uniform_displacement(&self, displacement: &Vector3<f64>) -> Result<&Self, Box<dyn Error>> {
        let num_vertices = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow()
            .get_component(self.entity)
            .ok_or_else(|| "Cannot set displacement without a FE model.")?
            .model
            .ndof()
            / 3;
        if num_vertices > 0 {
            self.set_displacement(&flatten_vertically(&vec![*displacement; num_vertices]).unwrap())?;
        }

        Ok(self)
    }

    /// Sets the velocity components for all dofs of the attached finite element model
    pub fn set_velocity(&self, velocity: &DVector<f64>) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            if velocity.len() == model.v.len() {
                model.v = velocity.clone();
            } else {
                return Err(Box::from(
                    "Cannot set velocity: Supplied velocity vector does not have the right dimensions.",
                ));
            }

            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set velocity: Entity does not have a finite element model yet.",
            ))
        }
    }

    /// Adds an elastic finite element model and a volume mesh component to the entity.
    pub fn add_finite_element_model(
        &self,
        fe_model: impl Into<FiniteElementModel3d>,
        material_volume_mesh: impl Into<VolumeMesh3d>,
    ) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();
        let mut volume_meshes = self
            .state
            .get_component_storage::<VolumeMesh3d>()
            .borrow_mut();

        if fe_models.get_component(self.entity).is_some() {
            panic!("Cannot add finite elment model: Entity already has finite element model.");
        } else {
            let fe_model = fe_model.into();
            let material_volume_mesh = material_volume_mesh.into();
            let ndof = fe_model.ndof();

            let mesh_interpolator = match_on_finite_element_model_3d!(fe_model, fe_model => {
                let accelerator = GeometryCollectionAccelerator::new(fe_model);
                FiniteElementInterpolator::interpolate_space(
                    &accelerator,
                    material_volume_mesh.vertices())?
            });

            let model = FiniteElementElasticModel3d {
                model: fe_model,
                u: DVector::zeros(ndof),
                v: DVector::zeros(ndof),
                factorization: None,
                material_volume_mesh: material_volume_mesh.clone(),
                material_volume_interpolator: mesh_interpolator,
                material: Material::default(),
                integrator: Default::default(),
                gravity_enabled: true,
                rotations: None,
                model_matrix_storage: None,
            };

            fe_models.insert(self.entity, model);
            volume_meshes.insert(self.entity, material_volume_mesh);

            Ok(self)
        }
    }

    /// Set the integrator used for time integration of the model
    pub fn set_integrator(&self, integrator: IntegrationMethod) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            model.integrator = integrator;

            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set integrator: Entity does not have a finite element model yet.",
            ))
        }
    }

    pub fn set_gravity_enabled(&self, enabled: bool) -> Result<&Self, Box<dyn Error>> {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();

        if let Some(model) = fe_models.get_component_mut(self.entity) {
            model.gravity_enabled = enabled;
            Ok(self)
        } else {
            Err(Box::from(
                "Cannot set gravity: Entity does not have a finite element model yet.",
            ))
        }
    }

    /// Sets the boundary conditions of this entity to homogeneous Dirichlet conditions over the given nodes
    pub fn set_static_nodes(&self, static_nodes: Vec<usize>) -> &Self {
        let mut bc_storage = self
            .state
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow_mut();

        bc_storage.insert(self.entity, Homogeneous::new_3d(static_nodes.as_slice()).into());

        self
    }

    pub fn add_name(&self, name: impl Into<Name>) -> &Self {
        let mut names = self.state.get_component_storage::<Name>().borrow_mut();

        if names.get_component(self.entity).is_some() {
            panic!("Cannot add name: Entity already has name.");
        } else {
            names.insert(self.entity, name.into());
            self
        }
    }

    pub fn set_material(&self, material: impl Into<Material>) -> &Self {
        let mut fe_models = self
            .state
            .get_component_storage::<FiniteElementElasticModel3d>()
            .borrow_mut();

        if let Some(mut fe_model) = fe_models.get_component_mut(self.entity) {
            fe_model.material = material.into();
            self
        } else {
            panic!(
                "No finite element model found for entity. Cannot add material.\
                 Please add a finite element model to the entity first."
            );
        }
    }

    pub fn add_volume_mesh(&self, volume_mesh: impl Into<VolumeMesh3d>) -> &Self {
        let volume_mesh = volume_mesh.into();
        let mut meshes = self
            .state
            .get_component_storage::<VolumeMesh3d>()
            .borrow_mut();

        if meshes.get_component(self.entity).is_some() {
            panic!("Entity already has volume mesh.");
        } else {
            meshes.insert_component(self.entity, volume_mesh);
        }

        self
    }

    pub fn add_boundary_conditions(&self, new_bc: Box<dyn DirichletBoundaryConditions>) -> &Self {
        let mut storage = self
            .state
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow_mut();

        let component = if let Some(dirichlet_bc) = storage.get_component_mut(self.entity) {
            let current_bc = mem::replace(&mut dirichlet_bc.bc, Box::new(Empty));
            let union = Union::try_new(vec![current_bc, new_bc]).expect("We expect BCs to be disjoint at construction");
            DirichletBoundaryConditionComponent::from(union)
        } else {
            DirichletBoundaryConditionComponent::from(new_bc)
        };
        storage.insert_component(self.entity, component);
        self
    }
}
