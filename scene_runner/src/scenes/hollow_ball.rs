use std::error::Error;

use crate::scenes::{filtered_vertex_indices, Scene, SceneConstructor, SceneParameters};

use simulation_toolbox::fem::{
    DirichletBoundaryConditionComponent, DirichletBoundaryConditions, FiniteElementIntegrator,
    FiniteElementMeshDeformer, FiniteElementModel3d, IntegratorSettings, Material,
};

use crate::meshes::load_mesh_from_file;
use crate::scenes::helpers::BodyInitializer3d;
use core::fmt;
use fenris::connectivity::{Connectivity, ConnectivityMut};
use fenris::embedding::{
    embed_mesh_3d, embed_quadrature_3d, embed_quadrature_3d_with_opts, EmbeddedModelBuilder, EmbeddedQuadrature,
    QuadratureOptions, StabilizationOptions,
};
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::{ConvexPolygon3d, Quad3d};
use fenris::lp_solvers::GlopSolver;
use fenris::mesh::{Mesh3d, Tet10Mesh, Tet4Mesh};
use fenris::model::NodalModel;
use fenris::nalgebra::{Point3, Vector3, U3};
use fenris::nested_vec::NestedVec;
use fenris::quadrature::{
    tet_quadrature_strength_1, tet_quadrature_strength_2, tet_quadrature_strength_3, tet_quadrature_strength_5,
    Quadrature,
};
use fenris::reorder::reorder_mesh_par;
use fenris::rtree::GeometryCollectionAccelerator;
use fenris::solid::materials::{StableNeoHookeanMaterial, YoungPoisson};
use fenris::space::FiniteElementSpace;
use hamilton::{Component, Entity, StorageContainer, System, Systems};
use log::info;
use simulation_toolbox::components::{get_simulation_time, PolyMesh3dCollection, PolyMesh3dComponent, TimeStep};
use simulation_toolbox::fem::bcs::{
    ConstantUniformAngularVelocity, ConstantUniformDisplacement, ConstantUniformVelocity, Union,
};
use simulation_toolbox::io::obj::load_single_surface_polymesh3d_obj;
use simulation_toolbox::io::ply::dump_polymesh_faces_ply;
use simulation_toolbox::match_on_finite_element_model_3d;

pub fn scenes() -> Vec<SceneConstructor> {
    vec![
        SceneConstructor {
            name: "hollow_ball_fem_tet4_coarse".to_string(),
            constructor: hollow_ball_fem_tet4_coarse,
        },
        SceneConstructor {
            name: "hollow_ball_fem_tet4_medium".to_string(),
            constructor: hollow_ball_fem_tet4_medium,
        },
        SceneConstructor {
            name: "hollow_ball_fem_tet4_fine".to_string(),
            constructor: hollow_ball_fem_tet4_fine,
        },
        SceneConstructor {
            name: "hollow_ball_fem_tet10_coarse".to_string(),
            constructor: hollow_ball_fem_tet10_coarse,
        },
        SceneConstructor {
            name: "hollow_ball_fem_tet10_medium".to_string(),
            constructor: hollow_ball_fem_tet10_medium,
        },
        SceneConstructor {
            name: "hollow_ball_fem_tet10_fine".to_string(),
            constructor: hollow_ball_fem_tet10_fine,
        },
        SceneConstructor {
            name: "hollow_ball_embedded_tet4_coarse".to_string(),
            constructor: hollow_ball_embedded_tet4_coarse,
        },
        SceneConstructor {
            name: "hollow_ball_embedded_tet4_medium".to_string(),
            constructor: hollow_ball_embedded_tet4_medium,
        },
        SceneConstructor {
            name: "hollow_ball_embedded_tet10_coarse".to_string(),
            constructor: hollow_ball_embedded_tet10_coarse,
        },
        SceneConstructor {
            name: "hollow_ball_embedded_tet10_medium".to_string(),
            constructor: hollow_ball_embedded_tet10_medium,
        },
    ]
}

fn initial_scene(name: &str) -> Scene {
    // Use y-axis gravity to simplify working with meshes that are oriented along the y-axis
    let mut initial_state = StorageContainer::default();
    // set_gravity(&mut initial_state, Vector3::new(0.0, -9.81, 0.0));
    let dt = 2e-3;
    initial_state.replace_storage(<TimeStep as Component>::Storage::new(TimeStep(dt)));
    Scene {
        initial_state,
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: 4.50,
        name: String::from(name),
    }
}

fn default_material() -> Material {
    Material {
        density: 1000.0,
        mass_damping_coefficient: None,
        stiffness_damping_coefficient: Some(0.01),
        elastic_model: StableNeoHookeanMaterial::from(YoungPoisson {
            young: 1e7,
            poisson: 0.48,
        })
        .into(),
    }
}

fn add_systems(systems: &mut Systems) {
    let integrator_settings = IntegratorSettings::default().set_project_stiffness(false);
    systems.add_system(Box::new(FiniteElementIntegrator::with_settings(integrator_settings)));
    systems.add_system(Box::new(FiniteElementMeshDeformer));
}

fn load_mesh(params: &SceneParameters, filename: &str) -> Result<Tet4Mesh<f64>, Box<dyn Error>> {
    load_mesh_from_file(
        &params.asset_dir,
        &format!("meshes/hollow_ball/ball2/proper/{}", filename),
    )
}

fn load_fine_embedded_mesh(params: &SceneParameters) -> Result<Tet4Mesh<f64>, Box<dyn Error>> {
    // TODO: Use the full resolution mesh eventually
    load_mesh(params, FINE_MESH_FILENAME)
}

const COARSE_CAGE_FILENAME: &'static str = "math_ball_nested_cage_7500_corrected.msh";
const MEDIUM_MESH_FILENAME: &'static str = "math_ball_medium.msh";
const FINE_MESH_FILENAME: &'static str = "math_ball.msh";

/// System that switches out BCs after a given simulation time
#[derive(Debug)]
pub struct BoundaryConditionSwitchSystem {
    time_for_switch: f64,
    new_bcs: Option<Box<dyn DirichletBoundaryConditions>>,
    entity: Entity,
}

impl fmt::Display for BoundaryConditionSwitchSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BoundaryConditionSwitchSystem")
    }
}

impl System for BoundaryConditionSwitchSystem {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let t = get_simulation_time(data);
        let mut bcs = data
            .get_component_storage::<DirichletBoundaryConditionComponent>()
            .borrow_mut();

        if let Ok(t) = t {
            if t >= self.time_for_switch {
                if let Some(new_bcs) = self.new_bcs.take() {
                    let bc_component = bcs
                        .get_component_mut(self.entity)
                        .expect("Entity must be in simulation");
                    bc_component.bc = new_bcs;
                }
            }
        }

        Ok(())
    }
}

struct BoundaryConditions {
    // BCs at the beginning of the scene
    initial_bc: Box<dyn DirichletBoundaryConditions>,
    // BCs after switch
    final_bc: Box<dyn DirichletBoundaryConditions>,
    // Time when switch takes place
    switch_time: f64,
}

fn set_up_boundary_conditions(mesh_vertices: &[Point3<f64>]) -> BoundaryConditions {
    // Epsilon for classifying nodes as belonging to a constrained surface
    let dist_eps = 0.01;

    // Connectors are placed along z-axis, so we identify them by plus and minus
    // (above and below z = 0)
    let connector_plus_quad = Quad3d::from_vertices([
        Point3::new(-0.2, -0.2, 2.1),
        Point3::new(0.2, -0.2, 2.1),
        Point3::new(0.2, 0.2, 2.1),
        Point3::new(-0.2, 0.2, 2.1),
    ]);
    let connector_plus_nodes = filtered_vertex_indices(mesh_vertices, |v| {
        connector_plus_quad.project_point(v).distance <= dist_eps
    });

    let connector_minus_quad = Quad3d::from_vertices([
        Point3::new(0.2, -0.2, -2.1),
        Point3::new(-0.2, -0.2, -2.1),
        Point3::new(-0.2, 0.2, -2.1),
        Point3::new(0.2, 0.2, -2.1),
    ]);

    let connector_minus_nodes = filtered_vertex_indices(mesh_vertices, |v| {
        connector_minus_quad.project_point(v).distance <= dist_eps
    });

    let plus_velocity = Vector3::new(0.0, 0.0, -1.0);
    let plus_bc = ConstantUniformVelocity::new(&connector_plus_nodes, plus_velocity);
    // let minus_bc = ConstantUniformVelocity::new(&connector_minus_nodes, Vector3::new(0.0, 0.0, 0.4));

    let minus_node_positions = connector_minus_nodes
        .iter()
        .copied()
        .map(|index| mesh_vertices[index])
        .collect();
    let minus_bc = ConstantUniformAngularVelocity::new(
        &connector_minus_nodes,
        Vector3::new(0.0, 0.0, 0.8),
        Point3::new(0.0, 0.0, -2.1),
        minus_node_positions,
    );

    let switch_time = 2.5;
    let plus_bc_after = ConstantUniformDisplacement::new(&connector_plus_nodes, plus_velocity * switch_time);

    BoundaryConditions {
        initial_bc: Box::new(Union::try_new(vec![Box::new(minus_bc.clone()), Box::new(plus_bc.clone())]).unwrap()),
        final_bc: Box::new(Union::try_new(vec![Box::new(minus_bc.clone()), Box::new(plus_bc_after.clone())]).unwrap()),
        switch_time,
    }
}

fn add_model<'a, Model, C>(
    params: &SceneParameters,
    scene: &mut Scene,
    model: Model,
    mesh: &Mesh3d<f64, C>,
    _volume_poly_mesh: PolyMesh3d<f64>,
) -> Result<(), Box<dyn Error>>
where
    Model: Into<FiniteElementModel3d>,
    C: Connectivity,
    C::FaceConnectivity: ConnectivityMut,
{
    let model = model.into();

    match_on_finite_element_model_3d!(model, model => {
        info!("Setting up model. Vertices: {}. Elements: {}",
           model.vertices().len(), model.num_connectivities());
    });

    info!("Generating render mesh");
    let render_mesh_path = params
        .asset_dir
        .join("meshes/hollow_ball/ball2/proper/math_ball.obj");
    let render_surface_mesh = load_single_surface_polymesh3d_obj(&render_mesh_path)?;

    info!("Generating wireframes");
    let (wireframe_volume, wireframe_surface) = {
        let fe_mesh_volume = mesh.extract_face_soup();
        let fe_mesh_surface = mesh.extract_surface_mesh();
        let mut wireframe_volume = PolyMesh3d::from_surface_mesh(&fe_mesh_volume);
        let mut wireframe_surface = PolyMesh3d::from_surface_mesh(&fe_mesh_surface);
        wireframe_volume.split_edges_n_times(2);
        wireframe_surface.split_edges_n_times(2);

        (wireframe_volume, wireframe_surface)
    };

    let render_component = match_on_finite_element_model_3d!(&model, model => {
        let accelerator = GeometryCollectionAccelerator::new(model);
        PolyMesh3dComponent::new("render", render_surface_mesh)
                .with_subfolder("render_meshes")
                .with_interpolator(&accelerator)?
    });

    info!("Setting up model");
    let material = default_material();
    let bcs = match_on_finite_element_model_3d!(&model, model => {
         set_up_boundary_conditions(model.vertices())
    });

    // For now we're not interested in exporting the volume meshes as we want to save the space
    let volume_poly_mesh = PolyMesh3d::from_poly_data(Vec::new(), NestedVec::new(), NestedVec::new());

    let entity = BodyInitializer3d::initialize_in_state(&scene.initial_state)
        .add_name(scene.name.clone())
        .add_finite_element_model(model, volume_poly_mesh)?
        .set_material(material)
        .add_boundary_conditions(bcs.initial_bc)
        .entity();

    scene
        .simulation_systems
        .add_system(Box::new(BoundaryConditionSwitchSystem {
            time_for_switch: bcs.switch_time,
            new_bcs: Some(bcs.final_bc),
            entity,
        }));

    scene
        .initial_state
        .insert_component(entity, PolyMesh3dCollection(vec![render_component]));

    {
        // We don't want to export wireframes at every step, only at scene creation
        let ply_dir = params.output_dir.join("ply");
        dump_polymesh_faces_ply(&wireframe_surface, &ply_dir, "wireframe_surface.ply")?;
        dump_polymesh_faces_ply(&wireframe_volume, &ply_dir, "wireframe_volume.ply")?;
    }

    Ok(())
}

fn hollow_ball_fem_tet4(params: &SceneParameters, name: &str, filename: &str) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_mesh(params, filename)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);
    let tet_mesh = reorder_mesh_par(&tet_mesh).apply(&tet_mesh);

    let quadrature = tet_quadrature_strength_1();
    let fe_model = NodalModel::from_mesh_and_quadrature(tet_mesh.clone(), quadrature)
        .with_mass_quadrature(tet_quadrature_strength_2());
    add_model(params, &mut scene, fe_model, &tet_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn hollow_ball_fem_tet4_coarse(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_fem_tet4_coarse";
    hollow_ball_fem_tet4(params, &name, COARSE_CAGE_FILENAME)
}

pub fn hollow_ball_fem_tet4_medium(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_fem_tet4_medium";
    hollow_ball_fem_tet4(params, &name, MEDIUM_MESH_FILENAME)
}

pub fn hollow_ball_fem_tet4_fine(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_fem_tet4_fine";
    hollow_ball_fem_tet4(params, &name, FINE_MESH_FILENAME)
}

fn hollow_ball_fem_tet10(params: &SceneParameters, name: &str, filename: &str) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_mesh(params, filename)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);
    let tet_mesh = Tet10Mesh::from(&tet_mesh);
    let tet_mesh = reorder_mesh_par(&tet_mesh).apply(&tet_mesh);

    let quadrature = tet_quadrature_strength_3();
    let fe_model = NodalModel::from_mesh_and_quadrature(tet_mesh.clone(), quadrature)
        .with_mass_quadrature(tet_quadrature_strength_5());
    add_model(params, &mut scene, fe_model, &tet_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn hollow_ball_fem_tet10_coarse(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_fem_tet10_coarse";
    hollow_ball_fem_tet10(params, &name, COARSE_CAGE_FILENAME)
}

pub fn hollow_ball_fem_tet10_medium(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_fem_tet10_medium";
    hollow_ball_fem_tet10(params, &name, MEDIUM_MESH_FILENAME)
}

pub fn hollow_ball_fem_tet10_fine(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_fem_tet10_fine";
    hollow_ball_fem_tet10(params, &name, FINE_MESH_FILENAME)
}

fn count_interface_points(quadrature: &EmbeddedQuadrature<f64, U3>) -> usize {
    quadrature
        .interface_quadratures()
        .iter()
        .map(|quadrature| quadrature.points().len())
        .sum::<usize>()
}

fn hollow_ball_embedded_tet10(
    params: &SceneParameters,
    name: &str,
    background_mesh_filename: &str,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let embedded_mesh = load_fine_embedded_mesh(params)?;
    let volume_poly_mesh = PolyMesh3d::from(&embedded_mesh);
    let background_mesh = load_mesh(params, background_mesh_filename)?;
    let background_mesh = Tet10Mesh::from(&background_mesh);
    let background_mesh = reorder_mesh_par(&background_mesh).apply(&background_mesh);

    // TODO: Use different stabilization options once we have different quadrature rules for
    // mass/stiffness?
    info!("Embedding mesh...");
    let embedding = embed_mesh_3d(&background_mesh, &volume_poly_mesh);
    let quadrature_opts = QuadratureOptions {
        stabilization: Some(StabilizationOptions {
            stabilization_factor: 1e-8,
            stabilization_quadrature: tet_quadrature_strength_2(),
        }),
    };
    info!("Constructing mass quadrature...");
    let mass_quadrature = embed_quadrature_3d_with_opts(
        &background_mesh,
        &embedding,
        tet_quadrature_strength_5(),
        tet_quadrature_strength_5(),
        &quadrature_opts,
    )?;
    info!(
        "Constructed mass quadrature: {} points in interface elements.",
        count_interface_points(&mass_quadrature)
    );
    info!("Constructing stiffness quadrature...");
    let stiffness_quadrature = embed_quadrature_3d_with_opts(
        &background_mesh,
        &embedding,
        tet_quadrature_strength_3(),
        tet_quadrature_strength_3(),
        &quadrature_opts,
    )?;
    info!(
        "Constructed stiffness quadrature: {} points in interface elements.",
        count_interface_points(&stiffness_quadrature)
    );
    info!("Simplifying stiffness quadrature...");
    let stiffness_quadrature = stiffness_quadrature.simplified(3, &GlopSolver::new())?;
    info!(
        "Simplified stiffness quadrature: {} points in interface elements",
        count_interface_points(&stiffness_quadrature)
    );
    let elliptic_quadrature = stiffness_quadrature.clone();

    let fe_model = EmbeddedModelBuilder::from_embedding(&background_mesh, embedding)
        .mass_quadrature(mass_quadrature)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();

    add_model(params, &mut scene, fe_model, &background_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn hollow_ball_embedded_tet10_coarse(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_embedded_tet10_coarse";
    hollow_ball_embedded_tet10(params, &name, COARSE_CAGE_FILENAME)
}

pub fn hollow_ball_embedded_tet10_medium(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_embedded_tet10_medium";
    hollow_ball_embedded_tet10(params, &name, MEDIUM_MESH_FILENAME)
}

fn hollow_ball_embedded_tet4(
    params: &SceneParameters,
    name: &str,
    background_mesh_filename: &str,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let embedded_mesh = load_fine_embedded_mesh(params)?;
    let volume_poly_mesh = PolyMesh3d::from(&embedded_mesh);
    let background_mesh = load_mesh(params, background_mesh_filename)?;
    let background_mesh = reorder_mesh_par(&background_mesh).apply(&background_mesh);

    let embedding = embed_mesh_3d(&background_mesh, &volume_poly_mesh);

    // TODO: We don't use any stabilization because that should not be necessary for linear
    // tet elements. Is this correct?

    let mass_quadrature = embed_quadrature_3d(
        &background_mesh,
        &embedding,
        tet_quadrature_strength_2(),
        tet_quadrature_strength_2(),
    )?;

    let stiffness_quadrature = embed_quadrature_3d(
        &background_mesh,
        &embedding,
        tet_quadrature_strength_1(),
        tet_quadrature_strength_1(),
    )?
    // We want to ensure that we obtain a single-point zeroth-order quadrature
    .simplified(0, &GlopSolver::new())?;
    let elliptic_quadrature = stiffness_quadrature.clone();

    let fe_model = EmbeddedModelBuilder::from_embedding(&background_mesh, embedding)
        .mass_quadrature(mass_quadrature)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();

    add_model(params, &mut scene, fe_model, &background_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn hollow_ball_embedded_tet4_coarse(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_embedded_tet4_coarse";
    hollow_ball_embedded_tet4(params, &name, COARSE_CAGE_FILENAME)
}

pub fn hollow_ball_embedded_tet4_medium(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "hollow_ball_embedded_tet4_medium";
    hollow_ball_embedded_tet4(params, &name, MEDIUM_MESH_FILENAME)
}
