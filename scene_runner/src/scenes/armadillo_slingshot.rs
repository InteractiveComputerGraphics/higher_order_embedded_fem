use std::error::Error;

use crate::scenes::{filtered_vertex_indices, Scene, SceneConstructor, SceneParameters};

use crate::meshes::load_mesh_from_file;
use crate::scenes::helpers::BodyInitializer3d;
use core::fmt;
use fenris::connectivity::{Connectivity, ConnectivityMut};
use fenris::embedding::{
    embed_mesh_3d, embed_quadrature_3d, embed_quadrature_3d_with_opts, EmbeddedModelBuilder, QuadratureOptions,
    StabilizationOptions,
};
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::{ConvexPolygon3d, Quad3d};
use fenris::lp_solvers::GlopSolver;
use fenris::mesh::{Mesh3d, Tet10Mesh, Tet4Mesh};
use fenris::model::NodalModel;
use fenris::nalgebra::{Point3, Vector3};
use fenris::nested_vec::NestedVec;
use fenris::quadrature::{
    tet_quadrature_strength_1, tet_quadrature_strength_2, tet_quadrature_strength_3, tet_quadrature_strength_5,
};
use fenris::reorder::reorder_mesh_par;
use fenris::rtree::GeometryCollectionAccelerator;
use fenris::solid::materials::{StableNeoHookeanMaterial, YoungPoisson};
use fenris::space::FiniteElementSpace;
use hamilton::{Component, Entity, StorageContainer, System, Systems};
use log::info;
use simulation_toolbox::components::{
    get_simulation_time, set_gravity, PolyMesh3dCollection, PolyMesh3dComponent, TimeStep,
};
use simulation_toolbox::fem::bcs::{ConstantUniformDisplacement, ConstantUniformVelocity, Union};
use simulation_toolbox::io::obj::load_single_surface_polymesh3d_obj;
use simulation_toolbox::io::ply::dump_polymesh_faces_ply;
use simulation_toolbox::match_on_finite_element_model_3d;

#[allow(unused)]
use simulation_toolbox::fem::{FiniteElementIntegrator, IntegratorSettings};

use simulation_toolbox::fem::newton_cg::NewtonCgIntegrator3d;
use simulation_toolbox::fem::schwarz_precond::SchwarzPreconditionerComponent;
use simulation_toolbox::fem::{
    DirichletBoundaryConditionComponent, DirichletBoundaryConditions, FiniteElementMeshDeformer, FiniteElementModel3d,
    Material,
};

pub fn scenes() -> Vec<SceneConstructor> {
    vec![
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet4_500".to_string(),
            constructor: armadillo_slingshot_fem_tet4_500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet4_1500".to_string(),
            constructor: armadillo_slingshot_fem_tet4_1500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet4_3000".to_string(),
            constructor: armadillo_slingshot_fem_tet4_3000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet4_5000".to_string(),
            constructor: armadillo_slingshot_fem_tet4_5000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet4_full".to_string(),
            constructor: armadillo_slingshot_fem_tet4_full,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet10_500".to_string(),
            constructor: armadillo_slingshot_fem_tet10_500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet10_1000".to_string(),
            constructor: armadillo_slingshot_fem_tet10_1000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet10_3000".to_string(),
            constructor: armadillo_slingshot_fem_tet10_3000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_fem_tet10_5000".to_string(),
            constructor: armadillo_slingshot_fem_tet10_5000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet4_500".to_string(),
            constructor: armadillo_slingshot_embedded_tet4_500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet4_1500".to_string(),
            constructor: armadillo_slingshot_embedded_tet4_1500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet4_3000".to_string(),
            constructor: armadillo_slingshot_embedded_tet4_3000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet4_5000".to_string(),
            constructor: armadillo_slingshot_embedded_tet4_5000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet10_500".to_string(),
            constructor: armadillo_slingshot_embedded_tet10_500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet10_1000".to_string(),
            constructor: armadillo_slingshot_embedded_tet10_1000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet10_1500".to_string(),
            constructor: armadillo_slingshot_embedded_tet10_1500,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet10_3000".to_string(),
            constructor: armadillo_slingshot_embedded_tet10_3000,
        },
        SceneConstructor {
            name: "armadillo_slingshot_embedded_tet10_5000".to_string(),
            constructor: armadillo_slingshot_embedded_tet10_5000,
        },
    ]
}

fn initial_scene(name: &str) -> Scene {
    // Use y-axis gravity to simplify working with meshes that are oriented along the y-axis
    let mut initial_state = Default::default();
    set_gravity(&mut initial_state, Vector3::new(0.0, -9.81, 0.0));
    let dt = 2e-3;
    initial_state.replace_storage(<TimeStep as Component>::Storage::new(TimeStep(dt)));
    Scene {
        initial_state,
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: 10.0,
        name: String::from(name),
    }
}

fn default_material() -> Material {
    Material {
        density: 1000.0,
        mass_damping_coefficient: None,
        stiffness_damping_coefficient: Some(0.015),
        elastic_model: StableNeoHookeanMaterial::from(YoungPoisson {
            young: 5e5,
            poisson: 0.40,
        })
        .into(),
    }
}

fn add_systems(systems: &mut Systems) {
    //let integrator_settings = IntegratorSettings::default().set_project_stiffness(false);
    //systems.add_system(Box::new(FiniteElementIntegrator::with_settings(integrator_settings)));
    systems.add_system(Box::new(NewtonCgIntegrator3d::default()));
    systems.add_system(Box::new(FiniteElementMeshDeformer));
}

fn load_mesh(params: &SceneParameters, filename: &str) -> Result<Tet4Mesh<f64>, Box<dyn Error>> {
    load_mesh_from_file(&params.asset_dir, &format!("meshes/armadillo_slingshot/{}", filename))
}

fn load_fine_embedded_mesh(params: &SceneParameters) -> Result<Tet4Mesh<f64>, Box<dyn Error>> {
    load_mesh(params, "armadillo_slingshot.msh")
}

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
    let dist_eps = 0.005;

    let left_arm_quad = Quad3d::from_vertices([
        Point3::new(-1.3292, 1.18282, -0.500656),
        Point3::new(-1.17938, 1.08153, -0.221283),
        Point3::new(-1.14583, 1.29737, -0.16103),
        Point3::new(-1.29661, 1.39865, -0.439886),
    ]);

    let left_arm_indices =
        filtered_vertex_indices(mesh_vertices, |v| left_arm_quad.project_point(v).distance <= dist_eps);

    let right_arm_quad = Quad3d::from_vertices([
        Point3::new(1.18738, 1.11562, -0.0642211),
        Point3::new(1.29702, 1.26411, -0.28097),
        Point3::new(1.25493, 1.41509, -0.194934),
        Point3::new(1.14410, 1.26502, 0.0177633),
    ]);

    let right_arm_indices =
        filtered_vertex_indices(mesh_vertices, |v| right_arm_quad.project_point(v).distance <= dist_eps);

    let back_plate_quad = Quad3d::from_vertices([
        Point3::new(-0.131145, 0.405836, 0.711431),
        Point3::new(0.157981, 0.405836, 0.711431),
        Point3::new(0.157981, 0.705468, 0.711431),
        Point3::new(-0.130775, 0.705468, 0.711431),
    ]);
    let back_plate_indices =
        filtered_vertex_indices(mesh_vertices, |v| back_plate_quad.project_point(v).distance <= dist_eps);

    let feet_nodes = filtered_vertex_indices(mesh_vertices, |v| v.y <= -1.06);

    let mut static_nodes = Vec::new();
    static_nodes.extend(left_arm_indices);
    static_nodes.extend(right_arm_indices);
    static_nodes.extend(feet_nodes);

    let static_bc = ConstantUniformDisplacement::new(&static_nodes, Vector3::zeros());
    let slingshot_bc = ConstantUniformVelocity::new(&back_plate_indices, Vector3::new(0.0, 0.0, 0.5));

    BoundaryConditions {
        initial_bc: Box::new(
            Union::try_new(vec![Box::new(static_bc.clone()), Box::new(slingshot_bc.clone())]).unwrap(),
        ),
        final_bc: Box::new(static_bc),
        switch_time: 2.5,
    }
}

fn add_model<'a, Model, C>(
    params: &SceneParameters,
    scene: &mut Scene,
    model: Model,
    mesh: &Mesh3d<f64, C>,
    _volume_poly_mesh: PolyMesh3d<f64>,
) -> Result<Entity, Box<dyn Error>>
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
        .join("meshes/armadillo_slingshot/armadillo_slingshot_render.obj");
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

    Ok(entity)
}

fn armadillo_slingshot_fem_tet4(params: &SceneParameters, name: &str, filename: &str) -> Result<Scene, Box<dyn Error>> {
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

pub fn armadillo_slingshot_fem_tet4_500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet4_500";
    // TODO: This is currently just the cage
    armadillo_slingshot_fem_tet4(params, &name, "cages/armadillo_slingshot_cage_500_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet4_1500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet4_1500";
    // TODO: This is currently just the cage
    armadillo_slingshot_fem_tet4(params, &name, "cages/armadillo_slingshot_cage_1500_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet4_3000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet4_3000";
    // TODO: This is currently just the cage
    armadillo_slingshot_fem_tet4(params, &name, "cages/armadillo_slingshot_cage_3000_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet4_5000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet4_5000";
    // TODO: This is currently just the cage
    armadillo_slingshot_fem_tet4(params, &name, "cages/armadillo_slingshot_cage_5000_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet4_full(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet4_full";
    armadillo_slingshot_fem_tet4(params, &name, "armadillo_slingshot.msh")
}

fn armadillo_slingshot_fem_tet10(
    params: &SceneParameters,
    name: &str,
    filename: &str,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_mesh(params, filename)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);
    let tet_mesh = Tet10Mesh::from(&tet_mesh);
    let tet_mesh = reorder_mesh_par(&tet_mesh).apply(&tet_mesh);

    // TODO: Use better
    let quadrature = tet_quadrature_strength_5();
    let fe_model = NodalModel::from_mesh_and_quadrature(tet_mesh.clone(), quadrature)
        .with_mass_quadrature(tet_quadrature_strength_5());
    add_model(params, &mut scene, fe_model, &tet_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn armadillo_slingshot_fem_tet10_500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet10_500";
    armadillo_slingshot_fem_tet10(params, &name, "cages/armadillo_slingshot_cage_500_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet10_1000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet10_1000";
    armadillo_slingshot_fem_tet10(params, &name, "cages/armadillo_slingshot_cage_1000_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet10_3000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet10_3000";
    armadillo_slingshot_fem_tet10(params, &name, "cages/armadillo_slingshot_cage_3000_fixed.msh")
}

pub fn armadillo_slingshot_fem_tet10_5000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_fem_tet10_5000";
    armadillo_slingshot_fem_tet10(params, &name, "cages/armadillo_slingshot_cage_5000_fixed.msh")
}

fn armadillo_slingshot_embedded_tet10(
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
    let embedding = embed_mesh_3d(&background_mesh, &volume_poly_mesh);
    let mass_quadrature_opts = QuadratureOptions {
        stabilization: Some(StabilizationOptions {
            stabilization_factor: 1e-8,
            stabilization_quadrature: tet_quadrature_strength_5(),
        }),
    };

    let mass_quadrature = embed_quadrature_3d_with_opts(
        &background_mesh,
        &embedding,
        tet_quadrature_strength_5(),
        tet_quadrature_strength_5(),
        &mass_quadrature_opts,
    )?;

    let stiffness_quadrature_opts = QuadratureOptions {
        stabilization: Some(StabilizationOptions {
            stabilization_factor: 1e-8,
            stabilization_quadrature: tet_quadrature_strength_2(),
        }),
    };

    let stiffness_quadrature = embed_quadrature_3d_with_opts(
        &background_mesh,
        &embedding,
        tet_quadrature_strength_3(),
        tet_quadrature_strength_3(),
        &stiffness_quadrature_opts,
    )?
    .simplified(3, &GlopSolver::new())?;
    let elliptic_quadrature = stiffness_quadrature.clone();

    let fe_model = EmbeddedModelBuilder::from_embedding(&background_mesh, embedding)
        .mass_quadrature(mass_quadrature)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();

    let schwarz = SchwarzPreconditionerComponent::from_embedded_model(&fe_model, 0.5);
    let entity = add_model(params, &mut scene, fe_model, &background_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    scene.initial_state.insert_component(entity, schwarz);

    Ok(scene)
}

pub fn armadillo_slingshot_embedded_tet10_500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet10_500";
    armadillo_slingshot_embedded_tet10(params, &name, "cages/armadillo_slingshot_cage_500_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet10_1000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet10_1000";
    armadillo_slingshot_embedded_tet10(params, &name, "cages/armadillo_slingshot_cage_1000_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet10_1500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet10_1500";
    armadillo_slingshot_embedded_tet10(params, &name, "cages/armadillo_slingshot_cage_1500_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet10_3000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet10_3000";
    armadillo_slingshot_embedded_tet10(params, &name, "cages/armadillo_slingshot_cage_3000_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet10_5000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet10_5000";
    armadillo_slingshot_embedded_tet10(params, &name, "cages/armadillo_slingshot_cage_5000_fixed.msh")
}

fn armadillo_slingshot_embedded_tet4(
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

pub fn armadillo_slingshot_embedded_tet4_500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet4_500";
    armadillo_slingshot_embedded_tet4(params, &name, "cages/armadillo_slingshot_cage_500_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet4_1500(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet4_1500";
    armadillo_slingshot_embedded_tet4(params, &name, "cages/armadillo_slingshot_cage_1500_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet4_3000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet4_3000";
    armadillo_slingshot_embedded_tet4(params, &name, "cages/armadillo_slingshot_cage_3000_fixed.msh")
}

pub fn armadillo_slingshot_embedded_tet4_5000(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "armadillo_slingshot_embedded_tet4_5000";
    armadillo_slingshot_embedded_tet4(params, &name, "cages/armadillo_slingshot_cage_5000_fixed.msh")
}
