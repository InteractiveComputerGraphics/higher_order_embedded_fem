use std::error::Error;

use crate::scenes::{filtered_vertex_indices, Scene, SceneConstructor, SceneParameters};

use simulation_toolbox::fem::{
    DirichletBoundaryConditionComponent, DirichletBoundaryConditions, FiniteElementIntegrator3d,
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
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;
use fenris::lp_solvers::GlopSolver;
use fenris::mesh::{Hex20Mesh, Hex27Mesh, HexMesh, Mesh3d, Tet10Mesh, Tet4Mesh};
use fenris::model::NodalModel;
use fenris::nalgebra::{DVector, DVectorSliceMut, Point3, Vector3, U3};
use fenris::nested_vec::NestedVec;
use fenris::quadrature::{
    hex_quadrature_strength_11, hex_quadrature_strength_3, hex_quadrature_strength_5, tet_quadrature_strength_1,
    tet_quadrature_strength_10, tet_quadrature_strength_2, tet_quadrature_strength_3, tet_quadrature_strength_5,
    Quadrature,
};
use fenris::reorder::reorder_mesh_par;
use fenris::rtree::GeometryCollectionAccelerator;
use fenris::solid::materials::{StableNeoHookeanMaterial, YoungPoisson};
use hamilton::{Entity, StorageContainer, System, Systems};
use log::info;
use simulation_toolbox::components::{get_simulation_time, PolyMesh3dCollection, PolyMesh3dComponent};
use simulation_toolbox::fem::bcs::{
    ConstantDisplacement, ConstantUniformAngularVelocity, ConstantUniformDisplacement, ConstantUniformVelocity, Union,
};
use simulation_toolbox::io::ply::dump_polymesh_faces_ply;
use simulation_toolbox::match_on_finite_element_model_3d;

pub fn scenes() -> Vec<SceneConstructor> {
    vec![
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_5k".to_string(),
            constructor: cylinder_shell_fem_tet4_5k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_10k".to_string(),
            constructor: cylinder_shell_fem_tet4_10k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_20k".to_string(),
            constructor: cylinder_shell_fem_tet4_20k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_40k".to_string(),
            constructor: cylinder_shell_fem_tet4_40k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_80k".to_string(),
            constructor: cylinder_shell_fem_tet4_80k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_160k".to_string(),
            constructor: cylinder_shell_fem_tet4_160k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet4_320k".to_string(),
            constructor: cylinder_shell_fem_tet4_320k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet10_5k_strength2".to_string(),
            constructor: cylinder_shell_fem_tet10_5k_strength2,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet10_5k_strength3".to_string(),
            constructor: cylinder_shell_fem_tet10_5k_strength3,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet10_5k_strength5".to_string(),
            constructor: cylinder_shell_fem_tet10_5k_strength5,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet10_10k".to_string(),
            constructor: cylinder_shell_fem_tet10_10k,
        },
        SceneConstructor {
            name: "cylinder_shell_fem_tet10_20k".to_string(),
            constructor: cylinder_shell_fem_tet10_20k,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res2".to_string(),
            constructor: cylinder_shell_embedded_hex8_res2,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res3".to_string(),
            constructor: cylinder_shell_embedded_hex8_res3,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res5".to_string(),
            constructor: cylinder_shell_embedded_hex8_res5,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res10".to_string(),
            constructor: cylinder_shell_embedded_hex8_res10,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res14".to_string(),
            constructor: cylinder_shell_embedded_hex8_res14,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res16".to_string(),
            constructor: cylinder_shell_embedded_hex8_res16,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res18".to_string(),
            constructor: cylinder_shell_embedded_hex8_res18,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res20".to_string(),
            constructor: cylinder_shell_embedded_hex8_res20,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res22".to_string(),
            constructor: cylinder_shell_embedded_hex8_res22,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res24".to_string(),
            constructor: cylinder_shell_embedded_hex8_res24,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res26".to_string(),
            constructor: cylinder_shell_embedded_hex8_res26,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res28".to_string(),
            constructor: cylinder_shell_embedded_hex8_res28,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res29".to_string(),
            constructor: cylinder_shell_embedded_hex8_res29,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res30".to_string(),
            constructor: cylinder_shell_embedded_hex8_res30,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex8_res32".to_string(),
            constructor: cylinder_shell_embedded_hex8_res32,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res1".to_string(),
            constructor: cylinder_shell_embedded_hex20_res1,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res2".to_string(),
            constructor: cylinder_shell_embedded_hex20_res2,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res3".to_string(),
            constructor: cylinder_shell_embedded_hex20_res3,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res4".to_string(),
            constructor: cylinder_shell_embedded_hex20_res4,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res5".to_string(),
            constructor: cylinder_shell_embedded_hex20_res5,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res6_strength5".to_string(),
            constructor: cylinder_shell_embedded_hex20_res6_strength5,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res6_strength5_no_simp".to_string(),
            constructor: cylinder_shell_embedded_hex20_res6_strength5_no_simp,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res6_strength3".to_string(),
            constructor: cylinder_shell_embedded_hex20_res6_strength3,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res7_strength3".to_string(),
            constructor: cylinder_shell_embedded_hex20_res7_strength3,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res7_strength5".to_string(),
            constructor: cylinder_shell_embedded_hex20_res7_strength5,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res7_strength5_no_simp".to_string(),
            constructor: cylinder_shell_embedded_hex20_res7_strength5_no_simp,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res8".to_string(),
            constructor: cylinder_shell_embedded_hex20_res8,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex20_res10".to_string(),
            constructor: cylinder_shell_embedded_hex20_res10,
        },
        SceneConstructor {
            name: "cylinder_shell_embedded_hex27_res5".to_string(),
            constructor: cylinder_shell_embedded_hex27_res5,
        },
    ]
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum QuadratureRule {
    Tet1,
    Tet2,
    Tet3,
    Tet5,
    Tet10,
    Hex3,
    Hex5,
    Hex11,
}

impl QuadratureRule {
    fn is_hex(&self) -> bool {
        match self {
            Self::Hex3 | Self::Hex5 | Self::Hex11 => true,
            _ => false,
        }
    }

    fn is_tet(&self) -> bool {
        !self.is_hex()
    }

    fn construct_quadrature(&self) -> (Vec<f64>, Vec<Vector3<f64>>) {
        match self {
            Self::Tet1 => tet_quadrature_strength_1(),
            Self::Tet2 => tet_quadrature_strength_2(),
            Self::Tet3 => tet_quadrature_strength_3(),
            Self::Tet5 => tet_quadrature_strength_5(),
            Self::Tet10 => tet_quadrature_strength_10(),
            Self::Hex3 => hex_quadrature_strength_3(),
            Self::Hex5 => hex_quadrature_strength_5(),
            Self::Hex11 => hex_quadrature_strength_11(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum QuadratureStrength {
    NonSimplified(QuadratureRule),
    Simplified { source: QuadratureRule, simplified: u16 },
}

impl QuadratureStrength {
    fn non_simplified(source: QuadratureRule) -> Self {
        Self::NonSimplified(source)
    }

    fn simplified(source: QuadratureRule, simplified: u16) -> Self {
        Self::Simplified { source, simplified }
    }

    fn hex5_simp4() -> Self {
        Self::simplified(QuadratureRule::Hex5, 4)
    }

    fn source_quadrature(&self) -> QuadratureRule {
        match self {
            Self::NonSimplified(q) => *q,
            Self::Simplified { source, .. } => *source,
        }
    }

    fn is_simplified(&self) -> bool {
        match self {
            Self::NonSimplified(_) => false,
            _ => true,
        }
    }
}

fn initial_scene(name: &str) -> Scene {
    Scene {
        initial_state: Default::default(),
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: 7.0,
        name: String::from(name),
    }
}

fn default_material() -> Material {
    Material {
        density: 1000.0,
        mass_damping_coefficient: None,
        stiffness_damping_coefficient: Some(0.05),
        elastic_model: StableNeoHookeanMaterial::from(YoungPoisson {
            young: 5e6,
            poisson: 0.48,
        })
        .into(),
    }
}

fn add_systems(systems: &mut Systems) {
    let settings = IntegratorSettings::default().set_project_stiffness(false);
    systems.add_system(Box::new(FiniteElementIntegrator3d::with_settings(settings)));
    systems.add_system(Box::new(FiniteElementMeshDeformer));
}

fn load_mesh(params: &SceneParameters, filename: &str) -> Result<Tet4Mesh<f64>, Box<dyn Error>> {
    load_mesh_from_file(&params.asset_dir, &format!("meshes/cylinder_shell/{}", filename))
}

fn load_fine_embedded_mesh(params: &SceneParameters) -> Result<Tet4Mesh<f64>, Box<dyn Error>> {
    load_mesh(params, "cylinder_shell_10k.msh")
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
    let minus_nodes = filtered_vertex_indices(mesh_vertices, |v| v.y < -6.99);
    let plus_nodes = filtered_vertex_indices(mesh_vertices, |v| v.y > 6.99);

    let minus_velocity = Vector3::new(0.0, 0.25, 0.0);
    let minus_moving_bc = ConstantUniformVelocity::new(&minus_nodes, minus_velocity);
    let plus_node_positions = plus_nodes
        .iter()
        .copied()
        .map(|index| mesh_vertices[index])
        .collect();
    let plus_moving_bc = ConstantUniformAngularVelocity::new(
        &plus_nodes,
        Vector3::new(0.0, 0.8, 0.0),
        Point3::new(0.0, 7.0, 0.0),
        plus_node_positions,
    );
    // Time when the cylinder stops moving/rotating
    let t_final = 4.0;
    let minus_displacement = t_final * minus_velocity;
    let minus_static_bc = ConstantUniformDisplacement::new(&minus_nodes, minus_displacement);

    // We need to figure out the displacements of the rotating end
    let mut plus_displacements = DVector::zeros(plus_moving_bc.nrows());
    plus_moving_bc.apply_displacement_bcs(DVectorSliceMut::from(&mut plus_displacements), t_final);
    let plus_static_bc = ConstantDisplacement::new_3d(&plus_nodes, plus_displacements);

    let moving_bc = Union::try_new(vec![minus_moving_bc.into(), plus_moving_bc.into()]).unwrap();
    let static_bc = Union::try_new(vec![minus_static_bc.into(), plus_static_bc.into()]).unwrap();

    BoundaryConditions {
        initial_bc: Box::new(moving_bc),
        final_bc: Box::new(static_bc),
        switch_time: 4.0,
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
    info!("Generating render mesh");
    let render_surface_mesh = PolyMesh3d::from_surface_mesh(&load_fine_embedded_mesh(&params)?.extract_surface_mesh());

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

    // We don't need the poly mesh, so let's not clutter up the output storage
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

fn create_background_mesh(resolution: usize) -> HexMesh<f64> {
    let thickness = 2.0;
    let mut mesh = create_rectangular_uniform_hex_mesh(thickness, 1, 8, 1, resolution);
    mesh.translate(&Vector3::new(-thickness / 2.0, -8.0, -thickness / 2.0));
    mesh
}

fn count_interface_points(quadrature: &EmbeddedQuadrature<f64, U3>) -> usize {
    quadrature
        .interface_quadratures()
        .iter()
        .map(|quadrature| quadrature.points().len())
        .sum::<usize>()
}

fn cylinder_shell_fem_tet4(params: &SceneParameters, name: &str, filename: &str) -> Result<Scene, Box<dyn Error>> {
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

pub fn cylinder_shell_fem_tet4_5k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_5k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_5k.msh")
}

pub fn cylinder_shell_fem_tet4_10k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_10k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_10k.msh")
}

pub fn cylinder_shell_fem_tet4_20k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_20k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_20k.msh")
}

pub fn cylinder_shell_fem_tet4_40k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_40k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_40k.msh")
}

pub fn cylinder_shell_fem_tet4_80k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_80k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_80k.msh")
}

pub fn cylinder_shell_fem_tet4_160k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_160k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_160k.msh")
}

pub fn cylinder_shell_fem_tet4_320k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let name = "cylinder_shell_fem_tet4_320k";
    cylinder_shell_fem_tet4(params, &name, "cylinder_shell_320k.msh")
}

fn cylinder_shell_embedded_hex8(
    params: &SceneParameters,
    name: &str,
    resolution: usize,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_fine_embedded_mesh(params)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);

    let background_mesh = create_background_mesh(resolution);

    // Note: For now we only stabilize the mass matrix, which seems to be sufficient when
    // using a direct solver
    let mass_quadrature_opts = QuadratureOptions {
        stabilization: Some(StabilizationOptions {
            stabilization_factor: 1e-8,
            stabilization_quadrature: hex_quadrature_strength_5(),
        }),
    };

    info!("Embedding mesh...");
    let embedding = embed_mesh_3d(&background_mesh, &volume_poly_mesh);
    info!("Constructing mass quadrature...");
    let mass_quadrature = embed_quadrature_3d_with_opts(
        &background_mesh,
        &embedding,
        hex_quadrature_strength_5(),
        tet_quadrature_strength_5(),
        &mass_quadrature_opts,
    )?;
    info!(
        "Constructed mass quadrature: {} points in interface elements.",
        count_interface_points(&mass_quadrature)
    );

    info!("Constructing stiffness quadrature...");
    let stiffness_quadrature = embed_quadrature_3d(
        &background_mesh,
        &embedding,
        hex_quadrature_strength_3(),
        tet_quadrature_strength_2(),
    )?;
    info!(
        "Constructed stiffness quadrature: {} points in interface elements.",
        count_interface_points(&stiffness_quadrature)
    );
    info!("Simplifying stiffness quadrature...");
    let stiffness_quadrature = stiffness_quadrature.simplified(2, &GlopSolver::new())?;
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

    let mesh = fe_model.background_mesh().clone();
    add_model(params, &mut scene, fe_model, &mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn cylinder_shell_embedded_hex8_res2(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 2;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res2", resolution)
}

pub fn cylinder_shell_embedded_hex8_res3(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 3;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res3", resolution)
}

pub fn cylinder_shell_embedded_hex8_res5(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 5;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res5", resolution)
}

pub fn cylinder_shell_embedded_hex8_res10(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 10;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res10", resolution)
}

pub fn cylinder_shell_embedded_hex8_res14(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 14;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res14", resolution)
}

pub fn cylinder_shell_embedded_hex8_res16(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 16;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res16", resolution)
}

pub fn cylinder_shell_embedded_hex8_res18(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 18;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res18", resolution)
}

pub fn cylinder_shell_embedded_hex8_res20(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 20;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res20", resolution)
}

pub fn cylinder_shell_embedded_hex8_res22(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 22;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res22", resolution)
}

pub fn cylinder_shell_embedded_hex8_res24(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 24;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res24", resolution)
}

pub fn cylinder_shell_embedded_hex8_res26(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 26;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res26", resolution)
}

pub fn cylinder_shell_embedded_hex8_res28(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 28;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res28", resolution)
}

pub fn cylinder_shell_embedded_hex8_res29(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 29;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res29", resolution)
}

pub fn cylinder_shell_embedded_hex8_res30(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 30;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res30", resolution)
}

pub fn cylinder_shell_embedded_hex8_res32(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 32;
    cylinder_shell_embedded_hex8(params, "cylinder_shell_embedded_hex8_res32", resolution)
}

fn cylinder_shell_embedded_hex27(
    params: &SceneParameters,
    name: &str,
    resolution: usize,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_fine_embedded_mesh(params)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);

    let background_mesh = create_background_mesh(resolution);
    let background_mesh = Hex27Mesh::from(&background_mesh);

    let embedding = embed_mesh_3d(&background_mesh, &volume_poly_mesh);

    let mass_quadrature = embed_quadrature_3d(
        &background_mesh,
        &embedding,
        hex_quadrature_strength_11(),
        tet_quadrature_strength_10(),
    )?;
    // .simplified(10, &GlopSolver::new())?;
    let stiffness_quadrature = embed_quadrature_3d(
        &background_mesh,
        &embedding,
        hex_quadrature_strength_5(),
        tet_quadrature_strength_5(),
    )?
    .simplified(5, &GlopSolver::new())?;
    let elliptic_quadrature = stiffness_quadrature.clone();

    let mut fe_model = EmbeddedModelBuilder::from_embedding(&background_mesh, embedding)
        .mass_quadrature(mass_quadrature)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();

    fe_model.set_mass_regularization_factor(1e-6);

    let mesh = fe_model.background_mesh().clone();
    add_model(params, &mut scene, fe_model, &mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn cylinder_shell_embedded_hex27_res5(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 5;
    cylinder_shell_embedded_hex27(params, "cylinder_shell_embedded_hex27_res5", resolution)
}

fn cylinder_shell_embedded_hex20(
    params: &SceneParameters,
    name: &str,
    resolution: usize,
    quadrature_strength: QuadratureStrength,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_fine_embedded_mesh(params)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);

    let background_mesh = create_background_mesh(resolution);
    let background_mesh = Hex20Mesh::from(&background_mesh);

    info!("Embedding mesh...");
    let embedding = embed_mesh_3d(&background_mesh, &volume_poly_mesh);

    let mass_quadrature_opts = QuadratureOptions {
        stabilization: Some(StabilizationOptions {
            stabilization_factor: 1e-8,
            stabilization_quadrature: hex_quadrature_strength_11(),
        }),
    };
    info!("Constructing mass quadrature...");
    let mass_quadrature = embed_quadrature_3d_with_opts(
        &background_mesh,
        &embedding,
        hex_quadrature_strength_11(),
        tet_quadrature_strength_10(),
        &mass_quadrature_opts,
    )?;
    info!(
        "Constructed mass quadrature: {} points in interface elements.",
        count_interface_points(&mass_quadrature)
    );
    // .simplified(10, &GlopSolver::new())?;

    info!("Constructing stiffness quadrature {:?}...", quadrature_strength);
    let stiffness_quadrature = match quadrature_strength.source_quadrature() {
        QuadratureRule::Hex3 => embed_quadrature_3d(
            &background_mesh,
            &embedding,
            hex_quadrature_strength_3(),
            tet_quadrature_strength_3(),
        )?,
        QuadratureRule::Hex5 => embed_quadrature_3d(
            &background_mesh,
            &embedding,
            hex_quadrature_strength_5(),
            tet_quadrature_strength_5(),
        )?,
        QuadratureRule::Hex11 => embed_quadrature_3d(
            &background_mesh,
            &embedding,
            hex_quadrature_strength_11(),
            tet_quadrature_strength_10(),
        )?,
        _ => {
            return Err(Box::from(format!(
                "Unsupported quadrature for cylinder_shell_embedded_hex20: {:?}",
                quadrature_strength
            )))
        }
    };
    info!(
        "Constructed stiffness quadrature: {} points in interface elements.",
        count_interface_points(&stiffness_quadrature)
    );

    let stiffness_quadrature = if let QuadratureStrength::Simplified { source: _, simplified } = quadrature_strength {
        info!("Simplifying stiffness quadrature to strength {}...", simplified);
        let simplified = stiffness_quadrature.simplified(simplified as usize, &GlopSolver::new())?;
        info!(
            "Simplified stiffness quadrature: {} points in interface elements",
            count_interface_points(&simplified)
        );
        simplified
    } else {
        info!("Skipping stiffness quadrature simplification!");
        stiffness_quadrature
    };

    let elliptic_quadrature = stiffness_quadrature.clone();

    let fe_model = EmbeddedModelBuilder::from_embedding(&background_mesh, embedding)
        .mass_quadrature(mass_quadrature)
        .stiffness_quadrature(stiffness_quadrature)
        .elliptic_quadrature(elliptic_quadrature)
        .build();

    let mesh = fe_model.background_mesh().clone();
    add_model(params, &mut scene, fe_model, &mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn cylinder_shell_embedded_hex20_res1(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 1;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res1",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res2(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 2;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res2",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res3(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 3;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res3",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res4(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 4;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res4",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res5(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 5;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res5",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res6_strength3(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 6;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res6_strength3",
        resolution,
        QuadratureStrength::simplified(QuadratureRule::Hex3, 3),
    )
}

pub fn cylinder_shell_embedded_hex20_res6_strength5(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 6;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res6_strength5",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res6_strength5_no_simp(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 6;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res6_strength5_no_simp",
        resolution,
        QuadratureStrength::non_simplified(QuadratureRule::Hex5),
    )
}

pub fn cylinder_shell_embedded_hex20_res7_strength3(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 7;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res7_strength3",
        resolution,
        QuadratureStrength::simplified(QuadratureRule::Hex3, 3),
    )
}

pub fn cylinder_shell_embedded_hex20_res7_strength5(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 7;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res7_strength5",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res7_strength5_no_simp(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 7;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res7_strength5_no_simp",
        resolution,
        QuadratureStrength::non_simplified(QuadratureRule::Hex5),
    )
}

pub fn cylinder_shell_embedded_hex20_res8(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 8;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res8",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

pub fn cylinder_shell_embedded_hex20_res10(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let resolution = 10;
    cylinder_shell_embedded_hex20(
        params,
        "cylinder_shell_embedded_hex20_res10",
        resolution,
        QuadratureStrength::hex5_simp4(),
    )
}

fn cylinder_shell_fem_tet10(
    params: &SceneParameters,
    name: &str,
    filename: &str,
    quadrature_strength: QuadratureStrength,
) -> Result<Scene, Box<dyn Error>> {
    let mut scene = initial_scene(name);

    let tet_mesh = load_mesh(params, filename)?;
    let volume_poly_mesh = PolyMesh3d::from(&tet_mesh);
    let tet_mesh = Tet10Mesh::from(&tet_mesh);

    let tet_mesh = reorder_mesh_par(&tet_mesh).apply(&tet_mesh);

    if !quadrature_strength.source_quadrature().is_tet() || quadrature_strength.is_simplified() {
        return Err(Box::from(format!(
            "Unsupported quadrature for cylinder_shell_fem_tet10: {:?}",
            quadrature_strength
        )));
    }

    info!("Constructing stiffness quadrature {:?}...", quadrature_strength);
    let quadrature = quadrature_strength
        .source_quadrature()
        .construct_quadrature();

    let fe_model = NodalModel::from_mesh_and_quadrature(tet_mesh.clone(), quadrature);
    // TODO: Maybe use order 4 quadrature? (Or even 3?)
    add_model(params, &mut scene, fe_model, &tet_mesh, volume_poly_mesh)?;
    add_systems(&mut scene.simulation_systems);
    Ok(scene)
}

pub fn cylinder_shell_fem_tet10_5k_strength2(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    cylinder_shell_fem_tet10(
        params,
        "cylinder_shell_fem_tet10_5k_strength2",
        "cylinder_shell_5k.msh",
        QuadratureStrength::non_simplified(QuadratureRule::Tet2),
    )
}

pub fn cylinder_shell_fem_tet10_5k_strength3(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    cylinder_shell_fem_tet10(
        params,
        "cylinder_shell_fem_tet10_5k_strength3",
        "cylinder_shell_5k.msh",
        QuadratureStrength::non_simplified(QuadratureRule::Tet3),
    )
}

pub fn cylinder_shell_fem_tet10_5k_strength5(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    cylinder_shell_fem_tet10(
        params,
        "cylinder_shell_fem_tet10_5k_strength5",
        "cylinder_shell_5k.msh",
        QuadratureStrength::non_simplified(QuadratureRule::Tet5),
    )
}

pub fn cylinder_shell_fem_tet10_10k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    cylinder_shell_fem_tet10(
        params,
        "cylinder_shell_fem_tet10_10k",
        "cylinder_shell_10k.msh",
        QuadratureStrength::non_simplified(QuadratureRule::Tet5),
    )
}

pub fn cylinder_shell_fem_tet10_20k(params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    cylinder_shell_fem_tet10(
        params,
        "cylinder_shell_fem_tet10_20k",
        "cylinder_shell_20k.msh",
        QuadratureStrength::non_simplified(QuadratureRule::Tet5),
    )
}
