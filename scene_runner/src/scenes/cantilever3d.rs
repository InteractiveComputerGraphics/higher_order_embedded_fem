use std::convert::TryFrom;
use std::error::Error;

use crate::scenes::{filtered_vertex_indices, Scene, SceneParameters};

use simulation_toolbox::components::Name;
use simulation_toolbox::fem::{FiniteElementIntegrator, FiniteElementMeshDeformer, Material};

use crate::scenes::helpers::BodyInitializer3d;
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::procedural::create_rectangular_uniform_hex_mesh;
use fenris::mesh::Tet4Mesh;
use fenris::model::NodalModel;
use fenris::nalgebra::Vector3;
use fenris::quadrature::{hex_quadrature_strength_5, tet_quadrature_strength_5};
use fenris::solid::materials::{StableNeoHookeanMaterial, YoungPoisson};

pub fn cantilever3d(_params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene {
        initial_state: Default::default(),
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: 10.0,
        name: String::from("cantilever3d"),
    };

    let volume_mesh = create_rectangular_uniform_hex_mesh(1.0, 4, 1, 1, 1);
    let volume_poly_mesh = PolyMesh3d::from(&volume_mesh);

    let material = Material {
        density: 1000.0,
        mass_damping_coefficient: None,
        stiffness_damping_coefficient: None,
        elastic_model: StableNeoHookeanMaterial::from(YoungPoisson {
            young: 3e6,
            poisson: 0.4,
        })
        .into(),
    };

    // Tet model
    {
        let tet_poly_mesh = volume_poly_mesh.triangulate()?;
        let tet_mesh = Tet4Mesh::try_from(&tet_poly_mesh).unwrap();

        let quadrature = tet_quadrature_strength_5();
        let fe_model = NodalModel::from_mesh_and_quadrature(tet_mesh.clone(), quadrature);
        BodyInitializer3d::initialize_in_state(&scene.initial_state)
            .add_name(Name("cantilever_tet4".to_string()))
            .add_finite_element_model(fe_model, volume_poly_mesh.clone())?
            .set_static_nodes(filtered_vertex_indices(tet_mesh.vertices(), |v| v.x < 1e-6))
            .set_material(material.clone());
    }

    // Hex model
    {
        let mut volume_mesh_hex = volume_mesh.clone();
        let static_nodes = filtered_vertex_indices(volume_mesh_hex.vertices(), |v| v.x < 1e-6);
        volume_mesh_hex.translate(&Vector3::new(0.0, 2.0, 0.0));
        let quadrature = hex_quadrature_strength_5();

        let fe_model = NodalModel::from_mesh_and_quadrature(volume_mesh_hex.clone(), quadrature);

        BodyInitializer3d::initialize_in_state(&scene.initial_state)
            .add_name(Name("cantilever_hex8".to_string()))
            .add_finite_element_model(fe_model, &volume_mesh_hex)?
            .set_static_nodes(static_nodes)
            .set_material(material.clone());
    }

    scene
        .simulation_systems
        .add_system(Box::new(FiniteElementIntegrator::default()));
    scene
        .simulation_systems
        .add_system(Box::new(FiniteElementMeshDeformer));

    Ok(scene)
}
