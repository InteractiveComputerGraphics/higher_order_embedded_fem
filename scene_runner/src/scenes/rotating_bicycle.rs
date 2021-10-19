use std::error::Error;

use fenris::connectivity::Connectivity;
use fenris::embedding::construct_embedded_model_2d;
use fenris::mesh::{Mesh2d, TriangleMesh2d};
use fenris::model::NodalModel2d;
use fenris::nalgebra::{DVector, Point2, RealField, Vector2};
use fenris::quadrature::{quad_quadrature_strength_5_f64, tri_quadrature_strength_5_f64};
use fenris::solid::materials::{StVKMaterial, YoungPoisson};
use numeric_literals::replace_float_literals;
use simulation_toolbox::components::{set_gravity, PointInterpolator};
use simulation_toolbox::fem::{ElasticMaterialModel, FiniteElementIntegrator, FiniteElementMeshDeformer, Material};

use crate::meshes;
use crate::scenes::helpers::{
    generate_background_mesh_for_tri2d, generate_single_cell_background_mesh_for_tri2d, BodyInitializer2d,
};
use crate::scenes::{Scene, SceneParameters};

struct BicycleSceneSettings {
    /// Name of the scene
    name: String,
    /// Duration of the scene
    duration: f64,

    /// Whether to use the finite cell method (classic FEM otherwise)
    use_embedded_model: bool,
    /// Whether to use the fine mesh for the embedded/volume mesh
    use_fine_mesh: bool,

    /// Density of the bike
    density: f64,
    /// Young's modulus
    young: f64,
    /// Poisson ratio
    poisson: f64,
    /// Resolution of the background mesh used in the finite cell method
    background_resolution: Option<f64>,
    /// Rotational velocity of the bike
    omega: f64,
}

impl Default for BicycleSceneSettings {
    fn default() -> Self {
        Self {
            name: "bicycle".to_string(),
            duration: 15.0,
            use_embedded_model: true,
            use_fine_mesh: true,
            density: 100.0,
            young: 1e9,
            poisson: 0.4,
            background_resolution: Some(0.8),
            omega: 10.0,
        }
    }
}

pub fn build_bicycle_scene_embedded_super_coarse(_params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let mut settings = BicycleSceneSettings::default();
    settings.name = "bicycle_embedded_super_coarse".to_string();
    settings.use_embedded_model = true;
    settings.use_fine_mesh = true;
    settings.background_resolution = None;

    build_bicycle_scene_from_settings(&settings)
}

pub fn build_bicycle_scene_embedded_coarse(_params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let mut settings = BicycleSceneSettings::default();
    settings.name = "bicycle_embedded_coarse".to_string();
    settings.use_embedded_model = true;
    settings.use_fine_mesh = true;
    settings.background_resolution = Some(1.0);

    build_bicycle_scene_from_settings(&settings)
}

pub fn build_bicycle_scene_embedded_fine(_params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let mut settings = BicycleSceneSettings::default();
    settings.name = "bicycle_embedded_fine".to_string();
    settings.use_embedded_model = true;
    settings.use_fine_mesh = true;
    settings.background_resolution = Some(0.3);

    build_bicycle_scene_from_settings(&settings)
}

pub fn build_bicycle_scene_fem_coarse(_params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let mut settings = BicycleSceneSettings::default();
    settings.name = "bicycle_fem_coarse".to_string();
    settings.use_embedded_model = false;
    settings.use_fine_mesh = false;

    build_bicycle_scene_from_settings(&settings)
}

pub fn build_bicycle_scene_fem_fine(_params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    let mut settings = BicycleSceneSettings::default();
    settings.name = "bicycle_fem_fine".to_string();
    settings.use_embedded_model = false;
    settings.use_fine_mesh = true;

    build_bicycle_scene_from_settings(&settings)
}

fn get_reference_cog() -> Result<Vector2<f64>, Box<dyn std::error::Error>> {
    Ok(compute_mesh_cog(&*meshes::BIKE_TRI2D_MESH_FINE))
}

fn build_bicycle_scene_from_settings(settings: &BicycleSceneSettings) -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene {
        initial_state: Default::default(),
        simulation_systems: Default::default(),
        analysis_systems: Default::default(),
        duration: settings.duration,
        name: settings.name.clone(),
    };

    let embedded_mesh = if settings.use_fine_mesh {
        meshes::BIKE_TRI2D_MESH_FINE.clone()
    } else {
        meshes::BIKE_TRI2D_MESH_COARSE.clone()
    };
    let background_mesh = if let Some(bg_resolution) = settings.background_resolution {
        generate_background_mesh_for_tri2d(&embedded_mesh, bg_resolution)?
    } else {
        generate_single_cell_background_mesh_for_tri2d(&embedded_mesh)?
    };

    let volume_mesh = embedded_mesh.clone();
    let surface_mesh = embedded_mesh.extract_surface_mesh();

    let interior_quadrature = quad_quadrature_strength_5_f64();
    let triangle_quadrature = tri_quadrature_strength_5_f64();

    let elastic_model = ElasticMaterialModel::from(StVKMaterial::from(YoungPoisson {
        young: settings.young,
        poisson: settings.poisson,
    }));

    let own_cog = compute_mesh_cog(&embedded_mesh);
    let ref_cog = get_reference_cog()?;

    let points_to_interpolate = vec![own_cog.into()];

    if settings.use_embedded_model {
        let model = construct_embedded_model_2d(
            &background_mesh,
            &embedded_mesh,
            &triangle_quadrature,
            interior_quadrature,
        )?;
        let fe_mesh = &background_mesh;

        let cog_component = PointInterpolator {
            reference_points: points_to_interpolate.clone(),
            interpolator: model.make_interpolator(&points_to_interpolate)?,
        };

        let entity = {
            let body = BodyInitializer2d::initialize_in_state(&scene.initial_state);
            body.add_finite_element_model(model, volume_mesh)?
                .set_velocity(&apply_angular_velocity(settings.omega, &ref_cog, &fe_mesh))?
                .add_material_surface(surface_mesh)?
                .set_material(Material {
                    density: settings.density,
                    mass_damping_coefficient: None,
                    stiffness_damping_coefficient: None,
                    elastic_model,
                })
                .add_name("embedded_bike");
            body.entity()
        };

        scene.initial_state.insert_component(entity, cog_component);
    } else {
        let model = NodalModel2d::from_mesh_and_quadrature(volume_mesh.clone(), triangle_quadrature.clone());
        let fe_mesh = &embedded_mesh;

        let cog_component = PointInterpolator {
            reference_points: points_to_interpolate.clone(),
            interpolator: model.make_interpolator(&points_to_interpolate)?,
        };

        let entity = {
            let body = BodyInitializer2d::initialize_in_state(&scene.initial_state);
            body.add_finite_element_model(model, volume_mesh)?
                .set_velocity(&apply_angular_velocity(settings.omega, &ref_cog, &fe_mesh))?
                .add_material_surface(surface_mesh)?
                .set_material(Material {
                    density: settings.density,
                    mass_damping_coefficient: None,
                    stiffness_damping_coefficient: None,
                    elastic_model,
                })
                .add_name("fem_bike");
            body.entity()
        };

        scene.initial_state.insert_component(entity, cog_component);
    };

    set_gravity(&mut scene.initial_state, 0.0);

    scene
        .simulation_systems
        .add_system(Box::new(FiniteElementIntegrator::default()));
    scene
        .simulation_systems
        .add_system(Box::new(FiniteElementMeshDeformer));

    Ok(scene)
}

/// Computes the cog of a 2d triangle mesh
#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
fn compute_mesh_cog<T>(mesh: &TriangleMesh2d<T>) -> Vector2<T>
where
    T: RealField,
{
    let compute_triangle_area = |v1: &Point2<T>, v2: &Point2<T>, v3: &Point2<T>| -> T {
        let x = v1.coords - v3.coords;
        let y = v2.coords - v3.coords;
        0.5 * ((x.x * y.y) - (x.y * y.x)).abs()
    };

    let mut mesh_volume = T::zero();
    let mut mesh_cog = Vector2::zeros();
    for element in mesh.connectivity() {
        let v0 = &mesh.vertices()[element[0]];
        let v1 = &mesh.vertices()[element[1]];
        let v2 = &mesh.vertices()[element[2]];

        let element_cog = (v0.coords + v1.coords + v2.coords) * (3.0.recip());
        let element_volume = compute_triangle_area(v0, v1, v2);
        mesh_volume += element_volume;
        mesh_cog += element_cog * element_volume;
    }

    mesh_cog * mesh_volume.recip()
}

/// Computes a velocity field for the given mesh that represents a rotation
/// with the specified angular velocity around a center of rotation
fn apply_angular_velocity<T, C>(omega: T, center: &Vector2<T>, mesh: &Mesh2d<T, C>) -> DVector<T>
where
    T: RealField,
    C: Connectivity,
{
    let mut velocities = DVector::zeros(2 * mesh.vertices().len());
    for (i, v) in mesh.vertices().iter().enumerate() {
        let r = v.coords - center;
        let v_dir = r.yx();
        let v = v_dir * omega;

        velocities[2 * i + 0] = v.x;
        velocities[2 * i + 1] = v.y;
    }

    velocities
}
