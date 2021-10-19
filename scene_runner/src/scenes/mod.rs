use std::error::Error;
use std::path::PathBuf;

use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DefaultAllocator, DimName, Point, RealField};
use hamilton::{StorageContainer, Systems};
use once_cell::sync::Lazy;
use simulation_toolbox::io::json_helper::JsonWrapper;

// Single scenes
mod cantilever3d;
mod hollow_ball;
mod rotating_bicycle;

// Multiple scene modules
mod armadillo_slingshot;
mod cylinder_shell;

// Special scenes
mod quad_reduc;

// Other modules
mod helpers;

static SCENE_REGISTRY: Lazy<Vec<SceneConstructor>> = Lazy::new(|| {
    let mut scenes = Vec::new();
    scenes.push(SceneConstructor {
        name: "cantilever3d".to_string(),
        constructor: cantilever3d::cantilever3d,
    });
    scenes.push(SceneConstructor {
        name: "bicycle_embedded_super_coarse".to_string(),
        constructor: rotating_bicycle::build_bicycle_scene_embedded_super_coarse,
    });
    scenes.push(SceneConstructor {
        name: "bicycle_embedded_coarse".to_string(),
        constructor: rotating_bicycle::build_bicycle_scene_embedded_coarse,
    });
    scenes.push(SceneConstructor {
        name: "bicycle_embedded_fine".to_string(),
        constructor: rotating_bicycle::build_bicycle_scene_embedded_fine,
    });
    scenes.push(SceneConstructor {
        name: "bicycle_fem_coarse".to_string(),
        constructor: rotating_bicycle::build_bicycle_scene_fem_coarse,
    });
    scenes.push(SceneConstructor {
        name: "bicycle_fem_fine".to_string(),
        constructor: rotating_bicycle::build_bicycle_scene_fem_fine,
    });

    scenes.extend(cylinder_shell::scenes());
    scenes.extend(armadillo_slingshot::scenes());
    scenes.extend(hollow_ball::scenes());
    scenes.extend(quad_reduc::scenes());

    scenes.sort_by_key(|constructor| constructor.name.clone());
    scenes
});

#[derive(Debug)]
pub struct Scene {
    pub initial_state: StorageContainer,
    pub simulation_systems: Systems,
    pub analysis_systems: Systems,
    pub name: String,
    pub duration: f64,
}

#[derive(Debug, Clone)]
pub struct SceneParameters {
    pub output_dir: PathBuf,
    pub asset_dir: PathBuf,
    pub config_file: Option<JsonWrapper<'static>>,
}

#[doc(hidden)]
pub struct SceneConstructor {
    name: String,
    constructor: fn(&SceneParameters) -> Result<Scene, Box<dyn Error>>,
}

pub fn available_scenes() -> Vec<String> {
    let mut names = Vec::new();
    for scene in SCENE_REGISTRY.iter() {
        names.push(scene.name.clone());
    }
    names
}

pub fn load_scene(name: &str, params: &SceneParameters) -> Result<Scene, Box<dyn Error>> {
    for scene in SCENE_REGISTRY.iter() {
        if scene.name == name {
            return (scene.constructor)(params);
        }
    }

    Err(Box::from(format!("Could not find scene {}", name)))
}

fn filtered_vertex_indices<T, D, F>(vertices: &[Point<T, D>], filter: F) -> Vec<usize>
where
    T: RealField,
    D: DimName,
    F: Fn(&Point<T, D>) -> bool,
    DefaultAllocator: Allocator<T, D>,
{
    vertices
        .iter()
        .enumerate()
        .filter(|(_, v)| filter(v))
        .map(|(i, _)| i)
        .collect()
}
