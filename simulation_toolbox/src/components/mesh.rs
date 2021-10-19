use std::error::Error;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

use fenris::connectivity::Segment2d2Connectivity;
use fenris::geometry::polymesh::{PolyMesh2d, PolyMesh3d};
use fenris::mesh::{Mesh2d, QuadMesh2d, TriangleMesh2d};
use fenris::model::{FiniteElementInterpolator, MakeInterpolator};
use fenris::nalgebra::{Point2, U2, U3};
use hamilton::storages::VecStorage;
use hamilton::Component;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyMesh2dComponent {
    /// Mesh name used for the output file as: {entity_name}_{mesh_name}_polymesh_{sequence_name}.{file_extension}
    pub mesh_name: String,
    /// Optional subfolder (relative to output directory) for the output of the meshes
    pub subfolder: Option<PathBuf>,
    /// The polymesh to write to a file
    pub mesh: PolyMesh2d<f64>,
    /// Optional interpolator to interpolate the polymesh with on every output write
    pub interpolator: Option<FiniteElementInterpolator<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyMesh3dComponent {
    /// Mesh name used for the output file as: {entity_name}_{mesh_name}_polymesh_{sequence_name}.{file_extension}
    pub mesh_name: String,
    /// Optional subfolder (relative to output directory) for the output of the meshes
    pub subfolder: Option<PathBuf>,
    /// The polymesh to write to a file
    pub mesh: PolyMesh3d<f64>,
    /// Optional interpolator to interpolate the polymesh with on every output write
    pub interpolator: Option<FiniteElementInterpolator<f64>>,
}

impl PolyMesh3dComponent {
    /// Creates a PolyMesh3dComponent for static geometry
    pub fn new<S: Into<String>>(mesh_name: S, mesh: PolyMesh3d<f64>) -> Self {
        PolyMesh3dComponent {
            mesh_name: mesh_name.into(),
            subfolder: None,
            mesh,
            interpolator: None,
        }
    }

    /// Attaches an interpolator to this PolyMesh3dComponent
    pub fn with_interpolator<M: MakeInterpolator<f64, U3>>(mut self, model: &M) -> Result<Self, Box<dyn Error>> {
        self.interpolator = Some(model.make_interpolator(self.mesh.vertices())?);
        Ok(self)
    }

    /// Attaches a subfolder to this PolyMesh3dComponent
    pub fn with_subfolder<P: Into<PathBuf>>(mut self, subfolder: P) -> Self {
        self.subfolder = Some(subfolder.into());
        self
    }
}

impl PolyMesh2dComponent {
    /// Creates a PolyMesh2dComponent for static geometry
    pub fn new<S: Into<String>>(mesh_name: S, mesh: PolyMesh2d<f64>) -> Self {
        PolyMesh2dComponent {
            mesh_name: mesh_name.into(),
            subfolder: None,
            mesh,
            interpolator: None,
        }
    }

    /// Attaches an interpolator to this PolyMesh2dComponent
    pub fn with_interpolator<M: MakeInterpolator<f64, U2>>(mut self, model: &M) -> Result<Self, Box<dyn Error>> {
        self.interpolator = Some(model.make_interpolator(self.mesh.vertices())?);
        Ok(self)
    }

    /// Attaches a subfolder to this PolyMesh3dComponent
    pub fn with_subfolder<P: Into<PathBuf>>(mut self, subfolder: P) -> Self {
        self.subfolder = Some(subfolder.into());
        self
    }
}

/// Component storing interpolators for arbitrary 3D polymeshes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyMesh2dCollection(pub Vec<PolyMesh2dComponent>);

/// Component storing interpolators for arbitrary 3D polymeshes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyMesh3dCollection(pub Vec<PolyMesh3dComponent>);

impl Deref for PolyMesh2dCollection {
    type Target = Vec<PolyMesh2dComponent>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Component for PolyMesh2dCollection {
    type Storage = VecStorage<Self>;
}

impl Deref for PolyMesh3dCollection {
    type Target = Vec<PolyMesh3dComponent>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Component for PolyMesh3dCollection {
    type Storage = VecStorage<Self>;
}

/// Component storing an interpolator for a set of points
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointInterpolator {
    pub reference_points: Vec<Point2<f64>>,
    pub interpolator: FiniteElementInterpolator<f64>,
}

impl Component for PointInterpolator {
    type Storage = VecStorage<Self>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
// TODO: Replace this with polygonal/polyhedral meshes later on
pub enum VolumeMesh2d {
    QuadMesh(QuadMesh2d<f64>),
    TriMesh(TriangleMesh2d<f64>),
}

impl From<QuadMesh2d<f64>> for VolumeMesh2d {
    fn from(mesh: QuadMesh2d<f64>) -> Self {
        Self::QuadMesh(mesh)
    }
}

impl From<TriangleMesh2d<f64>> for VolumeMesh2d {
    fn from(mesh: TriangleMesh2d<f64>) -> Self {
        Self::TriMesh(mesh)
    }
}

impl VolumeMesh2d {
    pub fn vertices(&self) -> &[Point2<f64>] {
        match self {
            Self::QuadMesh(ref mesh) => mesh.vertices(),
            Self::TriMesh(ref mesh) => mesh.vertices(),
        }
    }

    pub fn vertices_mut(&mut self) -> &mut [Point2<f64>] {
        match self {
            Self::QuadMesh(ref mut mesh) => mesh.vertices_mut(),
            Self::TriMesh(ref mut mesh) => mesh.vertices_mut(),
        }
    }
}

impl Component for VolumeMesh2d {
    type Storage = VecStorage<Self>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceMesh2d(pub Mesh2d<f64, Segment2d2Connectivity>);

impl From<Mesh2d<f64, Segment2d2Connectivity>> for SurfaceMesh2d {
    fn from(mesh: Mesh2d<f64, Segment2d2Connectivity>) -> Self {
        Self(mesh)
    }
}

impl Component for SurfaceMesh2d {
    type Storage = VecStorage<Self>;
}

impl Deref for SurfaceMesh2d {
    type Target = Mesh2d<f64, Segment2d2Connectivity>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SurfaceMesh2d {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeMesh3d(pub PolyMesh3d<f64>);

impl Component for VolumeMesh3d {
    type Storage = VecStorage<Self>;
}

impl Deref for VolumeMesh3d {
    type Target = PolyMesh3d<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for VolumeMesh3d {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<P> From<P> for VolumeMesh3d
where
    P: Into<PolyMesh3d<f64>>,
{
    fn from(into_poly: P) -> Self {
        Self(into_poly.into())
    }
}
