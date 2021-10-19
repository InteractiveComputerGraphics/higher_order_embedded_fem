use std::error::Error;
use std::fmt::Debug;

use fenris::nalgebra::allocator::Allocator;
use fenris::nalgebra::{DVector, DVectorSliceMut, DefaultAllocator, DimName, Point, Rotation3, Unit, VectorN, U2, U3};
use hamilton::storages::VecStorage;
use hamilton::Component;
use serde::{Deserialize, Serialize};

use crate::util::all_items_unique;

/// Trait for evaluating Dirichlet boundary conditions
#[typetag::serde(tag = "type")]
pub trait DirichletBoundaryConditions: Debug {
    /// Returns the number of solution components per node.
    fn solution_dim(&self) -> usize;
    /// Returns the indices of the nodes that are affected by the boundary condition.
    fn nodes(&self) -> &[usize];
    /// Returns the total number of rows of a boundary condition vector.
    fn nrows(&self) -> usize {
        self.solution_dim() * self.nodes().len()
    }
    /// Evaluates the displacement boundary conditions at the specified time.
    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64);
    /// Evaluates the velocity boundary conditions at the specified time.
    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64);
}

/// Helper trait to implement an interface parallel to `DirichletBoundaryConditions` on `Option<&dyn DirichletBoundaryConditions>`
pub trait OptionalDirichletBoundaryConditions {
    /// Returns the number of solution components per node.
    fn solution_dim(&self) -> usize;
    /// Returns the indices of the nodes that are affected by the boundary condition.
    fn nodes(&self) -> &[usize];
    /// Returns the total number of rows of a boundary condition vector.
    fn nrows(&self) -> usize;
    /// Evaluates the displacement boundary conditions at the specified time.
    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64);
    /// Evaluates the velocity boundary conditions at the specified time.
    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64);
}

/// Component for storing Dirichlet boundary conditions
#[derive(Debug, Serialize, Deserialize)]
pub struct DirichletBoundaryConditionComponent {
    pub bc: Box<dyn DirichletBoundaryConditions>,
}

/// Dummy boundary condition that does not prescribe anything
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Empty;

/// Homogeneous Dirichlet boundary condition that prescribes zero values (i.e. overwrites them to zero)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Homogeneous {
    pub dim: usize,
    pub static_nodes: Vec<usize>,
}

/// Dirichlet boundary condition that adds a fixed displacement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantDisplacement {
    dim: usize,
    nodes: Vec<usize>,
    displacement: DVector<f64>,
}

/// Boundary condition that adds a constant and uniform displacement to the given nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "VectorN<f64, D>: Serialize",
    deserialize = "VectorN<f64, D>: Deserialize<'de>"
))]
pub struct ConstantUniformDisplacement<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    nodes: Vec<usize>,
    displacement: VectorN<f64, D>,
}

/// Boundary condition that adds a constant and uniform linear velocity to the given nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "VectorN<f64, D>: Serialize",
    deserialize = "VectorN<f64, D>: Deserialize<'de>"
))]
pub struct ConstantUniformVelocity<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    nodes: Vec<usize>,
    velocity: VectorN<f64, D>,
}

/// Boundary condition that applies a constant and uniform angular velocity to the given nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "VectorN<f64, D>: Serialize, Point<f64, D>: Serialize",
    deserialize = "VectorN<f64, D>: Deserialize<'de>, Point<f64, D>: Deserialize<'de>"
))]
pub struct ConstantUniformAngularVelocity<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    /// Node indices affected by this boundary condition
    nodes: Vec<usize>,
    /// Angular velocity of the nodes
    omega: VectorN<f64, D>,
    /// Center of rotation
    center: Point<f64, D>,
    /// Initial positions of boundary nodes
    x0: Vec<Point<f64, D>>,
}

/// Union of multiple disjoint Dirichlet boundary conditions
#[derive(Debug, Serialize, Deserialize)]
pub struct Union {
    dim: usize,
    nodes: Vec<usize>,
    bcs: Vec<Box<dyn DirichletBoundaryConditions>>,
}

impl<T: DirichletBoundaryConditions + 'static> From<T> for Box<dyn DirichletBoundaryConditions> {
    fn from(bc: T) -> Box<dyn DirichletBoundaryConditions> {
        Box::new(bc)
    }
}

impl<T: DirichletBoundaryConditions + 'static> From<T> for DirichletBoundaryConditionComponent {
    fn from(bcs: T) -> Self {
        Self { bc: bcs.into() }
    }
}

impl From<Box<dyn DirichletBoundaryConditions>> for DirichletBoundaryConditionComponent {
    fn from(bcs: Box<dyn DirichletBoundaryConditions>) -> Self {
        Self { bc: bcs }
    }
}

impl Component for DirichletBoundaryConditionComponent {
    type Storage = VecStorage<Self>;
}

#[typetag::serde]
impl DirichletBoundaryConditions for Empty {
    fn solution_dim(&self) -> usize {
        0
    }
    fn nodes(&self) -> &[usize] {
        &[]
    }
    fn apply_displacement_bcs(&self, _u: DVectorSliceMut<f64>, _t: f64) {}
    fn apply_velocity_bcs(&self, _v: DVectorSliceMut<f64>, _t: f64) {}
}

impl Homogeneous {
    pub fn new(dim: usize, static_nodes: &[usize]) -> Self {
        Self {
            dim,
            static_nodes: static_nodes.to_vec(),
        }
    }

    /// Creates 2D homogeneous boundary conditions for the specified node indices
    pub fn new_2d(static_nodes: &[usize]) -> Self {
        Self::new(2, static_nodes)
    }

    /// Creates 3D homogeneous boundary conditions for the specified node indices
    pub fn new_3d(static_nodes: &[usize]) -> Self {
        Self::new(3, static_nodes)
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for Homogeneous {
    fn solution_dim(&self) -> usize {
        self.dim
    }

    fn nodes(&self) -> &[usize] {
        &self.static_nodes
    }

    fn apply_displacement_bcs(&self, mut u: DVectorSliceMut<f64>, _t: f64) {
        assert!(u.nrows() == self.nrows());
        u.fill(0.0);
    }

    fn apply_velocity_bcs(&self, mut v: DVectorSliceMut<f64>, _t: f64) {
        assert!(v.nrows() == self.nrows());
        v.fill(0.0);
    }
}

impl ConstantDisplacement {
    pub fn new(dim: usize, nodes: &[usize], displacement: DVector<f64>) -> Self {
        assert!(displacement.nrows() == dim * nodes.len());
        Self {
            dim,
            nodes: nodes.to_vec(),
            displacement,
        }
    }

    pub fn new_2d(nodes: &[usize], displacement: DVector<f64>) -> Self {
        Self::new(2, nodes, displacement)
    }

    pub fn new_3d(nodes: &[usize], displacement: DVector<f64>) -> Self {
        Self::new(3, nodes, displacement)
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for ConstantDisplacement {
    fn solution_dim(&self) -> usize {
        self.dim
    }

    fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    fn apply_displacement_bcs(&self, mut u: DVectorSliceMut<f64>, _t: f64) {
        assert!(u.nrows() == self.nrows());
        u += &self.displacement;
    }

    fn apply_velocity_bcs(&self, _v: DVectorSliceMut<f64>, _t: f64) {}
}

impl Union {
    /// Returns the union of the supplied Dirichlet boundary conditions if they are disjoint
    ///
    /// Note that the node indices affected by the BCs are cached on construction.
    /// Therefore the stored BCs should not modify their set of boundary nodes.
    pub fn try_new(bcs: Vec<Box<dyn DirichletBoundaryConditions>>) -> Result<Self, Box<dyn Error>> {
        // Obtain the solution dimension of all supplied BCs
        let mut bc_iter = bcs.iter();
        let dim = if let Some(first_bc) = bc_iter.next() {
            let first_dim = first_bc.solution_dim();
            if bc_iter.any(|bc| bc.solution_dim() != first_dim) {
                return Err(Box::<dyn Error>::from("BCs do not have the same dimension"));
            }
            first_dim
        } else {
            return Ok(Self {
                dim: 0,
                nodes: Vec::new(),
                bcs: Vec::new(),
            });
        };

        // Collect the constrained nodes of all bcs
        let all_nodes = bcs
            .iter()
            .map(|bc| bc.nodes())
            .fold(Vec::new(), |mut vec, elems| {
                vec.extend(elems);
                vec
            });

        // Check if the boundary conditions are disjoint
        if all_items_unique(&all_nodes) {
            Ok(Self {
                dim,
                nodes: all_nodes,
                bcs,
            })
        } else {
            Err(Box::<dyn Error>::from("BCs are not disjoint"))
        }
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for Union {
    fn solution_dim(&self) -> usize {
        self.dim
    }

    fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    fn apply_displacement_bcs(&self, mut u: DVectorSliceMut<f64>, t: f64) {
        assert!(u.nrows() == self.nrows());

        let mut offset = 0;
        for bc in &self.bcs {
            let len = bc.nrows();
            bc.apply_displacement_bcs(u.rows_mut(offset, len), t);
            offset += len;
        }
    }

    fn apply_velocity_bcs(&self, mut v: DVectorSliceMut<f64>, t: f64) {
        assert!(v.nrows() == self.nrows());

        let mut offset = 0;
        for bc in &self.bcs {
            let len = bc.nrows();
            bc.apply_velocity_bcs(v.rows_mut(offset, len), t);
            offset += len;
        }
    }
}

impl<D> ConstantUniformDisplacement<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    VectorN<f64, D>: Serialize,
{
    pub fn new(nodes: &[usize], displacement: VectorN<f64, D>) -> Self {
        Self {
            nodes: nodes.to_vec(),
            displacement,
        }
    }

    fn solution_dim(&self) -> usize {
        D::dim()
    }

    fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    fn apply_displacement_bcs(&self, mut u: DVectorSliceMut<f64>, _t: f64) {
        assert!(u.nrows() == D::dim() * self.nodes.len());

        for i in 0..self.nodes.len() {
            let mut ui = u.fixed_rows_mut::<D>(D::dim() * i);
            ui.axpy(1.0, &self.displacement, 1.0);
        }
    }

    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, _t: f64) {
        assert!(v.nrows() == D::dim() * self.nodes.len());
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for ConstantUniformDisplacement<U3> {
    fn solution_dim(&self) -> usize {
        self.solution_dim()
    }

    fn nodes(&self) -> &[usize] {
        self.nodes()
    }

    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64) {
        self.apply_displacement_bcs(u, t)
    }

    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64) {
        self.apply_velocity_bcs(v, t)
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for ConstantUniformDisplacement<U2> {
    fn solution_dim(&self) -> usize {
        self.solution_dim()
    }

    fn nodes(&self) -> &[usize] {
        self.nodes()
    }

    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64) {
        self.apply_displacement_bcs(u, t)
    }

    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64) {
        self.apply_velocity_bcs(v, t)
    }
}

impl<D> ConstantUniformVelocity<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    VectorN<f64, D>: Serialize,
{
    pub fn new(nodes: &[usize], velocity: VectorN<f64, D>) -> Self {
        Self {
            nodes: nodes.to_vec(),
            velocity,
        }
    }

    fn solution_dim(&self) -> usize {
        D::dim()
    }

    fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    fn apply_displacement_bcs(&self, mut u: DVectorSliceMut<f64>, t: f64) {
        assert!(u.nrows() == D::dim() * self.nodes.len());

        for i in 0..self.nodes.len() {
            let mut ui = u.fixed_rows_mut::<D>(D::dim() * i);
            ui.axpy(t, &self.velocity, 1.0);
        }
    }

    fn apply_velocity_bcs(&self, mut v: DVectorSliceMut<f64>, _t: f64) {
        assert!(v.nrows() == D::dim() * self.nodes.len());

        for i in 0..self.nodes.len() {
            let mut vi = v.fixed_rows_mut::<D>(D::dim() * i);
            vi.axpy(1.0, &self.velocity, 1.0);
        }
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for ConstantUniformVelocity<U3> {
    fn solution_dim(&self) -> usize {
        self.solution_dim()
    }

    fn nodes(&self) -> &[usize] {
        self.nodes()
    }

    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64) {
        self.apply_displacement_bcs(u, t)
    }

    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64) {
        self.apply_velocity_bcs(v, t)
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for ConstantUniformVelocity<U2> {
    fn solution_dim(&self) -> usize {
        self.solution_dim()
    }

    fn nodes(&self) -> &[usize] {
        self.nodes()
    }

    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64) {
        self.apply_displacement_bcs(u, t)
    }

    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64) {
        self.apply_velocity_bcs(v, t)
    }
}

impl<D> ConstantUniformAngularVelocity<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    VectorN<f64, D>: Serialize,
    Point<f64, D>: Serialize,
{
    pub fn new(nodes: &[usize], omega: VectorN<f64, D>, center: Point<f64, D>, x0: Vec<Point<f64, D>>) -> Self {
        Self {
            nodes: nodes.to_vec(),
            omega,
            center,
            x0,
        }
    }
}

#[typetag::serde]
impl DirichletBoundaryConditions for ConstantUniformAngularVelocity<U3> {
    fn solution_dim(&self) -> usize {
        3
    }

    fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    fn apply_displacement_bcs(&self, mut u: DVectorSliceMut<f64>, t: f64) {
        let rot_angle = (self.omega.norm() * t) % (2.0 * std::f64::consts::PI);
        let rot = Rotation3::from_axis_angle(&Unit::new_normalize(self.omega.clone()), rot_angle);

        for i in 0..self.nodes.len() {
            let mut ui = u.fixed_rows_mut::<U3>(3 * i);
            let r = &self.x0[i] - &self.center;
            let r_rot = &rot * r;
            let dr = r_rot - r;
            ui.copy_from(&dr);
        }
    }

    fn apply_velocity_bcs(&self, mut v: DVectorSliceMut<f64>, t: f64) {
        // TODO: This doesn't work properly with the apply method, because it only applies itself
        let mut u = DVector::zeros(3 * self.nodes.len());
        self.apply_displacement_bcs(DVectorSliceMut::from(&mut u), t);

        for i in 0..self.nodes.len() {
            let ui = u.fixed_rows::<U3>(3 * i);
            let mut vi = v.fixed_rows_mut::<U3>(3 * i);
            let v_tang = self.omega.cross(&ui);
            vi.copy_from(&v_tang);
        }
    }
}

impl OptionalDirichletBoundaryConditions for Option<&dyn DirichletBoundaryConditions> {
    fn solution_dim(&self) -> usize {
        self.map(|bc| bc.solution_dim()).unwrap_or(0)
    }

    fn nodes(&self) -> &[usize] {
        self.map(|bc| bc.nodes()).unwrap_or(&[])
    }

    fn nrows(&self) -> usize {
        self.map(|bc| bc.nrows()).unwrap_or(0)
    }

    fn apply_displacement_bcs(&self, u: DVectorSliceMut<f64>, t: f64) {
        if let Some(bc) = self {
            bc.apply_displacement_bcs(u, t);
        }
    }

    fn apply_velocity_bcs(&self, v: DVectorSliceMut<f64>, t: f64) {
        if let Some(bc) = self {
            bc.apply_velocity_bcs(v, t);
        }
    }
}
