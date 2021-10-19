use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

use coarse_prof::profile;
use fenris::nalgebra::{Vector2, Vector3};
use hamilton::storages::{ImmutableSingletonStorage, SingletonStorage, VecStorage};
use hamilton::{Component, FilterSystem, RunOnceSystem, StorageContainer, System};
use serde::{Deserialize, Serialize};

mod mesh;
pub use mesh::*;

pub fn new_delayed_once_system<F>(closure: F, wakeup_time: f64) -> impl System
where
    F: FnOnce(&StorageContainer) -> Result<(), Box<dyn Error>>,
{
    new_delayed_system(RunOnceSystem::new(closure), wakeup_time)
}

pub fn new_delayed_system<S>(system: S, wakeup_time: f64) -> impl System
where
    S: System,
{
    let delay_predicate = {
        let wakeup_time = wakeup_time;
        let mut has_woken_up = false;
        move |data: &StorageContainer| -> Result<bool, Box<dyn Error>> {
            if !has_woken_up {
                has_woken_up = get_simulation_time(data)? >= wakeup_time;
                Ok(has_woken_up)
            } else {
                Ok(false)
            }
        }
    };

    FilterSystem {
        predicate: delay_predicate,
        system,
    }
}

/// A system for timing the execution of a stored system
#[derive(Debug)]
pub struct TimingSystem<S>
where
    S: System,
{
    pub name: &'static str,
    pub system: S,
}

impl<S> TimingSystem<S>
where
    S: System,
{
    pub fn new(name: &'static str, system: S) -> Self {
        TimingSystem { name, system }
    }
}

impl<S> Display for TimingSystem<S>
where
    S: System,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TimingSystem for {}", self.name)
    }
}

impl<S> System for TimingSystem<S>
where
    S: System,
{
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        {
            profile!(self.name);
            self.system.run(data)
        }
    }
}

/// A component that represents a name for the given entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Name(pub String);

impl Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Component for Name {
    type Storage = VecStorage<Name>;
}

impl<T> From<T> for Name
where
    T: Into<String>,
{
    fn from(string_like: T) -> Self {
        Self(string_like.into())
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct TimeStep(pub f64);

impl Component for TimeStep {
    // Note: for now we assume that time step is immutable. Different systems may of course
    // still perform arbitrary time adaptivity within a single step if desirable.
    type Storage = ImmutableSingletonStorage<Self>;
}

impl From<TimeStep> for f64 {
    fn from(timestep: TimeStep) -> Self {
        timestep.0
    }
}

/// Models elapsed simulation time.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SimulationTime(pub f64);

impl Component for SimulationTime {
    type Storage = ImmutableSingletonStorage<Self>;
}

impl SimulationTime {
    pub fn elapsed(&self) -> f64 {
        self.0
    }

    pub fn add(&self, delta: f64) -> Self {
        Self(self.0 + delta)
    }
}

impl From<f64> for SimulationTime {
    fn from(time: f64) -> Self {
        Self(time)
    }
}

/// Models gravity.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Gravity {
    Scalar(f64),
    Vec2(Vector2<f64>),
    Vec3(Vector3<f64>),
}

impl Component for Gravity {
    type Storage = SingletonStorage<Self>;
}

impl From<f64> for Gravity {
    fn from(gravity: f64) -> Self {
        Gravity::Scalar(gravity)
    }
}

impl From<Vector2<f64>> for Gravity {
    fn from(gravity: Vector2<f64>) -> Self {
        Gravity::Vec2(gravity)
    }
}

impl From<Vector3<f64>> for Gravity {
    fn from(gravity: Vector3<f64>) -> Self {
        Gravity::Vec3(gravity)
    }
}

impl TryInto<Vector2<f64>> for Gravity {
    type Error = ();

    fn try_into(self) -> Result<Vector2<f64>, ()> {
        match self {
            Gravity::Vec2(g) => Ok(g),
            _ => Err(()),
        }
    }
}

impl TryInto<Vector3<f64>> for Gravity {
    type Error = ();

    fn try_into(self) -> Result<Vector3<f64>, ()> {
        match self {
            Gravity::Vec3(g) => Ok(g),
            _ => Err(()),
        }
    }
}

/// The index of the current step.
///
/// The index of the initial state of a simulation will typically have index 0.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct StepIndex(pub isize);

impl Component for StepIndex {
    type Storage = ImmutableSingletonStorage<StepIndex>;
}

/// The index in the sequence of exported data.
///
/// The index of the initial state of a simulation will typically have index 0.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ExportSequenceIndex {
    pub index: isize,
    pub prev_export_time: Option<f64>,
    pub export_interval: f64,
}

impl Component for ExportSequenceIndex {
    type Storage = ImmutableSingletonStorage<ExportSequenceIndex>;
}

pub fn get_simulation_time(state: &StorageContainer) -> Result<f64, Box<dyn Error>> {
    state
        .try_get_component_storage::<SimulationTime>()
        .map(|refcell| refcell.borrow())
        .map(|storage| storage.get_component().elapsed())
        .ok_or_else(|| Box::<dyn Error>::from("Failed to get SimulationTime component."))
}

pub fn get_time_step(state: &StorageContainer) -> Result<f64, Box<dyn Error>> {
    state
        .try_get_component_storage::<TimeStep>()
        .map(|refcell| refcell.borrow())
        .map(|storage| f64::from(*storage.get_component()))
        .ok_or_else(|| Box::<dyn Error>::from("Failed to get TimeStep component."))
}

pub fn get_gravity(state: &StorageContainer) -> Result<Gravity, Box<dyn Error>> {
    state
        .try_get_component_storage::<Gravity>()
        .map(|refcell| refcell.borrow())
        .map(|storage| *storage.get_component())
        .ok_or_else(|| Box::<dyn Error>::from("Failed to get Gravity component."))
}

pub fn set_gravity(state: &mut StorageContainer, new_gravity: impl Into<Gravity>) {
    state.replace_storage(<Gravity as Component>::Storage::new(new_gravity.into()));
}

pub fn get_export_sequence_index(state: &StorageContainer) -> Result<isize, Box<dyn Error>> {
    state
        .try_get_component_storage::<ExportSequenceIndex>()
        .map(|refcell| refcell.borrow())
        .map(|storage| storage.get_component().index)
        .ok_or_else(|| Box::<dyn Error>::from("Failed to get ExportSequenceIndex component."))
}

pub fn get_step_index(state: &StorageContainer) -> Result<isize, Box<dyn Error>> {
    state
        .try_get_component_storage::<StepIndex>()
        .map(|refcell| refcell.borrow())
        .map(|storage| storage.get_component().0)
        .ok_or_else(|| Box::<dyn Error>::from("Failed to get StepIndex component."))
}
