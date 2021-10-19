use crate::StorageContainer;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};

pub trait System: Debug + Display {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>>;
}

/// A system that runs only once and executes its contained closure
pub struct RunOnceSystem<F>
where
    F: FnOnce(&StorageContainer) -> Result<(), Box<dyn Error>>,
{
    pub closure: Option<F>,
    has_run: bool,
}

/// System that uses a closure to determine if a system should be run
pub struct FilterSystem<P, S>
where
    P: FnMut(&StorageContainer) -> Result<bool, Box<dyn Error>>,
    S: System,
{
    pub predicate: P,
    pub system: S,
}

/// Wrapper to store a vector of systems that are run in sequence
pub struct SystemCollection(pub Vec<Box<dyn System>>);

impl<F> Debug for RunOnceSystem<F>
where
    F: FnOnce(&StorageContainer) -> Result<(), Box<dyn Error>>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RunOnceSystem(has_run: {})", self.has_run)
    }
}

impl<F> RunOnceSystem<F>
where
    F: FnOnce(&StorageContainer) -> Result<(), Box<dyn Error>>,
{
    pub fn new(closure: F) -> Self {
        RunOnceSystem {
            closure: Some(closure),
            has_run: false,
        }
    }
}

impl<F> Display for RunOnceSystem<F>
where
    F: FnOnce(&StorageContainer) -> Result<(), Box<dyn Error>>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RunOnceSystem(has_run: {})", self.has_run)
    }
}

impl<F> System for RunOnceSystem<F>
where
    F: FnOnce(&StorageContainer) -> Result<(), Box<dyn Error>>,
{
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        if !self.has_run {
            let ret = (self
                .closure
                .take()
                .ok_or_else(|| Box::<dyn Error>::from("Closure gone"))?)(data)?;
            self.has_run = true;
            Ok(ret)
        } else {
            Ok(())
        }
    }
}

impl<P, S> Debug for FilterSystem<P, S>
where
    P: FnMut(&StorageContainer) -> Result<bool, Box<dyn Error>>,
    S: System,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Filter({:?})", self.system)
    }
}

impl<P, S> Display for FilterSystem<P, S>
where
    P: FnMut(&StorageContainer) -> Result<bool, Box<dyn Error>>,
    S: System,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Filter({})", self.system)
    }
}

impl<P, S> System for FilterSystem<P, S>
where
    P: FnMut(&StorageContainer) -> Result<bool, Box<dyn Error>>,
    S: System,
{
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        if (self.predicate)(data)? {
            self.system.run(data)
        } else {
            Ok(())
        }
    }
}

impl Debug for SystemCollection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SystemCollection({:?})", self.0)
    }
}

impl Display for SystemCollection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut systems = String::new();
        self.0.iter().for_each(|s| {
            systems.push_str(&format!("{}, ", s));
        });
        if systems.len() > 2 {
            write!(f, "SystemCollection({})", &systems[..systems.len() - 2])
        } else {
            write!(f, "SystemCollection()")
        }
    }
}

impl System for SystemCollection {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        for s in self.0.iter_mut() {
            s.run(data)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct Systems {
    systems: Vec<Box<dyn System>>,
}

impl Systems {
    pub fn add_system(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
    }

    pub fn run_all(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        for system in &mut self.systems {
            system.run(data)?;
        }
        Ok(())
    }
}
