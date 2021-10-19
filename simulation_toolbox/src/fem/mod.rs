#[macro_use]
mod fe_model;
pub use fe_model::*;

mod deformer;
pub use deformer::*;

mod system_assembly;

mod integrator;
pub use integrator::*;

pub mod newton_cg;

pub mod bcs;
pub use bcs::{DirichletBoundaryConditionComponent, DirichletBoundaryConditions, OptionalDirichletBoundaryConditions};

pub mod schwarz_precond;

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum IntegrationMethod {
    SymplecticEuler,
    BackwardEuler,
}

impl Default for IntegrationMethod {
    fn default() -> Self {
        IntegrationMethod::BackwardEuler
    }
}
