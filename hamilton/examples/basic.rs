use hamilton::storages::VecStorage;
use hamilton::{register_component, Component, Entity, StorageContainer};

use std::error::Error;

use serde::{Deserialize, Serialize};

use serde_json;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct TestComponent(pub usize);

impl Component for TestComponent {
    type Storage = VecStorage<Self>;
}

fn main() -> Result<(), Box<dyn Error>> {
    register_component::<TestComponent>()?;

    let container = StorageContainer::new();

    {
        let storage = container.get_storage::<VecStorage<TestComponent>>();
        storage.borrow_mut().insert(Entity::new(), TestComponent(0));
        storage.borrow_mut().insert(Entity::new(), TestComponent(1));

        dbg!(storage.borrow());
    }

    let json = serde_json::to_string_pretty(&container)?;

    println!("{}", json);

    let deserialized_container: StorageContainer = serde_json::from_str(&json)?;

    {
        let storage = deserialized_container.get_storage::<VecStorage<TestComponent>>();
        dbg!(storage.borrow());
    }

    Ok(())
}
