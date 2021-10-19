use std::any::Any;
use std::collections::HashMap;
use std::error::Error;

mod entity;
pub use entity::*;

mod container;
pub use container::*;

pub mod storages;

mod generic_factory;
pub use generic_factory::*;

mod systems;
pub use systems::*;

pub struct EntitySerializationMap {
    map: HashMap<SerializableEntity, Entity>,
}

impl EntitySerializationMap {
    fn new() -> Self {
        Self { map: HashMap::new() }
    }

    pub fn deserialize_entity(&mut self, id: SerializableEntity) -> Entity {
        *self.map.entry(id).or_insert_with(Entity::new)
    }
}

pub trait StorageFactory: Send + Sync {
    fn storage_tag(&self) -> String;

    fn serializable_storage(&self, storage: &dyn Any) -> Result<&dyn erased_serde::Serialize, Box<dyn Error>>;

    fn deserialize_storage(
        &self,
        deserializer: &mut dyn erased_serde::Deserializer,
        id_map: &mut EntitySerializationMap,
    ) -> Result<Box<dyn Any>, Box<dyn Error>>;
}

pub trait Storage {
    fn new_factory() -> Box<dyn StorageFactory>;
}

/// Storage that represents a one-to-one (bijective) correspondence between entities and components.
pub trait BijectiveStorage {
    // TODO: Move associated type to `Storage`?
    type Component;

    fn get_component_for_entity(&self, id: Entity) -> Option<&Self::Component>;
}

pub trait BijectiveStorageMut: BijectiveStorage {
    /// Inserts a component associated with the entity, overwriting any existing component
    /// that may already be associated with the given entity.
    fn insert_component(&mut self, id: Entity, component: Self::Component);

    fn get_component_for_entity_mut(&mut self, id: Entity) -> Option<&mut Self::Component>;
}

/// An extension of serde's `Deserialize` that allows deserialization of types containing
/// instances `Entity` (which are not deserializable)
pub trait EntityDeserialize<'de>: Sized {
    fn entity_deserialize<D>(deserializer: D, id_map: &mut EntitySerializationMap) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>;
}

impl<'de, T> EntityDeserialize<'de> for T
where
    T: serde::Deserialize<'de>,
{
    fn entity_deserialize<D>(deserializer: D, _: &mut EntitySerializationMap) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer)
    }
}

pub trait Component {
    type Storage: Storage;
}

pub fn register_component<C>() -> Result<RegistrationStatus, Box<dyn Error>>
where
    C: Component,
{
    register_storage::<C::Storage>()
}
