use crate::{
    BijectiveStorage, BijectiveStorageMut, Entity, EntityDeserialize, EntitySerializationMap, GenericFactory,
    SerializableEntity, Storage, StorageFactory,
};
use std::collections::HashMap;

#[derive(Clone, Debug, serde::Serialize)]
pub struct VecStorage<Component> {
    components: Vec<Component>,
    entities: Vec<Entity>,
    lookup_table: HashMap<Entity, usize>,
}

// Helper struct to ease implementation of deserialization
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct VecStorageCompanion<Component> {
    components: Vec<Component>,
    entities: Vec<SerializableEntity>,
    lookup_table: HashMap<SerializableEntity, usize>,
}

impl<Component> VecStorageCompanion<Component> {
    pub fn to_storage(self, id_map: &mut EntitySerializationMap) -> VecStorage<Component> {
        VecStorage {
            components: self.components,
            entities: self
                .entities
                .into_iter()
                .map(|id| id_map.deserialize_entity(id))
                .collect(),
            lookup_table: self
                .lookup_table
                .into_iter()
                .map(|(id, idx)| (id_map.deserialize_entity(id), idx))
                .collect(),
        }
    }
}

/// Stores component in a vector, with a one-to-one relationship between entities and components.
impl<Component> VecStorage<Component> {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            entities: Vec::new(),
            lookup_table: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.components.len(), self.entities.len());
        self.components.len()
    }

    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.components.is_empty(), self.entities.is_empty());
        self.components.is_empty()
    }

    pub fn get_index(&self, id: Entity) -> Option<usize> {
        self.lookup_table.get(&id).map(usize::to_owned)
    }

    pub fn get_component(&self, id: Entity) -> Option<&Component> {
        self.components.get(self.get_index(id)?)
    }

    pub fn get_component_mut(&mut self, id: Entity) -> Option<&mut Component> {
        let index = self.get_index(id)?;
        self.components.get_mut(index)
    }

    pub fn insert(&mut self, id: Entity, component: Component) -> usize {
        let len = self.len();
        let index = *self.lookup_table.entry(id).or_insert_with(|| len);

        if index < self.components.len() {
            *self.components.get_mut(index).unwrap() = component;
        } else {
            self.components.push(component);
            self.entities.push(id);
            debug_assert_eq!(index + 1, self.components.len());
        }

        index
    }

    /// Removes the component associated with the given entity, if it exists.
    ///
    /// Returns `true` if a component was removed, otherwise `false`.
    pub fn remove_entity(&mut self, id: &Entity) -> bool {
        if let Some(index) = self.lookup_table.remove(id) {
            self.entities.remove(index);
            self.components.remove(index);
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.entities.clear();
        self.components.clear();
        self.lookup_table.clear();
    }

    pub fn components(&self) -> &[Component] {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut [Component] {
        &mut self.components
    }

    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn entity_component_iter(&self) -> impl Iterator<Item = (&Entity, &Component)> {
        self.entities.iter().zip(self.components.iter())
    }

    pub fn entity_component_iter_mut(&mut self) -> impl Iterator<Item = (&Entity, &mut Component)> {
        self.entities.iter().zip(self.components.iter_mut())
    }
}

impl<Component> Default for VecStorage<Component> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Component> Storage for VecStorage<Component>
where
    for<'de> Component: 'static + serde::Serialize + serde::Deserialize<'de>,
{
    fn new_factory() -> Box<dyn StorageFactory> {
        Box::new(GenericFactory::<Self>::new())
    }
}

impl<T> Storage for Vec<T>
where
    for<'de> T: Clone + 'static + serde::Serialize + serde::Deserialize<'de>,
{
    fn new_factory() -> Box<dyn StorageFactory> {
        Box::new(GenericFactory::<Self>::new())
    }
}

impl<C> BijectiveStorage for VecStorage<C> {
    type Component = C;

    fn get_component_for_entity(&self, id: Entity) -> Option<&Self::Component> {
        self.components.get(self.get_index(id)?)
    }
}

impl<C> BijectiveStorageMut for VecStorage<C> {
    fn insert_component(&mut self, id: Entity, component: Self::Component) {
        self.insert(id, component);
    }

    fn get_component_for_entity_mut(&mut self, id: Entity) -> Option<&mut Self::Component> {
        let index = self.get_index(id)?;
        self.components.get_mut(index)
    }
}

impl<'de, Component> EntityDeserialize<'de> for VecStorage<Component>
where
    Component: serde::Deserialize<'de>,
{
    fn entity_deserialize<D>(deserializer: D, id_map: &mut EntitySerializationMap) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::Deserialize;
        let companion = VecStorageCompanion::<Component>::deserialize(deserializer)?;
        Ok(companion.to_storage(id_map))
    }
}

#[derive(Debug, Copy, Clone, serde::Serialize)]
pub struct SingletonStorage<Component> {
    component: Component,
}

impl<Component> SingletonStorage<Component> {
    pub fn new(component: Component) -> Self {
        Self { component }
    }

    pub fn get_component(&self) -> &Component {
        &self.component
    }

    pub fn get_component_mut(&mut self) -> &mut Component {
        &mut self.component
    }
}

impl<Component> Storage for SingletonStorage<Component>
where
    for<'de> Component: 'static + serde::Serialize + EntityDeserialize<'de>,
{
    fn new_factory() -> Box<dyn StorageFactory> {
        Box::new(GenericFactory::<SingletonStorage<Component>>::new())
    }
}

impl<'de, Component> EntityDeserialize<'de> for SingletonStorage<Component>
where
    Component: EntityDeserialize<'de>,
{
    fn entity_deserialize<D>(deserializer: D, id_map: &mut EntitySerializationMap) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            component: Component::entity_deserialize(deserializer, id_map)?,
        })
    }
}

#[derive(Debug, Copy, Clone, serde::Serialize)]
pub struct ImmutableSingletonStorage<Component> {
    component: Component,
}

impl<Component> ImmutableSingletonStorage<Component> {
    pub fn new(component: Component) -> Self {
        Self { component }
    }

    pub fn get_component(&self) -> &Component {
        &self.component
    }
}

impl<Component> Storage for ImmutableSingletonStorage<Component>
where
    for<'de> Component: 'static + serde::Serialize + EntityDeserialize<'de>,
{
    fn new_factory() -> Box<dyn StorageFactory> {
        Box::new(GenericFactory::<ImmutableSingletonStorage<Component>>::new())
    }
}

impl<'de, Component> EntityDeserialize<'de> for ImmutableSingletonStorage<Component>
where
    Component: EntityDeserialize<'de>,
{
    fn entity_deserialize<D>(deserializer: D, id_map: &mut EntitySerializationMap) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            component: Component::entity_deserialize(deserializer, id_map)?,
        })
    }
}
