use crate::{EntityDeserialize, EntitySerializationMap};
use serde::de::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ENTITY: AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Entity(u64);

impl Entity {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Entity(NEXT_ENTITY.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SerializableEntity(u64);

impl From<Entity> for SerializableEntity {
    fn from(id: Entity) -> Self {
        Self(id.0)
    }
}

impl<'de> EntityDeserialize<'de> for Entity {
    fn entity_deserialize<D>(deserializer: D, id_map: &mut EntitySerializationMap) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let deserializable = SerializableEntity::deserialize(deserializer)?;
        let entity = id_map.deserialize_entity(deserializable);
        Ok(entity)
    }
}

impl<'a, 'de> serde::de::DeserializeSeed<'de> for &'a mut EntitySerializationMap {
    type Value = Entity;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let deserializable = SerializableEntity::deserialize(deserializer)?;
        let entity = self.deserialize_entity(deserializable);
        Ok(entity)
    }
}
