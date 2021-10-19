use crate::{EntityDeserialize, EntitySerializationMap, StorageFactory};
use erased_serde::Serialize;
use std::any::Any;
use std::cell::RefCell;
use std::error::Error;
use std::marker::PhantomData;

#[derive(Debug, Default)]
pub struct GenericFactory<Storage> {
    marker: PhantomData<Storage>,
}

impl<Storage> GenericFactory<Storage> {
    pub fn new() -> Self {
        Self { marker: PhantomData }
    }
}

// Factory contains no data whatsoever and is therefore entirely safe to pass around across threads
unsafe impl<Storage> Sync for GenericFactory<Storage> {}
unsafe impl<Storage> Send for GenericFactory<Storage> {}

impl<Storage> StorageFactory for GenericFactory<Storage>
where
    for<'de> Storage: 'static + serde::Serialize + EntityDeserialize<'de>,
{
    fn storage_tag(&self) -> String {
        std::any::type_name::<Storage>().to_string()
    }

    fn serializable_storage(&self, storage: &dyn Any) -> Result<&dyn Serialize, Box<dyn Error>> {
        // TODO: Is this actually valid?
        storage
            .downcast_ref::<RefCell<Storage>>()
            .map(|storage| storage as *const RefCell<Storage>)
            // I'm not sure if Any actually provides the necessary guarantees for this to be valid.
            // In particular, we are extending the lifetime of the returned reference to
            // beyond that of the `downcast_ref` method. However, `Any` can only give a reference
            // to a concrete object, and the lifetime of this object must be at least as long
            // as the shared reference we have to the trait object, and so by converting our
            // reference into a pointer, this points to the contained item, which we know
            // has the required lifetime.
            .map(|ptr| unsafe { &*ptr } as &dyn Serialize)
            .ok_or_else(|| Box::from("provided storage is not known to factory"))
    }

    fn deserialize_storage(
        &self,
        deserializer: &mut dyn erased_serde::Deserializer,
        id_map: &mut EntitySerializationMap,
    ) -> Result<Box<dyn Any>, Box<dyn Error>> {
        let storage = Storage::entity_deserialize(deserializer, id_map)?;
        Ok(Box::new(RefCell::new(storage)))
    }
}
