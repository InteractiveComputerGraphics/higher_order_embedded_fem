use crate::{BijectiveStorageMut, Component, Entity};
use std::any::Any;
use std::cell::RefCell;
use std::ops::Deref;
use std::pin::Pin;

// Make container_serialize a submodule of this module, so that it can still
// access private members of `StorageContainer`, without exposing this to the rest of the
// crate (using e.g. `pub(crate)`).
mod container_serialize;

pub use container_serialize::{register_factory, register_storage, RegistrationStatus};

#[derive(Debug, Default)]
pub struct StorageContainer {
    // Note: The "dyn Any" actually contains instances of "RefCell<Storage>"
    storages: RefCell<Vec<(String, Pin<Box<dyn Any>>)>>,
}

impl StorageContainer {
    pub fn new() -> Self {
        Self {
            storages: RefCell::new(Vec::new()),
        }
    }

    pub fn get_component_storage<C>(&self) -> &RefCell<C::Storage>
    where
        C: Component,
        C::Storage: 'static + Default,
    {
        self.get_storage::<C::Storage>()
    }

    pub fn try_get_component_storage<C>(&self) -> Option<&RefCell<C::Storage>>
    where
        C: Component,
        C::Storage: 'static,
    {
        self.try_get_storage::<C::Storage>()
    }

    pub fn get_storage<Storage>(&self) -> &RefCell<Storage>
    where
        Storage: 'static + Default,
    {
        if let Some(storage) = self.try_get_storage() {
            return storage;
        }

        // We didn't find the storage, so make a new one based on default value
        let new_storage = Box::pin(RefCell::new(Storage::default()));
        let new_storage_ptr = {
            // Make sure that we take a pointer to the right type by first explicitly
            // generating a reference to it
            let new_storage_ref: &RefCell<Storage> = new_storage.deref();
            new_storage_ref as *const RefCell<Storage>
        };
        self.storages
            .borrow_mut()
            .push((String::from(std::any::type_name::<Storage>()), new_storage));

        // See the above comment for why this is valid, as the same argument applies.
        unsafe { &*new_storage_ptr }
    }

    pub fn try_get_storage<Storage>(&self) -> Option<&RefCell<Storage>>
    where
        Storage: 'static,
    {
        for (_, untyped_storage) in self.storages.borrow().iter() {
            if let Some(typed) = untyped_storage.downcast_ref::<RefCell<Storage>>() {
                let typed_ptr = typed as *const RefCell<_>;

                // This is valid because the RefCell<Storage> is contained inside Pin<Box>,
                // so as long as the pinned box is never deallocated, the pointer will remain
                // valid. We never remove any storages from the vector, so this could only
                // happen upon deallocation of the StorageContainer. However, the
                // returned reference is scoped to the lifetime of &self, so
                // it cannot outlive the container itself.
                return Some(unsafe { &*typed_ptr });
            }
        }

        // Did not find storage
        None
    }

    pub fn replace_storage<Storage>(&mut self, storage: Storage) -> bool
    where
        Storage: 'static,
    {
        // Note: Because we take &mut self here, we know that there cannot be any outstanding
        // references given out by `get_storage`, and so it's safe to replace
        // (and therefore destroy) the given storage
        let mut storages = self.storages.borrow_mut();

        let tag = std::any::type_name::<Storage>();
        let pinned_storage = Box::pin(RefCell::new(storage));

        for (existing_tag, existing_storage) in storages.iter_mut() {
            if *existing_tag == tag {
                *existing_storage = pinned_storage;
                return true;
            }
        }

        // No storage with the same tag found, so simply add it to existing storages
        storages.push((String::from(tag), pinned_storage));
        false
    }

    pub fn insert_component<C>(&self, id: Entity, component: C)
    where
        C: Component,
        C::Storage: 'static + Default + BijectiveStorageMut<Component = C>,
    {
        let mut storage = self.get_component_storage::<C>().borrow_mut();
        storage.insert_component(id, component);
    }
}
