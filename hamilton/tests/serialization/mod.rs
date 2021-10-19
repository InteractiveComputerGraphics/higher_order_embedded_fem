use hamilton::storages::VecStorage;
use hamilton::{register_component, Component, Entity, StorageContainer};

use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Foo(i32);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Bar(i32);

impl Component for Foo {
    type Storage = VecStorage<Foo>;
}

impl Component for Bar {
    type Storage = VecStorage<Bar>;
}

#[test]
fn json_roundtrip() {
    register_component::<Foo>().unwrap();
    register_component::<Bar>().unwrap();

    let container = StorageContainer::default();

    let id1 = Entity::new();
    let id2 = Entity::new();
    let id3 = Entity::new();

    {
        let mut foo_storage = container.get_component_storage::<Foo>().borrow_mut();
        foo_storage.insert(id2, Foo(1));
        foo_storage.insert(id1, Foo(2));

        let mut bar_storage = container.get_component_storage::<Bar>().borrow_mut();
        bar_storage.insert(id2, Bar(3));
        bar_storage.insert(id3, Bar(4));
        bar_storage.insert(id1, Bar(5));
    }

    let json = serde_json::to_string_pretty(&container).unwrap();

    // Drop container so that we make sure we don't accidentally reference it later
    drop(container);

    let deserialized_container: StorageContainer = serde_json::from_str(&json).unwrap();

    let foo_storage = deserialized_container
        .get_component_storage::<Foo>()
        .borrow();
    let bar_storage = deserialized_container
        .get_component_storage::<Bar>()
        .borrow();

    let foos = foo_storage.components();
    let bars = bar_storage.components();

    assert_eq!(foos, &[Foo(1), Foo(2)]);
    assert_eq!(bars, &[Bar(3), Bar(4), Bar(5)]);

    // We can not directly compare the entities with expected values, since we cannot predict
    // what they should be. However, entities only describe relations, and we can therefore
    // instead check that the components that shared the same entities still do after
    // serialization and deserialization.
    let foo_ids = foo_storage.entities();
    let bar_ids = bar_storage.entities();

    assert_eq!(foo_ids[0], bar_ids[0]);
    assert_eq!(foo_ids[1], bar_ids[2]);

    // Assert that the remaining entity is not equal to any of the others
    assert_ne!(bar_ids[1], bar_ids[0]);
    assert_ne!(bar_ids[1], bar_ids[2]);
}
