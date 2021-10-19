use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::rc::Rc;

use serde::ser::{Serialize, SerializeMap, Serializer};
use thiserror::Error;

// TODO: Rename the crate to something not on crates.io yet.

thread_local! {
    #[doc(hidden)]
    pub static STASH: RefCell<Stash> = RefCell::new(Stash::new())
}

/// Tries to open a new scope in the global stash to add a level of nesting. Panics if the scope name is already taken by a key-value pair.
#[macro_export]
macro_rules! stash_scope {
    ($name:expr) => {
        let _guard = $crate::STASH.with(|s| s.borrow_mut().enter($name)).unwrap();
    };
}

/// Tries to open a new scope in the global stash to add a level of nesting. Calls the `?`-operator and returns a `StashError` if not successful.
#[macro_export]
macro_rules! try_stash_scope {
    ($name:expr) => {
        let _guard = $crate::STASH.with(|s| s.borrow_mut().enter($name))?;
    };
}

#[doc(hidden)]
pub struct Stash {
    root: Rc<RefCell<Scope>>,
    current: Rc<RefCell<Scope>>,
}

impl Serialize for Stash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.root.borrow().serialize(serializer)
    }
}

/// Error type used by the crate.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StashError {
    /// Indicates that a scope with the given name could not be opened because the scope's name is already taken by a value.
    #[error("in the current stash scope exists a stashed value with the same key as the requested scope name (`{0}`)")]
    ScopeNameTaken(String),
    /// Indicates that a variable could not be added because the key value is already taken in the current scope.
    #[error("in the current stash scope exists a stashed value with the same key or a nested scope with the same name as the requested key (`{0}`)")]
    KeyTaken(String),
    /// Indicates that a push operation was not successful because the target variable is not an array.
    #[error("the given key `{0}` does not contain a value that is an array")]
    NotAnArray(String),
}

struct Scope {
    name: String,
    parent: Option<Rc<RefCell<Scope>>>,
    children: Vec<Rc<RefCell<Scope>>>,
    entries: BTreeMap<String, serde_json::Value>,
}

impl Serialize for Scope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.children.len() + self.entries.len()))?;
        for (k, v) in self.entries.iter() {
            map.serialize_entry(k, v)?;
        }
        for child in self.children.iter().map(|c| c.borrow()) {
            map.serialize_entry(&child.name, &*child)?;
        }
        map.end()
    }
}

impl Scope {
    fn new<S: AsRef<str>>(name: S, parent: Option<Rc<RefCell<Scope>>>) -> Self {
        Scope {
            name: name.as_ref().to_string(),
            parent: parent,
            children: Vec::new(),
            entries: BTreeMap::new(),
        }
    }

    #[allow(unused)]
    fn enter(&mut self) -> Guard {
        Guard {}
    }

    fn leave(&mut self) {}

    fn name_taken<S: AsRef<str>>(&self, key: S) -> bool {
        let key = key.as_ref();
        self.children.iter().any(|scope| scope.borrow().name == key) || self.entries.contains_key(key)
    }

    fn key_exists<S: AsRef<str>>(&self, key: S) -> bool {
        let key = key.as_ref();
        self.entries.contains_key(key)
    }

    fn insert_or_modify<S: AsRef<str>, T: Into<serde_json::Value>, F>(&mut self, key: S, value: T, modifier: F)
    where
        F: FnMut(&mut serde_json::Value),
    {
        let key = key.as_ref();
        self.entries
            .entry(key.to_string())
            .and_modify(modifier)
            .or_insert(value.into());
    }

    fn insert_no_overwrite<S: AsRef<str>, T: Into<serde_json::Value>>(
        &mut self,
        key: S,
        value: T,
    ) -> Result<(), StashError> {
        let key = key.as_ref();
        if !self.name_taken(key) {
            self.entries.insert(key.to_string(), value.into());
            Ok(())
        } else {
            Err(StashError::KeyTaken(key.to_string()))
        }
    }

    fn push<S: AsRef<str>, T: Into<serde_json::Value>>(&mut self, key: S, value: T) -> Result<(), StashError> {
        let key = key.as_ref();
        if !self.key_exists(key) {
            self.insert_no_overwrite(key, Vec::<serde_json::Value>::new())?;
        }

        let array = self.entries.get_mut(key).unwrap();
        if let Some(array) = array.as_array_mut() {
            array.push(value.into());
            Ok(())
        } else {
            Err(StashError::NotAnArray(key.to_string()))
        }
    }

    fn clear_values(&mut self) {
        self.entries.clear();
        for child in self.children.iter() {
            child.borrow_mut().clear_values();
        }
    }
}

#[doc(hidden)]
pub struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        STASH.with(|s| s.borrow_mut().leave());
    }
}

impl Stash {
    fn new() -> Self {
        let root = Rc::new(RefCell::new(Scope::new("root", None)));
        Stash {
            root: root.clone(),
            current: root,
        }
    }

    #[allow(unused)]
    #[doc(hidden)]
    pub fn enter<S: AsRef<str>>(&mut self, name: S) -> Result<Guard, StashError> {
        let name = name.as_ref();
        let requested_scope = {
            // Check if there already exists a child scope with the given name
            let existing_scope = self
                .current
                .borrow()
                .children
                .iter()
                .find(|child| child.borrow().name == name)
                .cloned();

            existing_scope
                // If there was no existing scope with the given name found, it has to be created
                .or_else(|| {
                    // First check if there's already a value with the given name as key
                    if self.current.borrow().name_taken(name) {
                        return None;
                    }

                    // If not, create a new child scope
                    let new_scope = Scope::new(name, Some(self.current.clone()));
                    let child = Rc::new(RefCell::new(new_scope));

                    self.current.borrow_mut().children.push(child.clone());

                    Some(child)
                })
                // If there is still no new scope, the name is blocked
                .ok_or(StashError::ScopeNameTaken(name.to_string()))?
        };

        let guard = requested_scope.borrow_mut().enter();
        self.current = requested_scope;

        Ok(guard)
    }

    fn leave(&mut self) {
        let new_current = {
            self.current.borrow_mut().leave();
            self.current.borrow().parent.clone().unwrap_or_else(|| {
                std::panic!("Tried to leave root node");
            })
        };
        self.current = new_current;
    }

    fn insert_or_modify<S: AsRef<str>, T: Into<serde_json::Value>, F>(&mut self, key: S, value: T, modifier: F)
    where
        F: FnMut(&mut serde_json::Value),
    {
        self.current
            .borrow_mut()
            .insert_or_modify(key, value, modifier)
    }

    fn insert_no_overwrite<S: AsRef<str>, T: Into<serde_json::Value>>(
        &mut self,
        key: S,
        value: T,
    ) -> Result<(), StashError> {
        self.current.borrow_mut().insert_no_overwrite(key, value)
    }

    fn push<S: AsRef<str>, T: Into<serde_json::Value>>(&mut self, key: S, value: T) -> Result<(), StashError> {
        self.current.borrow_mut().push(key, value)
    }

    fn clear_values(&mut self) {
        self.root.borrow_mut().clear_values();
    }
}

/// Removes all values from the global stash while leaving the scope structure unaffected.
pub fn clear_values() {
    crate::STASH.with(|s| s.borrow_mut().clear_values())
}

/// Tries to add the given key-value pair to the global stash, nested into the currently open scope. Panics on failure.
pub fn insert_value<S: AsRef<str>, T: Into<serde_json::Value>>(key: S, value: T) {
    crate::STASH
        .with(|s| s.borrow_mut().insert_no_overwrite(key, value))
        .unwrap();
}

/// TODO: Docs
pub fn insert_value_or_modify<S: AsRef<str>, T: Into<serde_json::Value>, F>(key: S, value: T, modifier: F)
where
    F: FnMut(&mut serde_json::Value),
{
    crate::STASH.with(|s| s.borrow_mut().insert_or_modify(key, value, modifier));
}

/// Tries to add the given key-value pair to the global stash, nested into the currently open scope.
pub fn try_insert_value<S: AsRef<str>, T: Into<serde_json::Value>>(key: S, value: T) -> Result<(), StashError> {
    crate::STASH.with(|s| s.borrow_mut().insert_no_overwrite(key, value))
}

/// Tries to push a value to an array with the given name in the currently open scope. Panics on failure.
pub fn push_value<S: AsRef<str>, T: Into<serde_json::Value>>(array_name: S, value: T) {
    crate::STASH
        .with(|s| s.borrow_mut().push(array_name, value))
        .unwrap()
}

/// Tries to push a value to an array with the given name in the currently open scope.
pub fn try_push_value<S: AsRef<str>, T: Into<serde_json::Value>>(array_name: S, value: T) -> Result<(), StashError> {
    crate::STASH.with(|s| s.borrow_mut().push(array_name, value))
}

/// Serialize the global stash as a `String` of JSON.
pub fn to_string() -> serde_json::Result<String> {
    crate::STASH.with(|s| serde_json::to_string(&*s.borrow()))
}

/// Serialize the global stash as a pretty-printed `String` of JSON.
pub fn to_string_pretty() -> serde_json::Result<String> {
    crate::STASH.with(|s| serde_json::to_string_pretty(&*s.borrow()))
}

/// Convert the global stash into a `serde_json::Value`.
pub fn to_value() -> serde_json::Result<serde_json::Value> {
    crate::STASH.with(|s| serde_json::to_value(&*s.borrow()))
}

/// Serialize the global stash as JSON into the IO stream.
pub fn to_writer<W, T: ?Sized>(writer: W) -> serde_json::Result<()>
where
    W: Write,
    T: Serialize,
{
    crate::STASH.with(|s| serde_json::to_writer(writer, &*s.borrow()))
}

/// Serialize the global stash as pretty-printed JSON into the IO stream.
pub fn to_writer_pretty<W, T: ?Sized>(writer: W) -> serde_json::Result<()>
where
    W: Write,
    T: Serialize,
{
    crate::STASH.with(|s| serde_json::to_writer_pretty(writer, &*s.borrow()))
}

#[cfg(test)]
mod tests {
    use crate as stash;

    #[test]
    fn test_basic() {
        let scope_name = "Test scope";
        stash_scope!(scope_name);

        let n_sub_scopes = 5;
        for i in 0..n_sub_scopes {
            stash_scope!(format!("{}", i));
            assert!(stash::try_insert_value("index", i).is_ok());
            assert!(stash::try_insert_value("square", format!("{} * {} = {}", i, i, i * i)).is_ok());
        }

        for i in 0..n_sub_scopes {
            assert!(stash::try_insert_value(format!("{}", i), i).is_err());
        }

        super::STASH.with(|s| {
            let s = s.borrow();
            {
                let current = s.current.borrow();

                assert_eq!(&current.name, scope_name);
                assert_eq!(current.children.len(), n_sub_scopes);
                for i in 0..n_sub_scopes {
                    let child = current.children[i].borrow();
                    assert_eq!(child.name, format!("{}", i));
                    assert_eq!(child.entries.len(), 2);
                    assert_eq!(child.entries["index"], i);
                    assert_eq!(child.entries["square"], format!("{} * {} = {}", i, i, i * i));
                }
            }
        });
    }

    #[test]
    fn test_try() {
        let scope_name = "Test scope";
        stash_scope!(scope_name);

        {
            stash::insert_value("Hello", 0);
            let produce_err = || -> Result<(), stash::StashError> {
                try_stash_scope!("Hello");
                Ok(())
            };
            let val = produce_err();
            assert_eq!(val, Err(stash::StashError::ScopeNameTaken("Hello".to_string())));

            let produce_err = || -> Result<(), stash::StashError> {
                stash::try_insert_value("Hello", 1)?;
                Ok(())
            };
            let val = produce_err();
            assert_eq!(val, Err(stash::StashError::KeyTaken("Hello".to_string())));
        }
    }

    #[test]
    fn test_serialize() {
        stash_scope!("Test scope");
        for i in 0..5 {
            stash_scope!(format!("{}", i));
            assert!(stash::try_insert_value("index", i).is_ok());
            assert!(stash::try_insert_value("square", format!("{} * {} = {}", i, i, i * i)).is_ok());
        }

        let json = stash::to_string().unwrap();
        assert_eq!(
            &json,
            r#"{"Test scope":{"0":{"index":0,"square":"0 * 0 = 0"},"1":{"index":1,"square":"1 * 1 = 1"},"2":{"index":2,"square":"2 * 2 = 4"},"3":{"index":3,"square":"3 * 3 = 9"},"4":{"index":4,"square":"4 * 4 = 16"}}}"#
        );

        stash::clear_values();

        let json = stash::to_string().unwrap();
        assert_eq!(&json, r#"{"Test scope":{"0":{},"1":{},"2":{},"3":{},"4":{}}}"#);
    }

    #[test]
    fn test_push() {
        let scope_name = "Test scope";
        stash_scope!(scope_name);

        let array_len = 20;
        {
            stash_scope!("An array");
            for i in 0..array_len {
                stash::push_value("arr", i * i);
            }

            stash::STASH.with(|s| {
                let s = s.borrow();
                let current = s.current.borrow();

                assert_eq!(current.entries.len(), 1);
                assert!(current.entries["arr"].is_array());
                assert_eq!(current.entries["arr"].as_array().unwrap().len(), array_len);
            });

            assert!(stash::try_insert_value("arr2", 0).is_ok());
            assert_eq!(
                stash::try_push_value("arr2", 1),
                Err(stash::StashError::NotAnArray("arr2".to_string()))
            );
        }
    }

    #[test]
    fn test_array_serialize() {
        let scope_name = "Test scope";
        stash_scope!(scope_name);

        {
            stash_scope!("An array");
            for i in 0..5 {
                stash::push_value("arr", i * i);
            }

            let json = stash::to_string().unwrap();
            assert_eq!(&json, r#"{"Test scope":{"An array":{"arr":[0,1,4,9,16]}}}"#);
        }
    }
}
