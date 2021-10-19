use std::borrow::Borrow;
use std::borrow::Cow;
use std::error::Error;
use std::fs;
use std::ops::Deref;
use std::path::Path;

use fenris::nalgebra::{Translation3, Vector3};
pub use serde_json;

#[derive(Clone, Debug)]
pub struct JsonWrapper<'a> {
    pub value: Cow<'a, serde_json::Value>,
}

/// Convenience function to parse a file to a serde_json::Value
pub fn parse_json_from_path<P: AsRef<Path>>(path: P) -> Result<JsonWrapper<'static>, Box<dyn Error>> {
    let json_string = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "Unable to open JSON file '{}' for reading ({:?})",
            path.as_ref().to_string_lossy(),
            e
        )
    })?;

    let json_value = serde_json::de::from_str(&json_string).map_err(|e| {
        format!(
            "Error during parsing of JSON file '{}': {}",
            path.as_ref().to_string_lossy(),
            e
        )
    })?;

    Ok(JsonWrapper { value: json_value })
}

/// Convenience trait to deserialize JSON values into types
pub trait TryFromJson: Sized {
    fn try_from_json<J: Borrow<serde_json::Value>>(json: J) -> Result<Self, Box<dyn Error>>;
}

impl<'a> JsonWrapper<'a> {
    pub fn new(value: serde_json::Value) -> Self {
        Self {
            value: Cow::Owned(value),
        }
    }

    pub fn new_borrowed(value: &'a serde_json::Value) -> Self {
        Self {
            value: Cow::Borrowed(value),
        }
    }

    pub fn to_string_pretty(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string_pretty(self.as_value())?)
    }

    /// Return a reference to the contained serde_json::Value
    pub fn as_value(&self) -> &serde_json::Value {
        &self.value
    }

    pub fn members(&self) -> Result<impl Iterator<Item = JsonWrapper>, Box<dyn Error>> {
        Ok(self
            .as_value()
            .as_array()
            .ok_or_else(|| Box::<dyn Error>::from("Expected array"))?
            .iter()
            .map(|v| JsonWrapper::new_borrowed(v)))
    }

    pub fn get_json_value_ref<S: AsRef<str>>(&self, index: S) -> Result<&serde_json::Value, Box<dyn Error>> {
        if let Some(obj) = self.as_object() {
            return obj
                .get(index.as_ref())
                .ok_or_else(|| Box::from(format!("Entry \"{}\" not found in JSON file", index.as_ref())));
        } else {
            return Err(Box::from(format!(
                "Cannot access entry \"{}\" as parent JSON value is not an object",
                index.as_ref()
            )));
        }
    }

    pub fn get<S: AsRef<str>>(&self, index: S) -> Result<JsonWrapper, Box<dyn Error>> {
        self.get_json_value_ref(index)
            .map(|v| JsonWrapper::new_borrowed(v))
    }

    pub fn get_bool<S: AsRef<str>>(&self, index: S) -> Result<bool, Box<dyn Error>> {
        self.get_json_value_ref(index.as_ref())
            .map(|v| v.as_bool())?
            .ok_or_else(|| {
                Box::from(format!(
                    "The entry \"{}\" does not contain a bool value",
                    index.as_ref()
                ))
            })
    }

    pub fn get_f64<S: AsRef<str>>(&self, index: S) -> Result<f64, Box<dyn Error>> {
        self.get_json_value_ref(index.as_ref())
            .map(|v| v.as_f64())?
            .ok_or_else(|| Box::from(format!("The entry \"{}\" does not contain a f64 value", index.as_ref())))
    }

    pub fn get_i64<S: AsRef<str>>(&self, index: S) -> Result<i64, Box<dyn Error>> {
        self.get_json_value_ref(index.as_ref())
            .map(|v| v.as_i64())?
            .ok_or_else(|| Box::from(format!("The entry \"{}\" does not contain a i64 value", index.as_ref())))
    }
}

impl<'a> Borrow<serde_json::Value> for JsonWrapper<'a> {
    fn borrow(&self) -> &serde_json::Value {
        &self.as_value()
    }
}

impl<'a> Deref for JsonWrapper<'a> {
    type Target = serde_json::Value;

    fn deref(&self) -> &Self::Target {
        &self.as_value()
    }
}

impl TryFromJson for Vector3<f64> {
    fn try_from_json<J: Borrow<serde_json::Value>>(json: J) -> Result<Self, Box<dyn Error>> {
        Ok(Vector3::from(<[f64; 3] as TryFromJson>::try_from_json(json)?))
    }
}

impl TryFromJson for Translation3<f64> {
    fn try_from_json<J: Borrow<serde_json::Value>>(json: J) -> Result<Self, Box<dyn Error>> {
        Ok(Translation3::from(Vector3::try_from_json(json)?))
    }
}

// TODO: Can this be implemented using Serde? Otherwise use macro?
impl TryFromJson for [f64; 3] {
    fn try_from_json<J: Borrow<serde_json::Value>>(json: J) -> Result<Self, Box<dyn Error>> {
        if let serde_json::Value::Array(arr) = json.borrow() {
            Ok([
                arr[0]
                    .as_f64()
                    .ok_or_else(|| "Expected float component of a three component array")?,
                arr[1]
                    .as_f64()
                    .ok_or_else(|| "Expected float component of a three component array")?,
                arr[2]
                    .as_f64()
                    .ok_or_else(|| "Expected float component of a three component array")?,
            ])
        } else {
            Err(Box::from(
                "Expected a three component float array in JSON, but no array object found",
            ))
        }
    }
}
