use fenris::geometry::polymesh::PolyMesh3d;
use fenris::nalgebra::Point3;
use fenris::nested_vec::NestedVec;
use obj::{IndexTuple, Obj, SimplePolygon};
use std::error::Error;
use std::path::Path;

pub fn load_single_surface_polymesh3d_obj(path: impl AsRef<Path>) -> Result<PolyMesh3d<f64>, Box<dyn Error>> {
    load_single_surface_polymesh3d_obj_(path.as_ref())
}

fn load_single_surface_polymesh3d_obj_(path: &Path) -> Result<PolyMesh3d<f64>, Box<dyn Error>> {
    let obj_file = Obj::load(path)?;

    let vertices: Vec<_> = obj_file
        .data
        .position
        .iter()
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .map(Point3::from)
        .collect();

    if obj_file.data.objects.len() != 1 {
        return Err(Box::from("Obj file must contain exactly one object"));
    }

    let object = obj_file.data.objects.first().unwrap();
    let mut faces = NestedVec::new();
    for group in &object.groups {
        for SimplePolygon(ref index_tuples) in &group.polys {
            let mut appender = faces.begin_array();
            for IndexTuple(vertex_idx, _, _) in index_tuples {
                appender.push_single(*vertex_idx);
            }
        }
    }

    Ok(PolyMesh3d::from_poly_data(vertices, faces, NestedVec::new()))
}
