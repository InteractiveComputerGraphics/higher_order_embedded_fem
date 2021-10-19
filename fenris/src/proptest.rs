use proptest::prelude::*;

use crate::element::Tet4Element;
use crate::geometry::Orientation::Counterclockwise;
use crate::geometry::{Orientation, Triangle, Triangle2d, Triangle3d};
use nalgebra::{Point2, Point3};

pub fn point2() -> impl Strategy<Value = Point2<f64>> {
    // Pick a reasonably small range to pick coordinates from,
    // otherwise we can easily get floating point numbers that are
    // so ridiculously large as to break anything we might want to do with them
    let range = -10.0..10.0;
    [range.clone(), range.clone()].prop_map(|[x, y]| Point2::new(x, y))
}

pub fn point3() -> impl Strategy<Value = Point3<f64>> {
    // Pick a reasonably small range to pick coordinates from,
    // otherwise we can easily get floating point numbers that are
    // so ridiculously large as to break anything we might want to do with them
    let range = -10.0..10.0;
    [range.clone(), range.clone(), range.clone()].prop_map(|[x, y, z]| Point3::new(x, y, z))
}

#[derive(Debug, Clone)]
pub struct Triangle3dParams {
    orientation: Orientation,
}

impl Triangle3dParams {
    pub fn with_orientation(self, orientation: Orientation) -> Self {
        Self { orientation, ..self }
    }
}

impl Default for Triangle3dParams {
    fn default() -> Self {
        Self {
            orientation: Counterclockwise,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Triangle2dParams {
    orientation: Orientation,
}

impl Triangle2dParams {
    pub fn with_orientation(self, orientation: Orientation) -> Self {
        Self { orientation, ..self }
    }
}

impl Default for Triangle2dParams {
    fn default() -> Self {
        Self {
            orientation: Counterclockwise,
        }
    }
}

impl Arbitrary for Triangle3d<f64> {
    // TODO: Parameter for extents (i.e. bounding box or so)
    type Parameters = Triangle3dParams; // TODO: Avoid boxing for performance...?
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let points = [point3(), point3(), point3()];
        points
            .prop_map(|points| Triangle(points))
            .prop_map(move |mut triangle| {
                if triangle.orientation() != args.orientation {
                    triangle.swap_vertices(0, 1);
                }
                triangle
            })
            .boxed()
    }
}

impl Arbitrary for Triangle2d<f64> {
    // TODO: Parameter for extents (i.e. bounding box or so)
    type Parameters = Triangle2dParams; // TODO: Avoid boxing for performance...?
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let points = [point2(), point2(), point2()];
        points
            .prop_map(|points| Triangle(points))
            .prop_map(move |mut triangle| {
                if triangle.orientation() != args.orientation {
                    triangle.swap_vertices(0, 1);
                }
                triangle
            })
            .boxed()
    }
}

impl Arbitrary for Tet4Element<f64> {
    // TODO: Reasonable parameters?
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any_with::<Triangle3d<f64>>(Triangle3dParams::default().with_orientation(Counterclockwise))
            .prop_flat_map(|triangle| {
                // To create an arbitrary tetrahedron element, we take a counter-clockwise oriented
                // triangle, and pick a point somewhere on the "positive" side of the
                // triangle plane. We do this by associating a parameter with each
                // tangent vector defined by the sides of the triangle,
                // plus a non-negative parameter that scales along the normal direction
                let range = -10.0..10.0;
                let tangent_params = [range.clone(), range.clone(), range.clone()];
                let normal_param = 0.0..=10.0;
                (Just(triangle), tangent_params, normal_param)
            })
            .prop_map(|(triangle, tangent_params, normal_param)| {
                let mut tangent_pos = triangle.centroid();

                for (side, param) in triangle.sides().iter().zip(&tangent_params) {
                    tangent_pos.coords += *param * side;
                }
                let coord = tangent_pos + normal_param * triangle.normal_dir().normalize();
                Tet4Element::from_vertices([triangle.0[0], triangle.0[1], triangle.0[2], coord])
            })
            .boxed()
    }
}
