use crate::geometry::{AxisAlignedBoundingBox2d, BoundedGeometry, Distance, LineSegment2d, Orientation};
use itertools::{izip, Itertools};
use nalgebra::{Point2, RealField, Scalar, Vector2, U2};
use serde::{Deserialize, Serialize};
use std::iter::once;

use numeric_literals::replace_float_literals;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneralPolygon<T>
where
    T: Scalar,
{
    vertices: Vec<Point2<T>>,
    // TODO: Also use acceleration structure for fast queries?
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ClosestEdge<T>
where
    T: Scalar,
{
    pub signed_distance: T,
    pub edge_parameter: T,
    pub edge_point: Point2<T>,
    pub edge_index: usize,
}

pub trait Polygon<T>
where
    T: RealField,
{
    fn vertices(&self) -> &[Point2<T>];

    fn num_edges(&self) -> usize;

    fn get_edge(&self, index: usize) -> Option<LineSegment2d<T>>;

    fn num_vertices(&self) -> usize {
        self.vertices().len()
    }

    /// Returns the given pseudonormal (angle-weighted normal) given an edge index and a parameter
    /// representing a point on edge.
    ///
    /// If t == 0, then the average normal of this edge and its predecessor neighbor is returned.
    /// If t == 1, then the average normal of this edge and its successor neighbor is returned.
    /// Otherwise the normal of the edge is returned.
    fn pseudonormal_on_edge(&self, edge_index: usize, t: T) -> Option<Vector2<T>>;

    fn for_each_edge(&self, mut func: impl FnMut(usize, LineSegment2d<T>)) {
        for edge_idx in 0..self.num_edges() {
            let segment = self
                .get_edge(edge_idx)
                .expect("Edge index must exist, given that we're in the loop body.");
            func(edge_idx, segment);
        }
    }

    fn closest_edge(&self, x: &Point2<T>) -> Option<ClosestEdge<T>> {
        let mut closest_edge_index = None;
        let mut smallest_squared_dist = T::max_value();

        self.for_each_edge(|edge_idx, edge| {
            let closest_point_on_edge = edge.closest_point(x);
            let dist2 = (x - closest_point_on_edge).magnitude_squared();
            if dist2 < smallest_squared_dist {
                closest_edge_index = Some(edge_idx);
                smallest_squared_dist = dist2;
            }
        });

        let closest_edge_index = closest_edge_index?;
        // We unwrap all the results below, because since we have a concrete index,
        // all results *must exist*, otherwise it's an error
        let closest_edge = self.get_edge(closest_edge_index).unwrap();
        let t = closest_edge.closest_point_parametric(x);
        let pseudonormal = self.pseudonormal_on_edge(closest_edge_index, t).unwrap();
        let closest_point_on_edge = closest_edge.point_from_parameter(t);
        let d = x - &closest_point_on_edge;
        let distance = d.magnitude();
        let sign = d.dot(&pseudonormal).signum();

        Some(ClosestEdge {
            signed_distance: sign * distance,
            edge_parameter: t,
            edge_point: closest_point_on_edge,
            edge_index: closest_edge_index,
        })
    }

    fn intersects_segment(&self, segment: &LineSegment2d<T>) -> bool {
        // A segment either
        //  - Intersects an edge in the polygon
        //  - Is completely contained in the polygon
        //  - Does not intersect the polygon at all
        // To determine if it is completely contained in the polygon, we keep track of
        // the closest edge to each endpoint of the segment. Technically, only one would be
        // sufficient, but due to floating-point errors there are some cases where a segment may
        // be classified as not intersecting an edge, yet only one of its endpoints will have
        // a negative signed distance to the polygon. Thus, for robustness, we compute the signed
        // distance of both endpoints.
        if self.num_edges() == 0 {
            return false;
        }

        let mut closest_edges = [0, 0];
        let mut smallest_squared_dists = [T::max_value(), T::max_value()];
        let endpoints = [*segment.from(), *segment.to()];

        let mut intersects = false;

        self.for_each_edge(|edge_idx, edge| {
            if edge.intersect_segment_parametric(segment).is_some() {
                intersects = true;
            } else {
                for (endpoint, closest_edge, smallest_dist2) in
                    izip!(&endpoints, &mut closest_edges, &mut smallest_squared_dists)
                {
                    let closest_point_on_edge = edge.closest_point(endpoint);
                    let dist2 = (endpoint - closest_point_on_edge).magnitude_squared();
                    if dist2 < *smallest_dist2 {
                        *closest_edge = edge_idx;
                        *smallest_dist2 = dist2;
                    }
                }
            }
        });

        for (endpoint, closest_edge_idx) in izip!(&endpoints, &closest_edges) {
            // We can unwrap here, because we know that the Polygon has at least one edge
            let closest_edge = self.get_edge(*closest_edge_idx).unwrap();
            let t = closest_edge.closest_point_parametric(endpoint);
            let pseudonormal = self.pseudonormal_on_edge(*closest_edge_idx, t).unwrap();
            let closest_point_on_edge = closest_edge.point_from_parameter(t);
            let sign = (endpoint - closest_point_on_edge).dot(&pseudonormal);

            if sign <= T::zero() {
                return true;
            }
        }

        false
    }

    /// Computes the signed area of the (simple) polygon.
    ///
    /// The signed area of a simple polygon is positive if the polygon has a counter-clockwise
    /// orientation, or negative if it is oriented clockwise.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn signed_area(&self) -> T {
        // The formula for the signed area of a simple polygon can easily be obtained from
        // Green's formula by rewriting the surface integral that defines its area
        // as an integral over the curve defining the polygon's boundary,
        // which furthermore can be decomposed into a sum of integrals over each edge.
        // See e.g.
        //  https://math.blogoverflow.com/2014/06/04/greens-theorem-and-area-of-polygons/
        // for details.
        let two_times_signed_area = (0..self.num_edges())
            .map(|edge_idx| self.get_edge(edge_idx).unwrap())
            .map(|segment| {
                let a = segment.from();
                let b = segment.to();
                (b.y - a.y) * (b.x + a.x)
            })
            .fold(T::zero(), |sum, contrib| sum + contrib);
        two_times_signed_area / 2.0
    }

    /// Computes the area of the (simple) polygon.
    fn area(&self) -> T {
        self.signed_area().abs()
    }

    fn orientation(&self) -> Orientation {
        let signed_area = self.signed_area();
        if signed_area > T::zero() {
            Orientation::Counterclockwise
        } else {
            Orientation::Clockwise
        }
    }
}

impl<T> GeneralPolygon<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: Vec<Point2<T>>) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>] {
        &self.vertices
    }

    pub fn transform_vertices<F>(&mut self, mut transform: F)
    where
        F: FnMut(&mut [Point2<T>]),
    {
        transform(&mut self.vertices)

        // TODO: Update acceleration structure etc., if we decide to internally use one later on
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_edges(&self) -> usize {
        self.vertices.len()
    }

    /// An iterator over edges as line segments
    pub fn edge_iter<'a>(&'a self) -> impl 'a + Iterator<Item = LineSegment2d<T>> {
        self.vertices
            .iter()
            .chain(once(self.vertices.first().unwrap()))
            .tuple_windows()
            .map(|(a, b)| LineSegment2d::new(a.clone(), b.clone()))
    }
}

impl<T> GeneralPolygon<T>
where
    T: RealField,
{
    /// Corrects the orientation of the polygon.
    ///
    /// The first vertex is guaranteed to be the same before and after the orientation
    /// change.
    pub fn orient(&mut self, desired_orientation: Orientation) {
        if desired_orientation != self.orientation() {
            self.vertices.reverse();
            self.vertices.rotate_right(1);
        }
    }
}

impl<T> Polygon<T> for GeneralPolygon<T>
where
    T: RealField,
{
    fn vertices(&self) -> &[Point2<T>] {
        &self.vertices
    }

    fn num_edges(&self) -> usize {
        self.vertices.len()
    }

    fn get_edge(&self, index: usize) -> Option<LineSegment2d<T>> {
        let a = self.vertices.get(index)?;
        let b = self.vertices.get((index + 1) % self.num_vertices())?;
        Some(LineSegment2d::new(*a, *b))
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn pseudonormal_on_edge(&self, edge_index: usize, t: T) -> Option<Vector2<T>> {
        let edge = self.get_edge(edge_index)?;
        let edge_normal = edge.normal_dir().normalize();

        // TODO: Handle potentially degenerate line segments (i.e. they degenerate to a single
        // point, and so normalization of the normal is arbitrary, if it at all works)

        let pseudonormal = if t == T::zero() {
            // Have to take care not to underflow usize, so we cannot subtract directly
            let previous_idx = ((edge_index + self.num_edges()) - 1) % self.num_edges();
            let previous_edge = self.get_edge(previous_idx)?;
            let previous_edge_normal = previous_edge.normal_dir().normalize();
            ((previous_edge_normal + edge_normal) / 2.0).normalize()
        } else if t == T::one() {
            let next_idx = (edge_index + 1) % self.num_edges();
            let next_edge = self.get_edge(next_idx)?;
            let next_edge_normal = next_edge.normal_dir().normalize();
            ((next_edge_normal + edge_normal) / 2.0).normalize()
        } else {
            edge_normal
        };

        Some(pseudonormal)
    }
}

impl<T> BoundedGeometry<T> for GeneralPolygon<T>
where
    T: RealField,
{
    type Dimension = U2;

    fn bounding_box(&self) -> AxisAlignedBoundingBox2d<T> {
        AxisAlignedBoundingBox2d::from_points(self.vertices()).expect("Vertex collection must be non-empty")
    }
}

impl<T> Distance<T, Point2<T>> for GeneralPolygon<T>
where
    T: RealField,
{
    fn distance(&self, point: &Point2<T>) -> T {
        let closest_edge = self
            .closest_edge(point)
            .expect("We don't support empty polygons at the moment (do we want to?)");
        T::max(closest_edge.signed_distance, T::zero())
    }
}
