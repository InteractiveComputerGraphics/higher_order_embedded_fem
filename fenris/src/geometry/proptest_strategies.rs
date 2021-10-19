use crate::geometry::procedural::create_rectangular_uniform_quad_mesh_2d;
use crate::geometry::{LineSegment2d, Orientation, Quad2d, Triangle, Triangle2d};
use crate::mesh::QuadMesh2d;

use crate::util::proptest::point2_f64_strategy;
use nalgebra::{Point2, Vector2};
use proptest::prelude::*;
use std::cmp::max;

pub fn triangle2d_strategy_f64() -> impl Strategy<Value = Triangle2d<f64>> {
    [point2_f64_strategy(), point2_f64_strategy(), point2_f64_strategy()].prop_map(|points| Triangle(points))
}

pub fn clockwise_triangle2d_strategy_f64() -> impl Strategy<Value = Triangle2d<f64>> {
    triangle2d_strategy_f64().prop_map(|mut triangle| {
        if triangle.orientation() != Orientation::Clockwise {
            triangle.swap_vertices(0, 2);
        }
        triangle
    })
}

pub fn nondegenerate_line_segment2d_strategy_f64() -> impl Strategy<Value = LineSegment2d<f64>> {
    // Make sure to construct the second point from non-zero components
    let gen = prop_oneof![0.5..3.5, -0.5..3.5, 1e-6..10.0, -10.0..-1e-6];
    (point2_f64_strategy(), gen.clone(), gen).prop_map(|(a, x, y)| {
        let d = Vector2::new(x, y);
        let b = a + d;
        LineSegment2d::new(a, b)
    })
}

/// A strategy for triangles that are oriented clockwise and not degenerate
/// (i.e. collapsed to a line, area
pub fn nondegenerate_triangle2d_strategy_f64() -> impl Strategy<Value = Triangle2d<f64>> {
    let segment = nondegenerate_line_segment2d_strategy_f64();
    let t1_gen = prop_oneof![-3.0..3.0, -10.0..10.0];
    let t2_gen = prop_oneof![0.5..3.0, 1e-6..10.0];
    (segment, t1_gen, t2_gen).prop_map(|(segment, t1, t2)| {
        let a = segment.from();
        let b = segment.to();
        let ab = b - a;
        let n = Vector2::new(-ab.y, ab.x);
        let c = Point2::from(a + t1 * ab + t2 * n);
        Triangle([*a, *b, c])
    })
}

fn extrude_triangle_to_convex_quad(triangle: &Triangle2d<f64>, t1: f64, t3: f64) -> Quad2d<f64> {
    // In order to generate a convex quad, we first generate one triangle,
    // then we "extrude" a vertex from one of the sides of the triangle, in such a way
    // that the vertex is contained in the convex cone defined by the two other sides,
    // constrained to lie on the side itself or further away.
    // The result is a convex quad.
    let t2 = 1.0 - t1;
    assert!(t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t3 >= 0.0);
    let a = &triangle.0[0];
    let b = &triangle.0[1];
    let c = &triangle.0[2];
    let d1 = b - a;
    let d2 = c - a;
    // Define a vector d3 pointing from a to a point on the opposite edge
    let d3_hat = t1 * d1 + t2 * d2;
    // Choose a parameter t3 >= 0. Then (1 + t3) * d3_hat is a vector pointing from a to the new
    // point
    let d3 = (1.0 + t3) * d3_hat;

    Quad2d([*a, *b, a + d3, *c])
}

pub fn convex_quad2d_strategy_f64() -> impl Strategy<Value = Quad2d<f64>> {
    let t1_gen = 0.0..=1.0;
    let t3_gen = 0.0..10.0;
    (t1_gen, t3_gen, clockwise_triangle2d_strategy_f64())
        .prop_map(|(t1, t3, triangle)| extrude_triangle_to_convex_quad(&triangle, t1, t3))
}

pub fn nondegenerate_convex_quad2d_strategy_f64() -> impl Strategy<Value = Quad2d<f64>> {
    let t1_gen = prop_oneof![0.25..=0.75, 1e-6..=(1.0 - 1e-6)];
    let t3_gen = prop_oneof![0.5..3.0, 1e-6..10.0];
    (t1_gen, t3_gen, nondegenerate_triangle2d_strategy_f64())
        .prop_map(|(t1, t3, triangle)| extrude_triangle_to_convex_quad(&triangle, t1, t3))
}

pub fn parallelogram_strategy_f64() -> impl Strategy<Value = Quad2d<f64>> {
    nondegenerate_triangle2d_strategy_f64().prop_map(|triangle| {
        let a = &triangle.0[0];
        let b = &triangle.0[1];
        let d = &triangle.0[2];
        let ab = b - a;
        let ad = d - a;
        let c = a + ab + ad;
        Quad2d([*a, *b, c, *d])
    })
}

// Returns a strategy in which each value is a triplet (cells_per_unit, units_x, units_y)
// such that cells_per_unit^2 * units_x * units_y <= max_cells
fn rectangular_uniform_mesh_cell_distribution_strategy(
    max_cells: usize,
) -> impl Strategy<Value = (usize, usize, usize)> {
    let max_cells_per_unit = f64::floor(f64::sqrt(max_cells as f64)) as usize;
    (1..=max(1, max_cells_per_unit))
        .prop_flat_map(move |cells_per_unit| (Just(cells_per_unit), 0..=max_cells / (cells_per_unit * cells_per_unit)))
        .prop_flat_map(move |(cells_per_unit, units_x)| {
            let units_y_strategy = 0..=max_cells / (cells_per_unit * cells_per_unit * max(1, units_x));
            (Just(cells_per_unit), Just(units_x), units_y_strategy)
        })
}

pub fn rectangular_uniform_mesh_strategy(unit_length: f64, max_cells: usize) -> impl Strategy<Value = QuadMesh2d<f64>> {
    rectangular_uniform_mesh_cell_distribution_strategy(max_cells).prop_map(
        move |(cells_per_unit, units_x, units_y)| {
            create_rectangular_uniform_quad_mesh_2d(
                unit_length,
                units_x,
                units_y,
                cells_per_unit,
                &Vector2::new(0.0, 0.0),
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        convex_quad2d_strategy_f64, nondegenerate_convex_quad2d_strategy_f64, nondegenerate_triangle2d_strategy_f64,
        rectangular_uniform_mesh_cell_distribution_strategy,
    };
    use crate::geometry::Orientation;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn rectangular_uniform_mesh_cell_distribution_strategy_respects_max_cells(
            (max_cells, cells_per_unit, units_x, units_y)
             in (0..20usize).prop_flat_map(|max_cells| {
                rectangular_uniform_mesh_cell_distribution_strategy(max_cells)
                    .prop_map(move |(cells_per_unit, units_x, units_y)| {
                    (max_cells, cells_per_unit, units_x, units_y)
                })
             })
        ) {
            // Test that the distribution strategy for rectangular meshes
            // respects the maximum number of cells given
            prop_assert!(cells_per_unit * cells_per_unit * units_x * units_y <= max_cells);
        }

        #[test]
        fn convex_quads_are_convex(quad in convex_quad2d_strategy_f64()) {
            prop_assert!(quad.concave_corner().is_none());
        }

        #[test]
        fn nondegenerate_triangles_have_positive_area(
            triangle in nondegenerate_triangle2d_strategy_f64()
        ){
            prop_assert!(triangle.area() > 0.0);
            prop_assert!(triangle.orientation() == Orientation::Clockwise);
        }

        #[test]
        fn nondegenerate_quads_have_positive_area(
            quad in nondegenerate_convex_quad2d_strategy_f64()
        ){
            prop_assert!(quad.area() > 0.0);
        }
    }
}
