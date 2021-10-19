//! Functionality for error estimation.

use crate::allocators::VolumeFiniteElementAllocator;
use crate::element::FiniteElement;
use crate::quadrature::{Quadrature, Quadrature2d};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimMin, DimName, DimNameMul, MatrixMN, Point, RealField, VectorN, U1, U2};

/// Estimate the squared L^2 error of `u_h - u` on the given element with the given basis
/// weights and quadrature points.
///
/// `u(x, i)` represents the value of `u` at physical coordinate `x`. `i` is the index of the
/// quadrature point.
///
/// More precisely, estimate the integral of `dot(u_h - u, u_h - u)`, where `u_h = u_i N_i`,
/// with `u_i` the `i`-th column in `u` denoting the `m`-dimensional weight associated with node `i`,
/// and `N_i` is the basis function associated with node `i`.
#[allow(non_snake_case)]
pub fn estimate_element_L2_error_squared<T, SolutionDim, GeometryDim, Element>(
    element: &Element,
    u: impl Fn(&Point<T, GeometryDim>, usize) -> VectorN<T, SolutionDim>,
    u_weights: &MatrixMN<T, SolutionDim, Element::NodalDim>,
    quadrature: impl Quadrature<T, GeometryDim>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    SolutionDim: DimNameMul<Element::NodalDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, Element::GeometryDim, Element::NodalDim>
        + Allocator<T, SolutionDim, Element::NodalDim>
        + Allocator<T, SolutionDim, U1>,
{
    let weights = quadrature.weights();
    let points = quadrature.points();

    use itertools::izip;

    let mut result = T::zero();
    for (i, (w, xi)) in izip!(weights, points).enumerate() {
        let x = element.map_reference_coords(xi);
        let j = element.reference_jacobian(xi);
        let g = element.evaluate_basis(xi);
        let u_h = u_weights * g.transpose();
        let u_at_x = u(&Point::from(x), i);
        let error = u_h - u_at_x;
        let error2 = error.dot(&error);
        result += error2 * *w * j.determinant().abs();
    }
    result
}

#[allow(non_snake_case)]
pub fn estimate_element_L2_error<T, SolutionDim, GeometryDim, Element>(
    element: &Element,
    u: impl Fn(&Point<T, GeometryDim>, usize) -> VectorN<T, SolutionDim>,
    u_weights: &MatrixMN<T, SolutionDim, Element::NodalDim>,
    quadrature: impl Quadrature<T, GeometryDim>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    SolutionDim: DimNameMul<Element::NodalDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, Element::GeometryDim, Element::NodalDim>
        + Allocator<T, SolutionDim, Element::NodalDim>
        + Allocator<T, SolutionDim, U1>,
{
    estimate_element_L2_error_squared(element, u, u_weights, quadrature).sqrt()
}

/// Estimate the squared L^2 norm on the given element with the given basis weights and quadrature
/// points.
///
/// More precisely, compute the integral of `dot(u_h, u_h)`, where `u_h = u_i N_i`, with `u_i`,
/// the `i`-th column in `u`, denoting the `m`-dimensional weight associated with node `i`,
/// and `N_i` is the basis function associated with node `i`.
#[allow(non_snake_case)]
pub fn estimate_element_L2_norm_squared<T, SolutionDim, Element>(
    element: &Element,
    u_weights: &MatrixMN<T, SolutionDim, Element::NodalDim>,
    quadrature: impl Quadrature2d<T>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = U2, ReferenceDim = U2>,
    SolutionDim: DimNameMul<Element::NodalDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, U2, Element::NodalDim>
        + Allocator<T, SolutionDim, Element::NodalDim>
        + Allocator<T, SolutionDim, U1>,
{
    estimate_element_L2_error_squared(
        element,
        |_, _| VectorN::<T, SolutionDim>::repeat(T::zero()),
        u_weights,
        quadrature,
    )
}

#[allow(non_snake_case)]
pub fn estimate_element_L2_norm<T, SolutionDim, Element>(
    element: &Element,
    u: &MatrixMN<T, SolutionDim, Element::NodalDim>,
    quadrature: impl Quadrature2d<T>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = U2, ReferenceDim = U2>,
    SolutionDim: DimNameMul<Element::NodalDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, U2, Element::NodalDim>
        + Allocator<T, SolutionDim, Element::NodalDim>
        + Allocator<T, SolutionDim, U1>,
{
    estimate_element_L2_norm_squared(element, u, quadrature).sqrt()
}
