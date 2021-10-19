//! Helper traits for collecting element allocator trait bounds.

use crate::element::{ConnectivityGeometryDim, ConnectivityNodalDim, ConnectivityReferenceDim, ElementConnectivity};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, DimNameMul, DimNameProd, Scalar, U1};

/// Helper trait to make specifying bounds on generic functions working with the
/// `ReferenceFiniteElement` trait easier.
pub trait ReferenceFiniteElementAllocator<T, ReferenceDim, NodalDim>:
Allocator<T, U1, NodalDim>
+ Allocator<T, NodalDim, U1>
+ Allocator<T, ReferenceDim, NodalDim>
+ Allocator<T, ReferenceDim, ReferenceDim>
+ Allocator<T, ReferenceDim, U1>
+ Allocator<T, U1, ReferenceDim>
+ Allocator<T, ReferenceDim, NodalDim>
// For representing the indices of the nodes
+ Allocator<usize, NodalDim, U1>
+ Allocator<(usize, usize), NodalDim>
+ Allocator<(usize, usize), ReferenceDim>
    where
        T: Scalar,
        ReferenceDim: DimName,
        NodalDim: DimName,
{

}

/// Helper trait to make specifying bounds on generic functions working with the
/// `FiniteElement` trait easier.
pub trait FiniteElementAllocator<T, GeometryDim, ReferenceDim, NodalDim>:
ReferenceFiniteElementAllocator<T, ReferenceDim, NodalDim>
+ Allocator<T, GeometryDim>
+ Allocator<T, U1, NodalDim>
+ Allocator<T, NodalDim, U1>
+ Allocator<T, NodalDim, GeometryDim>
+ Allocator<T, GeometryDim, NodalDim>
+ Allocator<T, GeometryDim, GeometryDim>
+ Allocator<T, GeometryDim, ReferenceDim>
+ Allocator<T, GeometryDim, U1>
+ Allocator<T, U1, GeometryDim>
+ Allocator<T, ReferenceDim, NodalDim>
+ Allocator<T, ReferenceDim, ReferenceDim>
+ Allocator<T, ReferenceDim, U1>
+ Allocator<T, U1, ReferenceDim>
+ Allocator<T, ReferenceDim, NodalDim>
+ Allocator<T, ReferenceDim, GeometryDim>
// For representing the indices of the nodes
+ Allocator<usize, NodalDim, U1>
+ Allocator<(usize, usize), GeometryDim>
+ Allocator<(usize, usize), NodalDim>
+ Allocator<(usize, usize), ReferenceDim>
    where
        T: Scalar,
        GeometryDim: DimName,
        ReferenceDim: DimName,
        NodalDim: DimName,
{

}

/// Helper trait to make specifying bounds on generic functions working with the
/// `FiniteElement` trait easier, for elements whose geometry dimension and reference element
/// dimension coincide.
pub trait VolumeFiniteElementAllocator<T, GeometryDim, NodalDim>:
    FiniteElementAllocator<T, GeometryDim, GeometryDim, NodalDim>
where
    T: Scalar,
    GeometryDim: DimName,
    NodalDim: DimName,
{
}

/// Helper trait to simplify specifying bounds on generic functions that need to
/// construct element (mass/stiffness) matrices when working with the `FiniteElement` trait.
pub trait FiniteElementMatrixAllocator<T, SolutionDim, GeometryDim, NodalDim>:
    VolumeFiniteElementAllocator<T, GeometryDim, NodalDim>
    + Allocator<T, DimNameProd<SolutionDim, NodalDim>, DimNameProd<SolutionDim, NodalDim>>
    + Allocator<T, NodalDim, NodalDim>
    + Allocator<T, NodalDim, SolutionDim>
    + Allocator<T, SolutionDim, GeometryDim>
    + Allocator<T, SolutionDim, NodalDim>
    + Allocator<T, SolutionDim, SolutionDim>
    + Allocator<T, GeometryDim, SolutionDim>
    + Allocator<(usize, usize), SolutionDim>
where
    T: Scalar,
    GeometryDim: DimName,
    SolutionDim: DimNameMul<NodalDim>,
    NodalDim: DimName,
{
}

impl<T, ReferenceDim, NodalDim> ReferenceFiniteElementAllocator<T, ReferenceDim, NodalDim> for DefaultAllocator
where
    T: Scalar,
    ReferenceDim: DimName,
    NodalDim: DimName,
    DefaultAllocator: Allocator<T, U1, NodalDim>
        + Allocator<T, NodalDim, U1>
        + Allocator<T, ReferenceDim, NodalDim>
        + Allocator<T, ReferenceDim, ReferenceDim>
        + Allocator<T, ReferenceDim, U1>
        + Allocator<T, U1, ReferenceDim>
        + Allocator<T, ReferenceDim, NodalDim>
        + Allocator<usize, NodalDim, U1>
        + Allocator<(usize, usize), NodalDim>
        + Allocator<(usize, usize), ReferenceDim>,
{
}

impl<T, GeometryDim, ReferenceDim, NodalDim> FiniteElementAllocator<T, GeometryDim, ReferenceDim, NodalDim>
    for DefaultAllocator
where
    T: Scalar,
    GeometryDim: DimName,
    NodalDim: DimName,
    ReferenceDim: DimName,
    DefaultAllocator: ReferenceFiniteElementAllocator<T, ReferenceDim, NodalDim>
        + Allocator<T, GeometryDim>
        + Allocator<T, U1, NodalDim>
        + Allocator<T, NodalDim, U1>
        + Allocator<T, NodalDim, GeometryDim>
        + Allocator<T, GeometryDim, NodalDim>
        + Allocator<T, GeometryDim, GeometryDim>
        + Allocator<T, GeometryDim, ReferenceDim>
        + Allocator<T, GeometryDim, U1>
        + Allocator<T, U1, GeometryDim>
        + Allocator<T, ReferenceDim, NodalDim>
        + Allocator<T, ReferenceDim, ReferenceDim>
        + Allocator<T, ReferenceDim, U1>
        + Allocator<T, U1, ReferenceDim>
        + Allocator<T, ReferenceDim, NodalDim>
        + Allocator<T, ReferenceDim, GeometryDim>
        + Allocator<usize, NodalDim, U1>
        + Allocator<(usize, usize), GeometryDim>
        + Allocator<(usize, usize), NodalDim>
        + Allocator<(usize, usize), ReferenceDim>,
{
}

impl<T, SolutionDim, GeometryDim, NodalDim> FiniteElementMatrixAllocator<T, SolutionDim, GeometryDim, NodalDim>
    for DefaultAllocator
where
    T: Scalar,
    SolutionDim: DimNameMul<NodalDim>,
    GeometryDim: DimName,
    NodalDim: DimName,
    DefaultAllocator: VolumeFiniteElementAllocator<T, GeometryDim, NodalDim>
        + Allocator<T, DimNameProd<SolutionDim, NodalDim>, DimNameProd<SolutionDim, NodalDim>>
        + Allocator<T, NodalDim, NodalDim>
        + Allocator<T, NodalDim, SolutionDim>
        + Allocator<T, SolutionDim, GeometryDim>
        + Allocator<T, SolutionDim, NodalDim>
        + Allocator<T, SolutionDim, SolutionDim>
        + Allocator<T, GeometryDim, SolutionDim>
        + Allocator<(usize, usize), SolutionDim>,
{
}

impl<T, GeometryDim, NodalDim> VolumeFiniteElementAllocator<T, GeometryDim, NodalDim> for DefaultAllocator
where
    T: Scalar,
    GeometryDim: DimName,
    NodalDim: DimName,
    DefaultAllocator: FiniteElementAllocator<T, GeometryDim, GeometryDim, NodalDim>,
{
}

pub trait ElementConnectivityAllocator<T, Connectivity>:
    FiniteElementAllocator<
    T,
    ConnectivityGeometryDim<T, Connectivity>,
    ConnectivityReferenceDim<T, Connectivity>,
    ConnectivityNodalDim<T, Connectivity>,
>
where
    T: Scalar,
    Connectivity: ElementConnectivity<T>,
    DefaultAllocator: FiniteElementAllocator<
        T,
        ConnectivityGeometryDim<T, Connectivity>,
        ConnectivityReferenceDim<T, Connectivity>,
        ConnectivityNodalDim<T, Connectivity>,
    >,
{
}

impl<T, C> ElementConnectivityAllocator<T, C> for DefaultAllocator
where
    T: Scalar,
    C: ElementConnectivity<T>,
    DefaultAllocator: FiniteElementAllocator<
        T,
        ConnectivityGeometryDim<T, C>,
        ConnectivityReferenceDim<T, C>,
        ConnectivityNodalDim<T, C>,
    >,
{
}
