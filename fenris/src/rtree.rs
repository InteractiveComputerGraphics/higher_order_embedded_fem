//! Helper functionality for working with an RTree for spatial acceleration.
use crate::allocators::ElementConnectivityAllocator;
use crate::element::ElementConnectivity;
use crate::geometry::{
    AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, AxisAlignedBoundingBox3d, BoundedGeometry, Distance,
    DistanceQuery, GeometryCollection,
};
use crate::model::{FiniteElementInterpolator, MakeInterpolator};
use crate::space::{FiniteElementSpace, GeometricFiniteElementSpace};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimMin, DimName, Point, Point3, RealField, Scalar, U2, U3};
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::marker::PhantomData;

pub type LabeledAABB3d<T> = LabeledGeometry<T, AxisAlignedBoundingBox3d<T>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabeledGeometry<T: Scalar, Geometry> {
    pub label: usize,
    pub geometry: Geometry,
    marker: PhantomData<T>,
}

impl<T: Scalar, Geometry> LabeledGeometry<T, Geometry> {
    pub fn new(label: usize, geometry: Geometry) -> Self {
        Self {
            label,
            geometry,
            marker: PhantomData,
        }
    }
}

pub trait RTreeDim<T: Scalar>: DimName
where
    DefaultAllocator: Allocator<T, Self>,
{
    type Envelope: rstar::Envelope;

    fn envelope_from_bounding_box(bounding_box: &AxisAlignedBoundingBox<T, Self>) -> Self::Envelope;

    fn translate_point(point: &Point<T, Self>) -> <Self::Envelope as rstar::Envelope>::Point;
}

impl<T: RealField> RTreeDim<T> for U2 {
    type Envelope = AABB<[T; 2]>;

    fn envelope_from_bounding_box(bounding_box: &AxisAlignedBoundingBox<T, Self>) -> Self::Envelope {
        rstar_aabb_from_bounding_box_2d(bounding_box)
    }

    fn translate_point(point: &Point<T, Self>) -> [T; 2] {
        point.coords.into()
    }
}

impl<T: RealField> RTreeDim<T> for U3 {
    type Envelope = AABB<[T; 3]>;

    fn envelope_from_bounding_box(bounding_box: &AxisAlignedBoundingBox<T, Self>) -> Self::Envelope {
        rstar_aabb_from_bounding_box_3d(bounding_box)
    }

    fn translate_point(point: &Point<T, Self>) -> [T; 3] {
        point.coords.into()
    }
}

impl<T, Geometry> RTreeObject for LabeledGeometry<T, Geometry>
where
    T: RealField,
    Geometry: BoundedGeometry<T>,
    Geometry::Dimension: RTreeDim<T>,
    DefaultAllocator: Allocator<T, Geometry::Dimension>,
{
    type Envelope = <Geometry::Dimension as RTreeDim<T>>::Envelope;

    fn envelope(&self) -> Self::Envelope {
        Geometry::Dimension::envelope_from_bounding_box(&self.geometry.bounding_box())
    }
}

impl<T, Geometry> PointDistance for LabeledGeometry<T, Geometry>
where
    T: RealField,
    Geometry: BoundedGeometry<T, Dimension = U3> + Distance<T, Point3<T>>,
{
    // TODO: Consider implementing the other functions in this trait
    fn distance_2(&self, point: &[T; 3]) -> T {
        let [x, y, z] = *point;
        let d = self.geometry.distance(&Point3::new(x, y, z));
        d * d
    }
}

pub struct GeometryCollectionAccelerator<'a, T, Collection>
where
    T: RealField,
    Collection: GeometryCollection<'a>,
    Collection::Geometry: BoundedGeometry<T>,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject,
    DefaultAllocator: Allocator<T, <Collection::Geometry as BoundedGeometry<T>>::Dimension>,
{
    collection: &'a Collection,
    r_tree: RTree<LabeledGeometry<T, Collection::Geometry>>,
}

impl<'a, T, Collection> GeometryCollectionAccelerator<'a, T, Collection>
where
    T: RealField,
    Collection: GeometryCollection<'a>,
    Collection::Geometry: BoundedGeometry<T>,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject,
    DefaultAllocator: Allocator<T, <Collection::Geometry as BoundedGeometry<T>>::Dimension>,
{
    pub fn new(collection: &'a Collection) -> Self {
        let geometries = (0..collection.num_geometries())
            .map(|i| LabeledGeometry::new(i, collection.get_geometry(i).unwrap()))
            .collect();

        Self {
            collection,
            r_tree: RTree::bulk_load(geometries),
        }
    }
}

impl<'a, 'b, T, Collection> GeometryCollection<'b> for GeometryCollectionAccelerator<'a, T, Collection>
where
    'a: 'b,
    T: RealField,
    Collection: GeometryCollection<'a>,
    Collection::Geometry: BoundedGeometry<T>,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject,
    DefaultAllocator: Allocator<T, <Collection::Geometry as BoundedGeometry<T>>::Dimension>,
{
    type Geometry = Collection::Geometry;

    fn num_geometries(&self) -> usize {
        self.collection.num_geometries()
    }

    fn get_geometry(&'b self, index: usize) -> Option<Self::Geometry> {
        self.collection.get_geometry(index)
    }
}

impl<'a, 'b, T, D, Collection> DistanceQuery<'b, Point<T, D>> for GeometryCollectionAccelerator<'a, T, Collection>
where
    'a: 'b,
    T: RealField,
    D: RTreeDim<T>,
    Collection: GeometryCollection<'a>,
    Collection::Geometry: BoundedGeometry<T, Dimension = D>,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject<Envelope = D::Envelope> + PointDistance,
    DefaultAllocator: Allocator<T, D>,
{
    fn nearest(&'b self, query_geometry: &Point<T, D>) -> Option<usize> {
        self.r_tree
            .nearest_neighbor(&D::translate_point(query_geometry))
            .map(|labeled_geometry| labeled_geometry.label)
    }
}

impl<'a, T, Collection> FiniteElementSpace<T> for GeometryCollectionAccelerator<'a, T, Collection>
where
    T: RealField,
    Collection: FiniteElementSpace<T>,
    Collection: GeometryCollection<'a>,
    Collection::Geometry: BoundedGeometry<
        T,
        Dimension = <<Collection as FiniteElementSpace<T>>::Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject,
    DefaultAllocator: ElementConnectivityAllocator<T, <Collection as FiniteElementSpace<T>>::Connectivity>,
{
    type Connectivity = <Collection as FiniteElementSpace<T>>::Connectivity;

    fn vertices(&self) -> &[Point<T, <Collection::Geometry as BoundedGeometry<T>>::Dimension>] {
        self.collection.vertices()
    }

    fn num_connectivities(&self) -> usize {
        self.collection.num_connectivities()
    }

    fn get_connectivity(&self, index: usize) -> Option<&Self::Connectivity> {
        self.collection.get_connectivity(index)
    }
}

impl<'a, 'b, T, Collection> GeometricFiniteElementSpace<'b, T> for GeometryCollectionAccelerator<'a, T, Collection>
where
    'a: 'b,
    T: RealField,
    Collection: GeometricFiniteElementSpace<'a, T>,
    Collection::Geometry: BoundedGeometry<
        T,
        Dimension = <<Collection as FiniteElementSpace<T>>::Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject,
    DefaultAllocator: ElementConnectivityAllocator<T, <Collection as FiniteElementSpace<T>>::Connectivity>,
{
}

impl<'a, T, D, Collection> MakeInterpolator<T, D> for GeometryCollectionAccelerator<'a, T, Collection>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D> + RTreeDim<T>,
    Collection: GeometricFiniteElementSpace<'a, T> + DistanceQuery<'a, Point<T, D>>,
    Collection::Geometry: BoundedGeometry<T, Dimension = D>,
    Collection::Connectivity: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    LabeledGeometry<T, Collection::Geometry>: RTreeObject<Envelope = D::Envelope> + PointDistance,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: ElementConnectivityAllocator<T, Collection::Connectivity>,
{
    fn make_interpolator(
        &self,
        interpolation_points: &[Point<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn Error>> {
        FiniteElementInterpolator::interpolate_space(self, interpolation_points)
    }
}

pub fn rstar_aabb_from_bounding_box_3d<T: RealField>(bounding_box: &AxisAlignedBoundingBox3d<T>) -> AABB<[T; 3]> {
    AABB::from_corners(bounding_box.min().clone().into(), bounding_box.max().clone().into())
}

pub fn rstar_aabb_from_bounding_box_2d<T: RealField>(bounding_box: &AxisAlignedBoundingBox2d<T>) -> AABB<[T; 2]> {
    AABB::from_corners(bounding_box.min().clone().into(), bounding_box.max().clone().into())
}
