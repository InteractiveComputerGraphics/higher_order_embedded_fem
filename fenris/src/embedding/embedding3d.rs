use crate::allocators::{ElementConnectivityAllocator, FiniteElementAllocator};
use crate::connectivity::{CellConnectivity, ConnectivityMut};
use crate::element::{map_physical_coordinates, ElementConnectivity, FiniteElement};
use crate::geometry::polymesh::PolyMesh3d;
use crate::geometry::{AxisAlignedBoundingBox, BoundedGeometry, ConvexPolyhedron};
use crate::mesh::{Mesh3d, Tet4Mesh};
use crate::quadrature::{Quadrature, QuadraturePair3d};

use itertools::izip;
use nalgebra::{DefaultAllocator, Point3, RealField, Scalar, U3};
use numeric_literals::replace_float_literals;
use rayon::prelude::*;
use rstar::RTree;

use crate::embedding::{EmbeddedModel3d, EmbeddedModelBuilder, EmbeddedQuadrature};
use std::convert::TryFrom;
use std::error::Error;
use std::iter::repeat;
use std::ops::Add;

use crate::rtree::{rstar_aabb_from_bounding_box_3d, LabeledAABB3d, LabeledGeometry};
use nalgebra::allocator::Allocator;

#[derive(Clone, Debug, PartialEq)]
pub struct Embedding<T: Scalar> {
    /// Background cells that fall outside the embedded geometry.
    pub exterior_cells: Vec<usize>,
    /// Background cells that fall completely inside the embedded geometry.
    pub interior_cells: Vec<usize>,
    /// Background cells that intersect an interface (i.e. boundary) of the
    /// embedded geometry.
    pub interface_cells: Vec<usize>,
    /// For each background interface cell, a poly mesh representing the intersection
    /// of the embedded geometry and the cell.
    pub interface_cell_embeddings: Vec<PolyMesh3d<T>>,
}

impl<T: Scalar> Default for Embedding<T> {
    fn default() -> Self {
        Embedding {
            exterior_cells: vec![],
            interior_cells: vec![],
            interface_cells: vec![],
            interface_cell_embeddings: vec![],
        }
    }
}

enum CellEmbedding<T: Scalar> {
    Exterior,
    Interior,
    Interface { intersection: PolyMesh3d<T> },
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn embed_in_cell<'a, T, Cell>(
    cell: &'a Cell,
    embedded_mesh: &'a PolyMesh3d<T>,
    embedded_cells_rtree: &'a RTree<LabeledAABB3d<T>>,
    opts: &EmbedOptions<T>,
) -> CellEmbedding<T>
where
    T: RealField,
    Cell: ConvexPolyhedron<'a, T> + BoundedGeometry<T, Dimension = U3>,
{
    let aabb = rstar_aabb_from_bounding_box_3d(&cell.bounding_box());
    let mut relevant_embedded_cells: Vec<_> = embedded_cells_rtree
        .locate_in_envelope_intersecting(&aabb)
        .map(|embedded_cell_candidate| embedded_cell_candidate.label)
        .collect();

    // Sorting here is not technically necessary, but it
    // may possibly improve performance if it turns out that the mesh itself
    // has a cache-efficient ordering
    relevant_embedded_cells.sort_unstable();

    let embedded_mesh_region = embedded_mesh.keep_cells(&relevant_embedded_cells);
    let intersection = embedded_mesh_region.intersect_convex_polyhedron(cell);

    if intersection.num_cells() > 0 {
        let bg_cell_volume = cell.compute_volume();
        // Note: Currently, PolyMesh3d::compute_volume() only works if the outside faces are
        // correctly oriented, but we haven't properly accounted for this in the
        // various mesh processing routines. TODO: We need to fix this long term,
        // but for now, we can convert to a tet mesh and compute the volume this way
        // (this is not dependent on face orientations to work correctly).
        let triangulated_intersection = intersection.triangulate().expect(
            "Triangulation should always work in this case, provided that\
                         our input mesh is well formed. TODO: This should be verified by\
                         PolyMesh constructor.",
        );
        let intersected_cell_volume = Tet4Mesh::try_from(&triangulated_intersection)
            .expect(
                "Conversion to tet mesh cannot fail since we have a valid triangulated\
                         PolyMesh.",
            )
            .cell_iter()
            .map(|cell| cell.compute_volume())
            .fold(T::zero(), Add::add);

        if intersected_cell_volume < opts.lower_volume_threshold * bg_cell_volume {
            CellEmbedding::Exterior
        } else if intersected_cell_volume < opts.upper_volume_threshold * bg_cell_volume {
            CellEmbedding::Interface { intersection }
        } else {
            CellEmbedding::Interior
        }
    } else {
        CellEmbedding::Exterior
    }
}

/// Options for mesh embedding.
///
/// ### Thresholds
///
/// Let `cut_cell_volume` be the volume of the intersection between the embedded mesh and a
/// background cell, and let `bg_cell_volume` be the volume of the background cell. Then:
///
/// - If `cut_cell_volume < lower_volume_threshold * bg_cell_volume`, then
///   the cell is designated as an "exterior" cell, which is typically removed
///   from the simulation.
/// - If `cut_cell_volume < upper_volume_threshold * bg_cell_volume`, then
///   the cell is designated as an "interface" cell.
/// - Otherwise, it is designated as an "interior" cell.
///
/// The default settings (`Default::default`) should work well for most/all practical problems.
#[derive(Debug, Clone)]
pub struct EmbedOptions<T> {
    pub upper_volume_threshold: T,
    pub lower_volume_threshold: T,
}

impl<T: RealField> Default for EmbedOptions<T> {
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn default() -> Self {
        Self {
            upper_volume_threshold: 0.999,
            lower_volume_threshold: 1e-4,
        }
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn embed_mesh_3d<T, BgConnectivity>(
    background_mesh: &Mesh3d<T, BgConnectivity>,
    embedded_mesh: &PolyMesh3d<T>,
) -> Embedding<T>
where
    T: RealField,
    BgConnectivity: CellConnectivity<T, U3>,
    BgConnectivity::Cell: Send + BoundedGeometry<T, Dimension = U3> + for<'a> ConvexPolyhedron<'a, T>,
{
    embed_mesh_3d_with_opts(background_mesh, embedded_mesh, &EmbedOptions::default())
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn embed_mesh_3d_with_opts<T, BgConnectivity>(
    background_mesh: &Mesh3d<T, BgConnectivity>,
    embedded_mesh: &PolyMesh3d<T>,
    opts: &EmbedOptions<T>,
) -> Embedding<T>
where
    T: RealField,
    BgConnectivity: CellConnectivity<T, U3>,
    BgConnectivity::Cell: Send + BoundedGeometry<T, Dimension = U3> + for<'a> ConvexPolyhedron<'a, T>,
{
    let embedded_cell_bounding_geometries = embedded_mesh
        .cell_connectivity_iter()
        .enumerate()
        .filter_map(|(cell_idx, face_indices)| {
            let cell_vertices = face_indices
                .iter()
                .copied()
                .flat_map(|i| embedded_mesh.face_vertices(i));
            AxisAlignedBoundingBox::from_points(cell_vertices)
                .map(|bounding_box| LabeledGeometry::new(cell_idx, bounding_box))
        })
        .collect();

    let embedded_cells_rtree = RTree::bulk_load(embedded_cell_bounding_geometries);

    let background_cells: Vec<_> = background_mesh.cell_iter().collect();
    let cell_embeddings: Vec<_> = background_cells
        .into_par_iter()
        .map(|cell| embed_in_cell(&cell, embedded_mesh, &embedded_cells_rtree, opts))
        .collect();

    let mut embedding = Embedding::default();
    for (i, cell_embedding) in cell_embeddings.into_iter().enumerate() {
        match cell_embedding {
            CellEmbedding::Interface { intersection } => {
                embedding.interface_cells.push(i);
                embedding.interface_cell_embeddings.push(intersection);
            }
            CellEmbedding::Interior => {
                embedding.interior_cells.push(i);
            }
            CellEmbedding::Exterior => {
                embedding.exterior_cells.push(i);
            }
        }
    }

    embedding
}

pub struct StabilizationOptions<T: Scalar> {
    // TODO: Only conditionally stabilize?
    // Only stabilize if embedded cell volume <= threshold * bg cell volume
    // pub relative_volume_threshold: Option<T>,
    /// The multiplicative factor for stabilization.
    ///
    /// The stabilization factor gets multiplied with the quadrature weights of the
    /// original quadrature for the background cell.
    pub stabilization_factor: T,
    /// Quadrature used for stabilization. This should normally correspond to an appropriate-order
    /// quadrature rule for the uncut cell.
    pub stabilization_quadrature: QuadraturePair3d<T>,
}

pub struct QuadratureOptions<T: Scalar> {
    pub stabilization: Option<StabilizationOptions<T>>,
}

impl<T: Scalar> Default for QuadratureOptions<T> {
    fn default() -> Self {
        Self { stabilization: None }
    }
}

pub fn compute_element_embedded_quadrature<'a, T, Element>(
    bg_element: &'a Element,
    embedding: &PolyMesh3d<T>,
    tetrahedron_quadrature: impl Quadrature<T, U3>,
    quadrature_options: &QuadratureOptions<T>,
) -> Result<QuadraturePair3d<T>, Box<dyn Error>>
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = U3, ReferenceDim = U3>,
    DefaultAllocator: FiniteElementAllocator<T, U3, Element::ReferenceDim, Element::NodalDim>,
{
    let embedded_tet_mesh = Tet4Mesh::try_from(&embedding.triangulate()?)?;

    let (mut w, mut p): QuadraturePair3d<_> = embedded_tet_mesh
        .connectivity()
        .iter()
        .flat_map(|conn| {
            // Zip each (repeated) element with the full set of tet quadrature points and weights
            // This way we can work on one quadrature point at a time in the following
            izip!(
                repeat(conn.element(embedded_tet_mesh.vertices()).unwrap()),
                tetrahedron_quadrature.weights(),
                tetrahedron_quadrature.points()
            )
        })
        .map(|(tet_element, w_tet, xi_tet)| {
            // Map points and weights in reference element for the tetrahedron to the
            // reference element of the background element
            let x = tet_element.map_reference_coords(xi_tet);
            let j_tet = tet_element.reference_jacobian(xi_tet);
            let xi_bg = map_physical_coordinates(bg_element, &Point3::from(x)).expect("TODO: Handle error");
            let j_bg = bg_element.reference_jacobian(&xi_bg.coords);

            // Note: we assume that the background element is not completely degenerate here
            debug_assert!(j_bg.determinant() != T::zero());
            let w_bg = *w_tet * j_tet.determinant().abs() / j_bg.determinant().abs();
            (w_bg, xi_bg.coords)
        })
        .unzip();

    if let Some(stabilization_options) = &quadrature_options.stabilization {
        let factor = stabilization_options.stabilization_factor;
        let (stab_w, stab_p) = &stabilization_options.stabilization_quadrature;

        w.extend(stab_w.iter().map(|w| factor * *w));
        p.extend(stab_p.iter().cloned());
    }

    Ok((w, p))
}

pub fn embed_quadrature_3d<T, C>(
    background_mesh: &Mesh3d<T, C>,
    embedding: &Embedding<T>,
    interior_quadrature: QuadraturePair3d<T>,
    embed_tet_quadrature_rule: (impl Sync + Quadrature<T, U3>),
) -> Result<EmbeddedQuadrature<T, U3>, Box<dyn Error>>
where
    T: RealField,
    C: Sync + ElementConnectivity<T, GeometryDim = U3, ReferenceDim = U3>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
    DefaultAllocator: Allocator<T, U3>,
    <DefaultAllocator as Allocator<T, U3>>::Buffer: Send + Sync,
{
    embed_quadrature_3d_with_opts(
        background_mesh,
        embedding,
        interior_quadrature,
        embed_tet_quadrature_rule,
        &QuadratureOptions::default(),
    )
}

pub fn embed_quadrature_3d_with_opts<T, C>(
    background_mesh: &Mesh3d<T, C>,
    embedding: &Embedding<T>,
    interior_quadrature: QuadraturePair3d<T>,
    embed_tet_quadrature_rule: (impl Sync + Quadrature<T, U3>),
    quadrature_opts: &QuadratureOptions<T>,
) -> Result<EmbeddedQuadrature<T, U3>, Box<dyn Error>>
where
    T: RealField,
    C: Sync + ElementConnectivity<T, GeometryDim = U3, ReferenceDim = U3>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
    DefaultAllocator: Allocator<T, U3>,
    <DefaultAllocator as Allocator<T, U3>>::Buffer: Send + Sync,
{
    let interface_quadratures =
        compute_interface_quadrature_rules(background_mesh, embedding, embed_tet_quadrature_rule, quadrature_opts)?;
    Ok(EmbeddedQuadrature::from_interior_and_interface(
        interior_quadrature,
        interface_quadratures,
    ))
}

/// Computes quadrature rules for interface elements in the background mesh.
///
/// More precisely, it returns a vector of quadrature rules
pub fn compute_interface_quadrature_rules<T, C>(
    background_mesh: &Mesh3d<T, C>,
    embedding: &Embedding<T>,
    embed_tet_quadrature_rule: (impl Sync + Quadrature<T, U3>),
    quadrature_opts: &QuadratureOptions<T>,
) -> Result<Vec<QuadraturePair3d<T>>, Box<dyn Error>>
where
    T: RealField,
    C: Sync + ElementConnectivity<T, GeometryDim = U3, ReferenceDim = U3>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
    DefaultAllocator: Allocator<T, U3>,
    <DefaultAllocator as Allocator<T, U3>>::Buffer: Send + Sync,
{
    let tet_quadrature = &embed_tet_quadrature_rule;
    assert_eq!(
        embedding.interface_cells.len(),
        embedding.interface_cell_embeddings.len()
    );

    let result: Result<Vec<_>, _> = embedding
        .interface_cells
        .par_iter()
        .zip(embedding.interface_cell_embeddings.par_iter())
        .map(|(bg_cell_idx, embedded_intersection)| {
            let element = background_mesh
                .connectivity()
                .get(*bg_cell_idx)
                .ok_or_else(|| Box::<dyn Error + Send + Sync>::from("Invalid interface cell index."))?
                .element(background_mesh.vertices())
                .unwrap();
            compute_element_embedded_quadrature(&element, embedded_intersection, tet_quadrature, quadrature_opts)
                // We cannot pass through the error since we cannot (or don't want to) guarantee
                // that the error implements Send
                .map_err(|err| {
                    Box::<dyn Error + Sync + Send>::from(format!(
                        "Failed to construct embedded. quadrature for element.\
                    Error description: {}",
                        err
                    ))
                })
        })
        .collect();

    result
        // For some reason, dyn Error + Sync + Send does not automatically coerce
        // to dyn Error in this case, so we need a simple cast
        .map_err(|err| err as _)
}

/// Constructs embedded model with catchall quadrature.
pub fn construct_embedded_model_3d<T, C>(
    background_mesh: &Mesh3d<T, C>,
    embedded_mesh: &PolyMesh3d<T>,
    interior_quadrature: QuadraturePair3d<T>,
    tet_quadrature: (impl Sync + Quadrature<T, U3>),
) -> Result<EmbeddedModel3d<T, C>, Box<dyn Error>>
where
    T: RealField,
    C: Sync + ConnectivityMut + CellConnectivity<T, U3> + ElementConnectivity<T, GeometryDim = U3, ReferenceDim = U3>,
    C::Cell: Send + BoundedGeometry<T, Dimension = U3> + for<'a> ConvexPolyhedron<'a, T>,
    DefaultAllocator: ElementConnectivityAllocator<T, C> + Allocator<T, U3>,
    <DefaultAllocator as Allocator<T, U3>>::Buffer: Send + Sync,
{
    construct_embedded_model_3d_with_opts(
        background_mesh,
        embedded_mesh,
        interior_quadrature,
        tet_quadrature,
        &EmbedOptions::default(),
    )
}

/// Constructs embedded model with catchall quadrature.
///
/// TODO: Consider deprecating this?
pub fn construct_embedded_model_3d_with_opts<T, C>(
    background_mesh: &Mesh3d<T, C>,
    embedded_mesh: &PolyMesh3d<T>,
    interior_quadrature: QuadraturePair3d<T>,
    tet_quadrature: (impl Sync + Quadrature<T, U3>),
    opts: &EmbedOptions<T>,
) -> Result<EmbeddedModel3d<T, C>, Box<dyn Error>>
where
    T: RealField,
    C: Sync + ConnectivityMut + CellConnectivity<T, U3> + ElementConnectivity<T, GeometryDim = U3, ReferenceDim = U3>,
    C::Cell: Send + BoundedGeometry<T, Dimension = U3> + for<'a> ConvexPolyhedron<'a, T>,
    DefaultAllocator: ElementConnectivityAllocator<T, C> + Allocator<T, U3>,
    <DefaultAllocator as Allocator<T, U3>>::Buffer: Send + Sync,
{
    let embedding = embed_mesh_3d_with_opts(background_mesh, embedded_mesh, opts);
    let interface_quadrature_rules = compute_interface_quadrature_rules(
        background_mesh,
        &embedding,
        tet_quadrature,
        &QuadratureOptions::default(),
    )
    .map_err(|err| Box::<dyn Error>::from(err))?;

    Ok(EmbeddedModelBuilder::from_embedding(background_mesh, embedding)
        .catchall_quadrature(EmbeddedQuadrature::from_interior_and_interface(
            interior_quadrature,
            interface_quadrature_rules,
        ))
        .build())
}
