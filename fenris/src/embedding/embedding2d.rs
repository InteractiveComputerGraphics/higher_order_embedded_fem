use crate::connectivity::{
    CellConnectivity, Quad4d2Connectivity, Quad9d2Connectivity, Tri3d2Connectivity, Tri6d2Connectivity,
};
use crate::element::{map_physical_coordinates, ElementConnectivity, FiniteElement, Tri3d2Element};
use crate::geometry::ConvexPolygon;
use crate::mesh::Mesh2d;
use crate::quadrature::Quadrature2d;
use itertools::{izip, Either, Itertools};
use nalgebra::{DefaultAllocator, DimNameMul, Point2, RealField, Vector2, U2};
use std::collections::HashSet;
use std::convert::TryFrom;
use std::error::Error;
use std::iter::repeat;

use crate::allocators::VolumeFiniteElementAllocator;
use crate::connectivity::CellFace;
use crate::embedding::{EmbeddedModel2d, EmbeddedModelBuilder, EmbeddedQuadrature};

pub type EmbeddedQuad4Model<T> = EmbeddedModel2d<T, Quad4d2Connectivity>;
pub type EmbeddedQuad9Model<T> = EmbeddedModel2d<T, Quad9d2Connectivity>;
pub type EmbeddedTri3Model<T> = EmbeddedModel2d<T, Tri3d2Connectivity>;
pub type EmbeddedTri6Model<T> = EmbeddedModel2d<T, Tri6d2Connectivity>;

fn try_convert_cells_to_polygons<T, Connectivity>(
    vertices: &[Point2<T>],
    connectivity: &[Connectivity],
) -> Result<Vec<ConvexPolygon<T>>, String>
where
    T: RealField,
    Connectivity: CellConnectivity<T, U2>,
    ConvexPolygon<T>: TryFrom<Connectivity::Cell>,
{
    let mut polygons = Vec::new();

    for conn in connectivity {
        let cell = conn.cell(vertices).ok_or(String::from(
            "Failed to construct cell from vertices. Index out of bounds?",
        ))?;
        let polygon =
            ConvexPolygon::try_from(cell).map_err(|_| String::from("Failed to construct convex polygon from cell."))?;
        polygons.push(polygon);
    }

    Ok(polygons)
}

/// Given a background mesh and an embedded mesh, find the indices of background cells
/// that intersect the embedded mesh.
pub fn find_background_cell_indices_2d<T, BgCell, EmbedCell>(
    background_mesh: &Mesh2d<T, BgCell>,
    embedded_mesh: &Mesh2d<T, EmbedCell>,
) -> Result<Vec<usize>, String>
where
    T: RealField,
    BgCell: CellConnectivity<T, U2>,
    EmbedCell: CellConnectivity<T, U2>,
    ConvexPolygon<T>: TryFrom<BgCell::Cell>,
    ConvexPolygon<T>: TryFrom<EmbedCell::Cell>,
{
    let embedded_polygons = try_convert_cells_to_polygons(embedded_mesh.vertices(), embedded_mesh.connectivity())?;
    let mut indices = Vec::new();

    for (i, connectivity) in background_mesh.connectivity().iter().enumerate() {
        let cell = connectivity
            .cell(background_mesh.vertices())
            .ok_or(String::from(
                "Failed to construct background cell from vertices. Index out of bounds?",
            ))?;
        let cell_polygon = ConvexPolygon::try_from(cell)
            .map_err(|_| String::from("Failed to create convex polygon from background cell."))?;

        let intersects_embedded = embedded_polygons
            .iter()
            .any(|embedded_poly| !cell_polygon.intersect_polygon(embedded_poly).is_empty());

        if intersects_embedded {
            indices.push(i);
        }
    }

    Ok(indices)
}

/// Given a background mesh and an embedded mesh (a mesh that is embedded into the background mesh),
/// return the indices of the background cells that intersect the boundary interfaces of the
/// embedded mesh.
///
/// The operation returns an error if any of the cells (background or embedded) are non-convex,
/// or more precisely cannot be convert into a convex polygon.
pub fn find_interface_background_cells_2d<T, BgCell, EmbedCell>(
    background_mesh: &Mesh2d<T, BgCell>,
    embedded_mesh: &Mesh2d<T, EmbedCell>,
) -> Result<Vec<usize>, ()>
where
    T: RealField,
    BgCell: CellConnectivity<T, U2>,
    EmbedCell: CellConnectivity<T, U2>,
    EmbedCell::FaceConnectivity: CellConnectivity<T, U2>,
    ConvexPolygon<T>: TryFrom<BgCell::Cell>,
    ConvexPolygon<T>: TryFrom<CellFace<T, EmbedCell>>,
{
    let embedded_boundary_faces: Vec<_> = embedded_mesh
        .find_boundary_faces()
        .into_iter()
        .map(|(connectivity, _, _)| {
            connectivity
                .cell(embedded_mesh.vertices())
                .expect("All embedded mesh vertex indices must be in bounds.")
        })
        .map(|face| ConvexPolygon::try_from(face).map_err(|_| ()))
        .collect::<Result<_, ()>>()?;

    let mut interface_cells = Vec::new();
    for (i, cell_connectivity) in background_mesh.connectivity().iter().enumerate() {
        let cell = cell_connectivity
            .cell(background_mesh.vertices())
            .expect("All background mesh vertex indices must be in bounds.");
        let cell_poly = ConvexPolygon::try_from(cell).map_err(|_| ())?;

        // Here we use the fact that faces are also polyhedra
        // TODO: Use spatial acceleration to improve complexity
        let intersects_interface = embedded_boundary_faces
            .iter()
            .map(|face_poly| cell_poly.intersect_polygon(face_poly))
            .any(|cell_face_intersection| !cell_face_intersection.is_empty());

        if intersects_interface {
            interface_cells.push(i);
        }
    }

    Ok(interface_cells)
}

/// Given a background mesh and an embedded mesh (a mesh that is embedded into the background mesh),
/// returns a vector consisting of tuples `(i, poly)` in which `i` is the index of a background cell
/// that intersects the boundary of the embedded mesh, and `poly` is a vector of convex polygons
/// representing the results of intersecting background cell `i` with all embedded cells.
///
/// The operation returns an error if any of the cells (background or embedded) are non-convex,
/// or more precisely cannot be convert into a convex polygon.
pub fn embed_mesh_2d<T, BgCell, EmbedCell>(
    background_mesh: &Mesh2d<T, BgCell>,
    embedded_mesh: &Mesh2d<T, EmbedCell>,
) -> Result<Vec<(usize, Vec<ConvexPolygon<T>>)>, ()>
where
    T: RealField,
    BgCell: CellConnectivity<T, U2>,
    EmbedCell: CellConnectivity<T, U2>,
    EmbedCell::FaceConnectivity: CellConnectivity<T, U2>,
    ConvexPolygon<T>: TryFrom<BgCell::Cell>,
    ConvexPolygon<T>: TryFrom<EmbedCell::Cell>,
    ConvexPolygon<T>: TryFrom<CellFace<T, EmbedCell>>,
{
    let embedded_polygons: Vec<_> = embedded_mesh
        .connectivity()
        .iter()
        .map(|connectivity| {
            connectivity
                .cell(embedded_mesh.vertices())
                .expect("Embedded cells must not have indices out of bounds")
        })
        .map(|cell| ConvexPolygon::try_from(cell).map_err(|_| ()))
        .collect::<Result<_, ()>>()?;

    let interface_cell_indices = find_interface_background_cells_2d(background_mesh, embedded_mesh)?;
    let mut result = Vec::new();

    for i in interface_cell_indices {
        let connectivity = &background_mesh.connectivity()[i];
        let cell = connectivity
            .cell(background_mesh.vertices())
            .expect("Background cells must not have indices out of bounds");
        let cell_poly = ConvexPolygon::try_from(cell).map_err(|_| ())?;
        let mut intersections = Vec::new();

        // TODO: Spatial acceleration
        for embedded_poly in &embedded_polygons {
            let intersection = cell_poly.intersect_polygon(embedded_poly);
            if !intersection.is_empty() {
                intersections.push(intersection);
            }
        }

        result.push((i, intersections));
    }

    Ok(result)
}

/// Computes intersections between the given polygon and all cells in the embedded geometry.
pub fn embed_cell_2d<T, EmbedCell>(
    polygon: &ConvexPolygon<T>,
    embedded_mesh: &Mesh2d<T, EmbedCell>,
) -> Result<Vec<ConvexPolygon<T>>, Box<dyn std::error::Error>>
where
    T: RealField,
    EmbedCell: CellConnectivity<T, U2>,
    ConvexPolygon<T>: TryFrom<EmbedCell::Cell>,
{
    let mut polygon_intersections = Vec::new();

    for embedded_cell in embedded_mesh.cell_iter() {
        let embedded_polygon = ConvexPolygon::try_from(embedded_cell)
            .map_err(|_| String::from("Could not convert embedded cell to convex polygon."))?;

        let intersection = polygon.intersect_polygon(&embedded_polygon);
        if !intersection.is_empty() {
            polygon_intersections.push(intersection);
        }
    }

    Ok(polygon_intersections)
}

pub fn construct_embedded_quadrature_for_element_2d<T, Element>(
    element: &Element,
    intersected_polygons: &[ConvexPolygon<T>],
    triangle_quadrature: impl Quadrature2d<T>,
) -> (Vec<T>, Vec<Vector2<T>>)
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = U2, ReferenceDim = U2>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, U2, Element::NodalDim>,
{
    intersected_polygons
        .iter()
        .flat_map(ConvexPolygon::triangulate)
        .map(Tri3d2Element::from)
        // TODO: Filter out degenerate triangles (they anyway don't contribute to the
        // integral value
        .flat_map(|tri| izip!(repeat(tri), triangle_quadrature.weights(), triangle_quadrature.points()))
        .map(|(tri, w_tri, xi_tri)| {
            // Map points and weights in reference element for the triangle to the
            // reference element of the background element
            let x = tri.map_reference_coords(xi_tri);
            let j_tri = tri.reference_jacobian(xi_tri);
            let xi_element = map_physical_coordinates(element, &Point2::from(x)).expect("TODO: Handle error");
            let j_element = element.reference_jacobian(&xi_element.coords);

            // Note: we assume that the element is not completely degenerate here
            debug_assert!(j_element.determinant() != T::zero());
            let w_element = *w_tri * j_tri.determinant().abs() / j_element.determinant().abs();
            (w_element, xi_element.coords)
        })
        .unzip()
}

///
///
/// TODO: Return a proper error type
#[allow(non_snake_case)]
pub fn construct_embedded_quadrature<T, Element>(
    embedded_elements: impl IntoIterator<Item = (Element, Vec<ConvexPolygon<T>>)>,
    triangle_quadrature: impl Quadrature2d<T>,
) -> Vec<(Vec<T>, Vec<Vector2<T>>)>
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = U2, ReferenceDim = U2>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, U2, Element::NodalDim>,
{
    embedded_elements
        .into_iter()
        .map(|(element, polygons)| {
            construct_embedded_quadrature_for_element_2d(&element, &polygons, &triangle_quadrature)
        })
        .collect()
}

pub fn construct_embedded_model_2d<T, BgConn, EmbedConn>(
    background_mesh: &Mesh2d<T, BgConn>,
    embedded_mesh: &Mesh2d<T, EmbedConn>,
    triangle_quadrature: &impl Quadrature2d<T>,
    interior_quadrature: (Vec<T>, Vec<Vector2<T>>),
) -> Result<EmbeddedModel2d<T, BgConn>, Box<dyn Error>>
where
    T: RealField,
    BgConn: CellConnectivity<T, U2> + ElementConnectivity<T, GeometryDim = U2, ReferenceDim = U2>,
    EmbedConn: CellConnectivity<T, U2>,
    EmbedConn::FaceConnectivity: CellConnectivity<T, U2>,
    ConvexPolygon<T>: TryFrom<BgConn::Cell>,
    ConvexPolygon<T>: TryFrom<EmbedConn::Cell>,
    ConvexPolygon<T>: TryFrom<CellFace<T, EmbedConn>>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, U2, BgConn::NodalDim>,
    U2: DimNameMul<BgConn::NodalDim>,
    BgConn::NodalDim: DimNameMul<U2>,
{
    let interface_cell_indices: HashSet<usize> = find_interface_background_cells_2d(&background_mesh, embedded_mesh)
        .map_err(|_| String::from("Unknown error when finding background interface cells."))?
        .into_iter()
        .collect();

    let (interface_connectivity, interior_connectivity): (Vec<_>, Vec<_>) = background_mesh
        .connectivity()
        .iter()
        .cloned()
        .enumerate()
        .partition_map(|(i, connectivity)| {
            if interface_cell_indices.contains(&i) {
                Either::Left(connectivity)
            } else {
                Either::Right(connectivity)
            }
        });

    let mut interface_quadratures = Vec::new();
    let mut interface_element_connectivity = Vec::new();
    for connectivity in interface_connectivity {
        let cell = connectivity
            .cell(background_mesh.vertices())
            .ok_or_else(|| String::from("Failed to construct cell from vertices. Index out of bounds?"))?;
        let element = connectivity
            .element(background_mesh.vertices())
            .ok_or_else(|| String::from("Failed to construct element from vertices. Index out of bounds?"))?;
        let cell_polygon =
            ConvexPolygon::try_from(cell).map_err(|_| String::from("Failed to construct convex polygon from cell."))?;
        let intersections = embed_cell_2d(&cell_polygon, embedded_mesh)?;
        let quadrature = construct_embedded_quadrature_for_element_2d(&element, &intersections, &triangle_quadrature);
        interface_quadratures.push(quadrature);
        interface_element_connectivity.push(connectivity);
    }

    let model = EmbeddedModelBuilder::new()
        .vertices(background_mesh.vertices().to_vec())
        .interior_connectivity(interior_connectivity)
        .interface_connectivity(interface_element_connectivity)
        .catchall_quadrature(EmbeddedQuadrature::from_interior_and_interface(
            interior_quadrature,
            interface_quadratures,
        ))
        .build();

    Ok(model)
}
