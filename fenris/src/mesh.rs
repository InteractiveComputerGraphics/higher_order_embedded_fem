use crate::geometry::{
    AxisAlignedBoundingBox, BoundedGeometry, Distance, DistanceQuery, GeneralPolygon, GeometryCollection,
    LineSegment2d, Polygon, SignedDistance, SignedDistanceResult,
};
use arrayvec::ArrayVec;
use nalgebra::{DefaultAllocator, DimName, Point, Point2, RealField, Scalar, Vector2, VectorN, U2, U3};
use nested_vec::NestedVec;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::{BTreeMap, HashMap};
use std::iter::once;

use numeric_literals::replace_float_literals;

use crate::connectivity::{
    CellConnectivity, Connectivity, ConnectivityMut, Hex20Connectivity, Hex27Connectivity, Hex8Connectivity,
    Quad4d2Connectivity, Quad9d2Connectivity, Segment2d2Connectivity, Tet10Connectivity, Tet20Connectivity,
    Tet4Connectivity, Tri3d2Connectivity, Tri3d3Connectivity, Tri6d2Connectivity,
};
use crate::geometry::Orientation::Counterclockwise;
use nalgebra::allocator::Allocator;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::error::Error;

/// Index-based data structure for conforming meshes (i.e. no hanging nodes).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Mesh<T: Scalar, D, Connectivity>
where
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    // serde's not able correctly determine the necessary trait bounds in this case,
    // so write our own
    #[serde(bound(
        serialize = "<DefaultAllocator as Allocator<T, D>>::Buffer: Serialize",
        deserialize = "<DefaultAllocator as Allocator<T, D>>::Buffer: Deserialize<'de>"
    ))]
    vertices: Vec<Point<T, D>>,
    #[serde(bound(
        serialize = "Connectivity: Serialize",
        deserialize = "Connectivity: Deserialize<'de>"
    ))]
    connectivity: Vec<Connectivity>,
}

/// Index-based data structure for conforming meshes (i.e. no hanging nodes).
pub type Mesh2d<T, Connectivity> = Mesh<T, U2, Connectivity>;
pub type Mesh3d<T, Connectivity> = Mesh<T, U3, Connectivity>;

pub type TriangleMesh2d<T> = Mesh2d<T, Tri3d2Connectivity>;
pub type Tri6Mesh2d<T> = Mesh2d<T, Tri6d2Connectivity>;
pub type QuadMesh2d<T> = Mesh2d<T, Quad4d2Connectivity>;
pub type Quad9Mesh2d<T> = Mesh2d<T, Quad9d2Connectivity>;
pub type TriangleMesh3d<T> = Mesh3d<T, Tri3d3Connectivity>;
// TODO: Rename to Hex8Mesh
pub type HexMesh<T> = Mesh3d<T, Hex8Connectivity>;
pub type Hex20Mesh<T> = Mesh3d<T, Hex20Connectivity>;
pub type Hex27Mesh<T> = Mesh3d<T, Hex27Connectivity>;
pub type Tet4Mesh<T> = Mesh3d<T, Tet4Connectivity>;
pub type Tet10Mesh<T> = Mesh3d<T, Tet10Connectivity>;
pub type Tet20Mesh<T> = Mesh3d<T, Tet20Connectivity>;

impl<T, D, Connectivity> Mesh<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn vertices_mut(&mut self) -> &mut [Point<T, D>] {
        &mut self.vertices
    }

    pub fn vertices(&self) -> &[Point<T, D>] {
        &self.vertices
    }

    pub fn connectivity(&self) -> &[Connectivity] {
        &self.connectivity
    }

    /// Construct a mesh from vertices and connectivity.
    ///
    /// The provided connectivity is expected only to return valid (i.e. in-bounds) indices,
    /// but this can not be trusted. Users of the mesh are permitted to panic if they encounter
    /// invalid indices, but unchecked indexing may easily lead to undefined behavior.
    ///
    /// In other words, if the connectivity references indices out of bounds, then the code is
    /// incorrect. However, since this can be done exclusively with safe code, unchecked
    /// or unsafe indexing in which the user is *trusted* to provide valid indices may
    /// produce undefined behavior.Therefore, the connectivity must always be checked.
    pub fn from_vertices_and_connectivity(vertices: Vec<Point<T, D>>, connectivity: Vec<Connectivity>) -> Self {
        Self { vertices, connectivity }
    }
}

impl<T, D, C> Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: ConnectivityMut,
    DefaultAllocator: Allocator<T, D>,
{
    /// Creates a new mesh with each cell disconnected from all its neighbors.
    ///
    /// In other words, each vertex is only referenced exactly once, and the result is
    /// effectively a "soup" of cells.
    pub fn disconnect_cells(&self) -> Self {
        let old_vertices = self.vertices();
        let mut new_vertices = Vec::new();
        let mut new_connectivity = Vec::new();

        for conn in self.connectivity() {
            let mut new_conn = conn.clone();

            for v_idx in new_conn.vertex_indices_mut() {
                let new_vertex_idx = new_vertices.len();
                new_vertices.push(old_vertices[*v_idx].clone());
                *v_idx = new_vertex_idx;
            }
            new_connectivity.push(new_conn);
        }

        Self::from_vertices_and_connectivity(new_vertices, new_connectivity)
    }
}

impl<T, D, Connectivity> Mesh<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    Connectivity: CellConnectivity<T, D>,
{
    pub fn get_cell(&self, index: usize) -> Option<Connectivity::Cell> {
        self.connectivity()
            .get(index)
            .and_then(|conn| conn.cell(self.vertices()))
    }

    pub fn cell_iter<'a>(&'a self) -> impl 'a + Iterator<Item = Connectivity::Cell> {
        self.connectivity().iter().map(move |connectivity| {
            connectivity
                .cell(&self.vertices)
                .expect("Mesh2d is not allowed to contain cells with indices out of bounds.")
        })
    }
}

impl<T, D, C> Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    C::FaceConnectivity: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    /// Finds cells that have at least one boundary face.
    pub fn find_boundary_cells(&self) -> Vec<usize> {
        let mut cells: Vec<_> = self
            .find_boundary_faces()
            .into_iter()
            .map(|(_, cell_index, _)| cell_index)
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// Finds faces which are only connected to exactly one cell, along with the connected cell
    /// index and the local index of the face within that cell.
    pub fn find_boundary_faces(&self) -> Vec<(C::FaceConnectivity, usize, usize)> {
        let mut sorted_slices = NestedVec::new();
        let mut face_info = Vec::new();

        // We want to use (sorted) slices as keys in a hash map, so we need to store
        // and sort the slices first
        for (conn_idx, cell_conn) in self.connectivity.iter().enumerate() {
            let num_faces = cell_conn.num_faces();
            for local_idx in 0..num_faces {
                let face_conn = cell_conn.get_face_connectivity(local_idx).unwrap();
                sorted_slices.push(face_conn.vertex_indices());
                let indices = sorted_slices.last_mut().unwrap();
                indices.sort_unstable();
                face_info.push((face_conn, conn_idx, local_idx));
            }
        }

        // Count the number of occurrences of "equivalent" faces (in the sense that they refer
        // to the same vertex indices). Use a BTreeMap to avoid non-determinism due to
        // HashMap's internal randomization.
        let mut slice_counts = BTreeMap::new();
        let num_slices = sorted_slices.len();
        for i in 0..num_slices {
            slice_counts
                .entry(sorted_slices.get(i).unwrap())
                .and_modify(|(_, count)| *count += 1)
                .or_insert((i, 1));
        }

        // Take only the faces which have a count of 1, which correspond to boundary faces
        slice_counts
            .into_iter()
            .map(|(_key, value)| value)
            .filter(|&(_, count)| count == 1)
            .map(move |(i, _)| face_info[i].clone())
            .collect()
    }

    /// Returns a sorted list of vertices that are determined to be on the boundary.
    ///
    /// A vertex is considered to be a part of the boundary if it belongs to a boundary face.
    pub fn find_boundary_vertices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for (connectivity, _, _) in self.find_boundary_faces() {
            indices.extend(connectivity.vertex_indices());
        }
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

impl<T, D, Connectivity> BoundedGeometry<T> for Mesh<T, D, Connectivity>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    Connectivity: CellConnectivity<T, D>,
    Connectivity::Cell: BoundedGeometry<T, Dimension = D>,
{
    type Dimension = D;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, D> {
        let mut bbs = self.cell_iter().map(|cell| cell.bounding_box());
        bbs.next()
            .map(|first_bb| bbs.fold(first_bb, |bb1, bb2| bb1.enclose(&bb2)))
            .unwrap_or_else(|| AxisAlignedBoundingBox::new(VectorN::zeros(), VectorN::zeros()))
    }
}

impl<T, D, C> Mesh<T, D, C>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Translates all vertices of the mesh by the given translation vector.
    pub fn translate(&mut self, translation: &VectorN<T, D>) {
        self.transform_vertices(|p| *p += translation);
    }

    /// Transform all vertices of the mesh by the given transformation function.
    pub fn transform_vertices<F>(&mut self, mut transformation: F)
    where
        F: FnMut(&mut Point<T, D>),
    {
        for p in &mut self.vertices {
            transformation(p);
        }
    }

    pub fn transform_all_vertices<F>(&mut self, mut transformation: F)
    where
        F: FnMut(&mut [Point<T, D>]),
    {
        transformation(&mut self.vertices);
    }
}

const EMPTY_SLICE: &[usize] = &[];

impl Connectivity for () {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &EMPTY_SLICE
    }
}

impl<T> QuadMesh2d<T>
where
    T: RealField,
{
    pub fn split_into_triangles(self) -> TriangleMesh2d<T> {
        let triangles = self
            .connectivity()
            .iter()
            .flat_map(|connectivity| {
                let Quad4d2Connectivity(c) = connectivity;
                let quad = connectivity
                    .cell(self.vertices())
                    .expect("Indices must be in bounds");
                let (tri1, tri2) = quad.split_into_triangle_connectivities();
                let tri1_global = Tri3d2Connectivity([c[tri1[0]], c[tri1[1]], c[tri1[2]]]);
                let tri2_global = Tri3d2Connectivity([c[tri2[0]], c[tri2[1]], c[tri2[2]]]);
                once(tri1_global).chain(once(tri2_global))
            })
            .collect();

        TriangleMesh2d::from_vertices_and_connectivity(self.vertices, triangles)
    }
}

impl<T, D, C> Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: ConnectivityMut,
    DefaultAllocator: Allocator<T, D>,
{
    /// Returns a new mesh in which only the desired cells are kept. The vertices are removed or
    /// relabeled as necessary.
    pub fn keep_cells(&self, cell_indices: &[usize]) -> Self {
        // TODO: Return Result instead of panicking if indices are out of bounds

        // Each entry is true if this vertex should be kept, false otherwise
        let vertex_keep_table = {
            let mut table = vec![false; self.vertices.len()];
            for cell_index in cell_indices {
                let cell_connectivity = &self.connectivity[*cell_index];

                for vertex_index in cell_connectivity.vertex_indices() {
                    table[*vertex_index] = true;
                }
            }
            table
        };

        let old_to_new_label_map = {
            let mut label_map = HashMap::new();
            let mut next_label = 0;
            for (i, keep) in vertex_keep_table.iter().enumerate() {
                if *keep {
                    label_map.insert(i, next_label);
                    next_label += 1;
                }
            }
            label_map
        };

        let relabeled_cells: Vec<_> = cell_indices
            .iter()
            .map(|i| self.connectivity()[*i].clone())
            .map(|mut cell| {
                for index in cell.vertex_indices_mut() {
                    *index = *old_to_new_label_map
                        .get(index)
                        .expect("Index must be in map");
                }
                cell
            })
            .collect();

        let relabeled_vertices: Vec<_> = vertex_keep_table
            .iter()
            .enumerate()
            .filter_map(|(i, should_keep)| if *should_keep { Some(i) } else { None })
            .map(|index| self.vertices[index].clone())
            .collect();

        Mesh::from_vertices_and_connectivity(relabeled_vertices, relabeled_cells)
    }
}

impl<T, Cell> Mesh2d<T, Cell>
where
    T: RealField,
    Cell: Connectivity<FaceConnectivity = Segment2d2Connectivity>,
{
    pub fn extract_contour(&self) -> Result<GeneralPolygon<T>, Box<dyn Error>> {
        let boundary_edges = self
            .find_boundary_faces()
            .into_iter()
            .map(|(edge, _, _)| edge);

        // For a "proper" mesh, any vertex may be connected to exactly two other vertices.
        // We build a "path" of vertices by associating each vertex with its neighbor
        // whose index is the smallest, and visiting each vertex once.
        let mut neighbors = HashMap::new();
        let mut smallest_index = std::usize::MAX;

        let mut insert_neighbor = |vertex_index, neighbor_index| {
            if vertex_index == neighbor_index {
                Err(format!(
                    "Cannot extract contour: vertex {} has edge to itself.",
                    vertex_index
                ))
            } else {
                neighbors
                    .entry(vertex_index)
                    .or_insert_with(|| ArrayVec::<[_; 2]>::new())
                    .try_push(neighbor_index)
                    .map_err(|_| {
                        format!(
                            "Cannot extract contour: vertex {} has more than two neighbors.",
                            vertex_index
                        )
                    })
            }
        };

        for edge in boundary_edges {
            let Segment2d2Connectivity([a, b]) = edge;
            insert_neighbor(a, b)?;
            insert_neighbor(b, a)?;
            smallest_index = min(smallest_index, a);
            smallest_index = min(smallest_index, b);
        }

        let num_vertices = neighbors.len();
        let mut take_next = |vertex_index, prev_index| {
            debug_assert_ne!(vertex_index, prev_index);
            let vertex_neighbors = neighbors
                .get_mut(&vertex_index)
                .expect("All vertices have neighbors");

            const ERROR_MSG: &str = "Cannot extract contour: There is no closed path connecting vertices.";

            if vertex_neighbors.is_empty() {
                Err(ERROR_MSG)
            } else {
                let neighbor_idx = vertex_neighbors
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter(|(_, vertex_idx)| *vertex_idx != prev_index)
                    .map(|(i, _)| i)
                    .next();

                if let Some(neighbor_idx) = neighbor_idx {
                    let neighbor = vertex_neighbors[neighbor_idx];
                    vertex_neighbors.remove(neighbor_idx);
                    Ok(neighbor)
                } else {
                    Err(ERROR_MSG)
                }
            }
        };

        // Given a current vertex and the previous vertex, we find the next vertex by
        // picking the neighbor of "current" which is not equal to the previous.
        // In order to start this sequence, we must first choose an arbitrary "next" vertex
        // out of the two neighbors of "prev"
        let mut vertices = Vec::with_capacity(num_vertices);
        let mut prev_vertex_index = smallest_index;
        let mut current_vertex_index = take_next(prev_vertex_index, std::usize::MAX)?;
        vertices.push(self.vertices()[prev_vertex_index]);

        while current_vertex_index != smallest_index {
            let next_vertex_index = take_next(current_vertex_index, prev_vertex_index)?;
            prev_vertex_index = current_vertex_index;
            current_vertex_index = next_vertex_index;
            vertices.push(self.vertices()[prev_vertex_index]);
        }

        // TODO: What if we have a hole in the polygon? Should eventually also support this,
        // but for the moment we are limited to simple polygons.
        let mut polygon = GeneralPolygon::from_vertices(vertices);
        polygon.orient(Counterclockwise);

        Ok(polygon)
    }
}

impl<T, D, C> Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    C::FaceConnectivity: Connectivity + ConnectivityMut,
    DefaultAllocator: Allocator<T, D>,
{
    /// Creates a mesh that consists of all unique faces of this mesh.
    /// Face normals are only preserved for boundary faces.
    pub fn extract_face_soup(&self) -> Mesh<T, D, C::FaceConnectivity> {
        let mut unique_connectivity = HashMap::new();
        let mut faces = Vec::new();

        for cell_conn in self.connectivity.iter() {
            let num_faces = cell_conn.num_faces();
            for i in 0..num_faces {
                let face_conn = cell_conn.get_face_connectivity(i).unwrap();

                let mut vertex_indices = face_conn.vertex_indices().to_vec();
                vertex_indices.sort_unstable();

                if let HashMapEntry::Vacant(entry) = unique_connectivity.entry(vertex_indices) {
                    entry.insert(faces.len());
                    faces.push(face_conn);
                }
            }
        }

        let new_mesh = Mesh::from_vertices_and_connectivity(self.vertices.clone(), faces);
        let cells_to_keep: Vec<_> = (0..new_mesh.connectivity().len()).collect();
        // Remove unconnected vertices
        new_mesh.keep_cells(&cells_to_keep)
    }
}

impl<T, D, C> Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    C::FaceConnectivity: ConnectivityMut,
    DefaultAllocator: Allocator<T, D>,
{
    /// Constructs a new mesh from the surface cells of the mesh.
    ///
    /// The orientation of the faces are preserved.
    pub fn extract_surface_mesh(&self) -> Mesh<T, D, C::FaceConnectivity> {
        let connectivity = self
            .find_boundary_faces()
            .into_iter()
            .map(|(face, _, _)| face)
            .collect();

        // TODO: This is rather inefficient
        let new_mesh = Mesh::from_vertices_and_connectivity(self.vertices.clone(), connectivity);
        let cells_to_keep: Vec<_> = (0..new_mesh.connectivity().len()).collect();
        new_mesh.keep_cells(&cells_to_keep)
    }
}

impl<'a, T, D, C> GeometryCollection<'a> for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: CellConnectivity<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    type Geometry = C::Cell;

    fn num_geometries(&self) -> usize {
        self.connectivity.len()
    }

    fn get_geometry(&'a self, index: usize) -> Option<Self::Geometry> {
        self.connectivity()
            .get(index)
            .map(|conn| conn.cell(self.vertices()).unwrap())
    }
}

impl<'a, T, D, C, QueryGeometry> DistanceQuery<'a, QueryGeometry> for Mesh<T, D, C>
where
    T: RealField,
    D: DimName,
    C: CellConnectivity<T, D>,
    C::Cell: Distance<T, QueryGeometry>,
    DefaultAllocator: Allocator<T, D>,
{
    fn nearest(&'a self, query_geometry: &QueryGeometry) -> Option<usize> {
        let (_, min_index) = (0..self.num_geometries())
            .map(|idx| {
                let geometry = self
                    .get_geometry(idx)
                    .expect("num_geometries must report the correct number of available geometries");
                (idx, geometry)
            })
            .fold(
                (T::max_value(), None),
                |(mut min_dist, mut min_index), (idx, geometry)| {
                    let dist = geometry.distance(query_geometry);
                    // TODO: Square distance?
                    if dist < min_dist {
                        min_index = Some(idx);
                        min_dist = dist;
                    }
                    (min_dist, min_index)
                },
            );
        min_index
    }
}

pub trait PlanarFace<T>
where
    T: Scalar,
    DefaultAllocator: Allocator<T, Self::Dimension>,
{
    type Dimension: DimName;

    fn normal(&self) -> VectorN<T, Self::Dimension>;
}

impl<T> PlanarFace<T> for LineSegment2d<T>
where
    T: RealField,
{
    type Dimension = U2;

    fn normal(&self) -> Vector2<T> {
        self.normal_dir().normalize()
    }
}

/// A closed surface mesh with a well-defined inside and outside.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ClosedSurfaceMesh2d<T, Connectivity>
where
    T: Scalar,
{
    mesh: Mesh2d<T, Connectivity>,
    vertex_normals: Vec<Vector2<T>>,
}

impl<T, Connectivity> ClosedSurfaceMesh2d<T, Connectivity>
where
    T: Scalar,
{
    pub fn mesh(&self) -> &Mesh2d<T, Connectivity> {
        &self.mesh
    }
}

impl<T, Connectivity> ClosedSurfaceMesh2d<T, Connectivity>
where
    T: RealField,
    Connectivity: CellConnectivity<T, U2>,
    Connectivity::Cell: PlanarFace<T, Dimension = U2>,
    DefaultAllocator: Allocator<T, <Connectivity::Cell as PlanarFace<T>>::Dimension>,
{
    /// Transform all vertices of the mesh by the given transformation function.
    pub fn transform_vertices<F>(&mut self, mut transformation: F)
    where
        F: FnMut(&mut [Point2<T>]),
    {
        transformation(self.mesh.vertices_mut());
        self.recompute_vertex_normals();
    }

    pub fn from_mesh(mesh: Mesh2d<T, Connectivity>) -> Result<Self, Box<dyn Error>> {
        let num_vertices = mesh.vertices().len();
        let mut closed_mesh = Self {
            mesh,
            vertex_normals: vec![Vector2::zeros(); num_vertices],
        };

        closed_mesh.recompute_vertex_normals();

        Ok(closed_mesh)
    }

    fn recompute_vertex_normals(&mut self) {
        for v in &mut self.vertex_normals {
            *v = Vector2::zeros();
        }

        let mesh = &self.mesh;

        for conn in mesh.connectivity() {
            let cell = conn.cell(mesh.vertices()).unwrap();
            let n = cell.normal();
            for v_idx in conn.vertex_indices() {
                self.vertex_normals[*v_idx] += &n;
            }
        }

        for v in &mut self.vertex_normals {
            let v_norm = v.norm();
            if v_norm > T::zero() {
                *v /= v_norm;
            } else {
                // TODO: How to handle this situation?
                *v = Vector2::zeros();
            }
        }
    }
}

impl<'a, T, C> GeometryCollection<'a> for ClosedSurfaceMesh2d<T, C>
where
    T: Scalar,
    Mesh2d<T, C>: GeometryCollection<'a>,
{
    type Geometry = <Mesh2d<T, C> as GeometryCollection<'a>>::Geometry;

    fn num_geometries(&self) -> usize {
        self.mesh.num_geometries()
    }

    fn get_geometry(&'a self, index: usize) -> Option<Self::Geometry> {
        self.mesh.get_geometry(index)
    }
}

impl<T> SignedDistance<T, U2> for ClosedSurfaceMesh2d<T, Segment2d2Connectivity>
where
    T: RealField,
{
    fn query_signed_distance(&self, point: &Point2<T>) -> Option<SignedDistanceResult<T, U2>> {
        let closest_edge = self.closest_edge(point)?;

        Some(SignedDistanceResult {
            feature_id: closest_edge.edge_index,
            point: closest_edge.edge_point,
            signed_distance: closest_edge.signed_distance,
        })
    }
}

impl<T> Polygon<T> for ClosedSurfaceMesh2d<T, Segment2d2Connectivity>
where
    T: RealField,
{
    fn vertices(&self) -> &[Point<T, U2>] {
        self.mesh().vertices()
    }

    fn num_edges(&self) -> usize {
        self.mesh().connectivity().len()
    }

    fn get_edge(&self, index: usize) -> Option<LineSegment2d<T>> {
        self.mesh().get_cell(index)
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn pseudonormal_on_edge(&self, edge_index: usize, t: T) -> Option<Vector2<T>> {
        let edge_conn = self.mesh().connectivity().get(edge_index)?;
        let edge = edge_conn.cell(self.mesh().vertices())?;

        let [a_idx, b_idx] = edge_conn.0;
        // If parameter suggests that the point is on a vertex, then use the vertex normal instead
        let pseudo_normal = if t <= T::zero() {
            self.vertex_normals[a_idx]
        } else if t >= T::one() {
            self.vertex_normals[b_idx]
        } else {
            edge.normal_dir().normalize()
        };
        Some(pseudo_normal)
    }
}
