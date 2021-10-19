use crate::geometry::{ConvexPolygon, GeneralPolygon};
use crate::mesh::{ClosedSurfaceMesh2d, Mesh, Mesh2d, QuadMesh2d, TriangleMesh2d};
use nalgebra::{DefaultAllocator, DimMin, DimName, Point, RealField, Scalar};
use num::Zero;
use std::convert::TryInto;
use std::iter::repeat;
use vtkio::model::{Attribute, Attributes, CellType, Cells, DataSet, PolyDataTopology, Version, Vtk};
use vtkio::{export_be, Error};

use crate::allocators::ElementConnectivityAllocator;
use crate::connectivity::{
    Connectivity, Hex20Connectivity, Hex27Connectivity, Hex8Connectivity, Quad4d2Connectivity, Quad9d2Connectivity,
    Segment2d2Connectivity, Tet10Connectivity, Tet20Connectivity, Tet4Connectivity, Tri3d2Connectivity,
    Tri3d3Connectivity, Tri6d2Connectivity,
};
use crate::element::{ElementConnectivity, FiniteElement};
use crate::geometry::polymesh::PolyMesh;
use crate::quadrature::Quadrature;
use itertools::zip_eq;
use nalgebra::allocator::Allocator;
use std::fs::create_dir_all;
use std::path::Path;

/// Represents connectivity that is supported by VTK.
pub trait VtkCellConnectivity: Connectivity {
    fn num_nodes(&self) -> usize {
        self.vertex_indices().len()
    }

    fn cell_type(&self) -> vtkio::model::CellType;

    /// Write connectivity and return number of nodes.
    ///
    /// Panics if `connectivity.len() != self.num_nodes()`.
    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.vertex_indices().len());
        connectivity.clone_from_slice(self.vertex_indices());
    }
}

impl VtkCellConnectivity for Segment2d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Line
    }
}

impl VtkCellConnectivity for Tri3d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Triangle
    }
}

impl VtkCellConnectivity for Tri6d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticTriangle
    }
}

impl VtkCellConnectivity for Quad4d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Quad
    }
}

impl VtkCellConnectivity for Quad9d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticQuad
    }
}

impl VtkCellConnectivity for Tet4Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Tetra
    }
}

impl VtkCellConnectivity for Hex8Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Hexahedron
    }
}

impl VtkCellConnectivity for Tri3d3Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Triangle
    }
}

impl VtkCellConnectivity for Tet10Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticTetra
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.vertex_indices().len());
        connectivity.clone_from_slice(self.vertex_indices());

        // Gmsh ordering and ParaView have different conventions for quadratic tets,
        // so we must adjust for that. In particular, nodes 8 and 9 are switched
        connectivity.swap(8, 9);
    }
}

// Note: There is no Tet20 in ParaView (legacy anyway),
// so we only export it as a Tet4 element
impl VtkCellConnectivity for Tet20Connectivity {
    fn num_nodes(&self) -> usize {
        4
    }

    fn cell_type(&self) -> CellType {
        CellType::Tetra
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.num_nodes());
        connectivity.clone_from_slice(&self.vertex_indices()[0..4]);
    }
}

impl VtkCellConnectivity for Hex20Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticHexahedron
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.num_nodes());

        let v = self.vertex_indices();
        // The first 8 entries are the same
        connectivity[0..8].clone_from_slice(&v[0..8]);
        connectivity[8] = v[8];
        connectivity[9] = v[11];
        connectivity[10] = v[13];
        connectivity[11] = v[9];
        connectivity[12] = v[16];
        connectivity[13] = v[18];
        connectivity[14] = v[19];
        connectivity[15] = v[17];
        connectivity[16] = v[10];
        connectivity[17] = v[12];
        connectivity[18] = v[14];
        connectivity[19] = v[15];
    }
}

impl VtkCellConnectivity for Hex27Connectivity {
    fn num_nodes(&self) -> usize {
        20
    }

    // There is no tri-quadratic Hex in legacy VTK, so use Hex20 instead
    fn cell_type(&self) -> CellType {
        CellType::QuadraticHexahedron
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.num_nodes());

        let v = self.vertex_indices();
        // The first 8 entries are the same
        connectivity[0..8].clone_from_slice(&v[0..8]);
        connectivity[8] = v[8];
        connectivity[9] = v[11];
        connectivity[10] = v[13];
        connectivity[11] = v[9];
        connectivity[12] = v[16];
        connectivity[13] = v[18];
        connectivity[14] = v[19];
        connectivity[15] = v[17];
        connectivity[16] = v[10];
        connectivity[17] = v[12];
        connectivity[18] = v[14];
        connectivity[19] = v[15];
    }
}

impl<'a, T, D, C> From<&'a Mesh<T, D, C>> for DataSet
where
    T: Scalar + Zero,
    D: DimName,
    C: VtkCellConnectivity,
    DefaultAllocator: Allocator<T, D>,
{
    fn from(mesh: &'a Mesh<T, D, C>) -> Self {
        // TODO: Create a "SmallDim" trait or something for this case...?
        // Or just implement the trait directly for U1/U2/U3?
        assert!(D::dim() <= 3, "Unable to support dimensions larger than 3.");
        let points: Vec<_> = {
            let mut points: Vec<T> = Vec::new();
            for v in mesh.vertices() {
                points.extend_from_slice(v.coords.as_slice());

                for _ in v.coords.len()..3 {
                    points.push(T::zero());
                }
            }
            points
        };

        // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
        // so for quads this becomes 4 followed by the four indices making up the quad
        let mut vertices = Vec::new();
        let mut cell_types = Vec::new();
        let mut vertex_indices = Vec::new();
        for cell in mesh.connectivity() {
            // TODO: Return Result or something
            vertices.push(cell.num_nodes() as u32);

            vertex_indices.clear();
            vertex_indices.resize(cell.num_nodes(), 0);
            cell.write_vtk_connectivity(&mut vertex_indices);

            // TODO: Safer cast? How to handle this? TryFrom instead of From?
            vertices.extend(vertex_indices.iter().copied().map(|i| i as u32));
            cell_types.push(cell.cell_type());
        }

        DataSet::UnstructuredGrid {
            points: points.into(),
            cells: Cells {
                num_cells: mesh.connectivity().len() as u32,
                vertices,
            },
            cell_types,
            data: Attributes::new(),
        }
    }
}

impl<'a, T, D> From<&'a PolyMesh<T, D>> for DataSet
where
    T: Scalar + Zero,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn from(mesh: &'a PolyMesh<T, D>) -> Self {
        assert!(D::dim() == 2 || D::dim() == 3, "Only dimensions 2 and 3 supported.");

        let points: Vec<_> = {
            let mut points: Vec<T> = Vec::new();
            for v in mesh.vertices() {
                points.extend_from_slice(v.coords.as_slice());

                if D::dim() == 2 {
                    points.push(T::zero());
                }
            }
            points
        };

        // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
        // so for quads this becomes 4 followed by the four indices making up the quad
        let mut vertices = Vec::new();
        for face in mesh.face_connectivity_iter() {
            vertices.push(face.len() as u32);
            for idx in face {
                // TODO: Safer cast? How to handle this? TryFrom instead of From?
                vertices.push(*idx as u32);
            }
        }

        let cells = Cells {
            num_cells: mesh.num_faces() as u32,
            vertices,
        };

        DataSet::PolyData {
            points: points.into(),
            topo: vec![PolyDataTopology::Polygons(cells)],
            data: Attributes::new(),
        }
    }
}

impl<'a, T> From<&'a ClosedSurfaceMesh2d<T, Segment2d2Connectivity>> for DataSet
where
    T: Scalar + Zero,
{
    fn from(mesh: &'a ClosedSurfaceMesh2d<T, Segment2d2Connectivity>) -> Self {
        Self::from(mesh.mesh())
    }
}

impl<'a, T> From<&'a GeneralPolygon<T>> for DataSet
where
    T: RealField,
{
    fn from(polygon: &'a GeneralPolygon<T>) -> Self {
        let mut points = Vec::with_capacity(polygon.num_vertices() * 3);
        let mut cells = Cells {
            num_cells: polygon.num_edges() as u32,
            vertices: Vec::new(),
        };

        for v in polygon.vertices() {
            points.push(v.x);
            points.push(v.y);
            points.push(T::zero());
        }

        for i in 0..polygon.num_edges() {
            cells.vertices.push(2);
            // Edge points from vertex i to i + 1 (modulo)
            cells.vertices.push(i as u32);
            cells.vertices.push(((i + 1) % polygon.num_edges()) as u32);
        }

        DataSet::PolyData {
            points: points.into(),
            topo: vec![PolyDataTopology::Lines(cells)],
            data: Attributes::new(),
        }
    }
}

pub fn create_vtk_data_set_from_quadratures<T, C, D>(
    vertices: &[Point<T, D>],
    connectivity: &[C],
    quadrature_rules: impl IntoIterator<Item = impl Quadrature<T, C::ReferenceDim>>,
) -> DataSet
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    C: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    DefaultAllocator: Allocator<T, D> + ElementConnectivityAllocator<T, C>,
{
    let quadrature_rules = quadrature_rules.into_iter();

    // Quadrature weights and points mapped to physical domain
    let mut physical_weights = Vec::new();
    let mut physical_points = Vec::new();
    // Cell indices map each individual quadrature point to its original cell
    let mut cell_indices = Vec::new();

    for ((cell_idx, conn), quadrature) in zip_eq(connectivity.iter().enumerate(), quadrature_rules) {
        let element = conn.element(vertices).unwrap();
        for (w_ref, xi) in zip_eq(quadrature.weights(), quadrature.points()) {
            let j = element.reference_jacobian(xi);
            let x = element.map_reference_coords(xi);
            let w_physical = j.determinant().abs() * *w_ref;
            physical_points.push(Point::from(x));
            physical_weights.push(w_physical);
            cell_indices.push(cell_idx as u64);
        }
    }

    //    let (new_weights, new_points): (Vec<_>, Vec<_>) = connectivity
    //        .iter()
    //        .enumerate()
    //        .zip_eq(quadrature_rules)
    //        .flat_map(|((cell_idx, conn), quadrature)| {
    //            let element = conn.element(vertices).unwrap();
    //            let quadrature = zip_eq(quadrature.weights(), quadrature.points())
    //                .map(|(w_ref, xi)| {
    //                    let j = element.reference_jacobian(xi);
    //                    let x = element.map_reference_coords(xi);
    //                    let w_physical = j.determinant().abs() * *w_ref;
    //                    (w_physical, Point::from(x))
    //                })
    //                .collect::<Vec<_>>();
    //            quadrature
    //        })
    //        .unzip();

    let mut dataset = create_vtk_data_set_from_points(&physical_points);
    let weight_point_attributes = Attribute::Scalars {
        num_comp: 1,
        lookup_table: None,
        data: physical_weights.into(),
    };

    let cell_idx_point_attributes = Attribute::Scalars {
        num_comp: 1,
        lookup_table: None,
        data: cell_indices.into(),
    };

    match dataset {
        DataSet::PolyData { ref mut data, .. } => {
            data.point
                .push(("weight".to_string(), weight_point_attributes));
            data.point
                .push(("cell_index".to_string(), cell_idx_point_attributes));
        }
        _ => panic!("Unexpected enum variant from data set."),
    }

    dataset
}

/// TODO: Remove in favor of `From`
pub fn create_vtk_data_set_from_quad_mesh<T>(mesh: &QuadMesh2d<T>) -> DataSet
where
    T: Scalar + Zero,
{
    let points: Vec<_> = {
        let mut points: Vec<T> = Vec::new();
        for v in mesh.vertices() {
            points.extend_from_slice(v.coords.as_slice());
            points.push(T::zero());
        }
        points
    };

    // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
    // so for quads this becomes 4 followed by the four indices making up the quad
    let mut vertices = Vec::new();
    vertices.reserve(5 * mesh.connectivity().len());
    for cell in mesh.connectivity() {
        // TODO: Return Result or something
        vertices.push(4);
        vertices.push(cell[0].try_into().unwrap());
        vertices.push(cell[1].try_into().unwrap());
        vertices.push(cell[2].try_into().unwrap());
        vertices.push(cell[3].try_into().unwrap());
    }

    DataSet::UnstructuredGrid {
        points: points.into(),
        cells: Cells {
            num_cells: mesh.connectivity().len() as u32,
            vertices,
        },
        cell_types: repeat(CellType::Quad)
            .take(mesh.connectivity().len())
            .collect(),
        data: Attributes::new(),
    }
}

/// TODO: Remove in favor of `From`
pub fn create_vtk_data_set_from_triangle_mesh<T>(mesh: &TriangleMesh2d<T>) -> DataSet
where
    T: Scalar + Zero,
{
    let points: Vec<_> = {
        let mut points: Vec<T> = Vec::new();
        for v in mesh.vertices() {
            points.extend_from_slice(v.coords.as_slice());
            points.push(T::zero());
        }
        points
    };

    // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
    // so for triangles this becomes 3 followed by the three indices making up the triangle
    let mut vertices = Vec::new();
    vertices.reserve(4 * mesh.connectivity().len());
    for cell in mesh.connectivity() {
        vertices.push(3);
        vertices.push(cell[0].try_into().unwrap());
        vertices.push(cell[1].try_into().unwrap());
        vertices.push(cell[2].try_into().unwrap());
    }

    DataSet::UnstructuredGrid {
        points: points.into(),
        cells: Cells {
            num_cells: mesh.connectivity().len() as u32,
            vertices,
        },
        cell_types: repeat(CellType::Triangle)
            .take(mesh.connectivity().len())
            .collect(),
        data: Attributes::new(),
    }
}

pub fn create_vtk_data_set_from_polygons<T>(polygons: &[ConvexPolygon<T>]) -> DataSet
where
    T: Scalar + Zero,
{
    let mut points = Vec::new();
    let mut cells = Cells {
        num_cells: polygons.len() as u32,
        vertices: Vec::new(),
    };

    for polygon in polygons {
        let point_start = (points.len() / 3) as u32;
        let num_points = polygon.vertices().len() as u32;

        cells.vertices.push(num_points);

        for (i, vertex) in polygon.vertices().iter().enumerate() {
            points.push(vertex.x.clone());
            points.push(vertex.y.clone());
            points.push(T::zero());
            cells.vertices.push(point_start + i as u32);
        }
    }

    DataSet::PolyData {
        points: points.into(),
        topo: vec![PolyDataTopology::Polygons(cells)],
        data: Attributes::new(),
    }
}

pub fn create_vtk_data_set_from_points<T, D>(points: &[Point<T, D>]) -> DataSet
where
    T: Scalar + Zero,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    assert!(D::dim() <= 3, "Only support dimensions up to 3.");

    let mut vtk_points = Vec::new();
    let mut cells = Cells {
        num_cells: points.len() as u32,
        vertices: Vec::new(),
    };

    for (i, point) in points.iter().enumerate() {
        for j in 0..D::dim() {
            vtk_points.push(point.coords[j].clone());
        }

        for _ in D::dim()..3 {
            vtk_points.push(T::zero());
        }

        cells.vertices.push(1);
        cells.vertices.push(i as u32);
    }

    DataSet::PolyData {
        points: vtk_points.into(),
        topo: vec![PolyDataTopology::Vertices(cells)],
        data: Attributes::new(),
    }
}

/// Convenience method for easily writing polygons to VTK files
pub fn write_vtk_polygons<T>(polygons: &[ConvexPolygon<T>], filename: &str, title: &str) -> Result<(), Error>
where
    T: Scalar + Zero,
{
    let data = create_vtk_data_set_from_polygons(polygons);
    write_vtk(data, filename, title)
}

/// Convenience function for writing meshes to VTK files.
pub fn write_vtk_mesh<'a, T, Connectivity>(
    mesh: &'a Mesh2d<T, Connectivity>,
    filename: &str,
    title: &str,
) -> Result<(), Error>
where
    T: Scalar + Zero,
    &'a Mesh2d<T, Connectivity>: Into<DataSet>,
{
    let data = mesh.into();
    write_vtk(data, filename, title)
}

pub fn write_vtk<P: AsRef<Path>>(data: impl Into<DataSet>, filename: P, title: &str) -> Result<(), Error> {
    let vtk_file = Vtk {
        version: Version::new((4, 1)),
        title: title.to_string(),
        data: data.into(),
    };

    let filename = filename.as_ref();

    if let Some(dir) = filename.parent() {
        create_dir_all(dir)?;
    }
    export_be(vtk_file, filename)
}
