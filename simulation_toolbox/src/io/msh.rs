use fenris::connectivity::{Connectivity, Tet4Connectivity, Tri3d2Connectivity, Tri3d3Connectivity};
use fenris::mesh::Mesh;
use fenris::nalgebra::{allocator::Allocator, DefaultAllocator, DimName, Point, Point2, Point3, RealField, U2, U3};
use log::warn;
use mshio::mshfile::{ElementType, MshFile};
use std::convert::TryInto;
use std::error::Error;

pub trait TryFromMshNodeBlock<T, D, F, I>
where
    Self: Sized,
    T: RealField,
    D: DimName,
    F: mshio::MshFloatT,
    I: mshio::MshIntT,
    Point<T, D>: TryVertexFromMshNode<T, D, F>,
    DefaultAllocator: Allocator<T, D>,
{
    fn try_from_node_block(node_block: &mshio::NodeBlock<u64, I, F>) -> Result<Vec<Point<T, D>>, Box<dyn Error>> {
        // Ensure that node tags are consecutive
        if node_block.node_tags.is_some() {
            return Err(Box::from("Node block tags are not consecutive"));
        } else {
            if node_block
                .entity_dim
                .to_usize()
                .ok_or_else(|| "Error converting node block entity dimension to usize")?
                != D::dim()
            {
                warn!("Warning: Node block entity does not have the right dimension for this mesh. Will be read as if they were of the same dimension.");
                /*
                return Err(Box::from(
                    "Node block entity does not have the right dimension for this mesh",
                ));
                */
            }

            let mut vertices = Vec::with_capacity(node_block.nodes.len());

            // Convert MSH vertices to points
            for node in &node_block.nodes {
                vertices.push(Point::try_vertex_from_msh_node(node)?);
            }

            Ok(vertices)
        }
    }
}

impl<T, D, F, I> TryFromMshNodeBlock<T, D, F, I> for Point<T, D>
where
    T: RealField,
    D: DimName,
    F: mshio::MshFloatT,
    I: mshio::MshIntT,
    Point<T, D>: TryVertexFromMshNode<T, D, F>,
    DefaultAllocator: Allocator<T, D>,
{
}

pub trait TryVertexFromMshNode<T, D, F>
where
    Self: Sized,
    T: RealField,
    D: DimName,
    F: mshio::MshFloatT,
    DefaultAllocator: Allocator<T, D>,
{
    fn try_vertex_from_msh_node(node: &mshio::Node<F>) -> Result<Point<T, D>, Box<dyn Error>>;
}

macro_rules! f_to_t {
    ($component:expr) => {
        T::from_f64(
            $component
                .to_f64()
                .ok_or_else(|| "Error converting node coordinate to f64")?,
        )
        .ok_or_else(|| "Error converting node coordinate to mesh data type")?
    };
}

impl<T, F> TryVertexFromMshNode<T, U2, F> for Point2<T>
where
    T: RealField,
    F: mshio::MshFloatT,
{
    fn try_vertex_from_msh_node(node: &mshio::Node<F>) -> Result<Self, Box<dyn Error>> {
        // TODO: Ensure that node.z is zero?
        Ok(Self::new(f_to_t!(node.x), f_to_t!(node.y)))
    }
}

impl<T, F> TryVertexFromMshNode<T, U3, F> for Point3<T>
where
    T: RealField,
    F: mshio::MshFloatT,
{
    fn try_vertex_from_msh_node(node: &mshio::Node<F>) -> Result<Self, Box<dyn Error>> {
        Ok(Self::new(f_to_t!(node.x), f_to_t!(node.y), f_to_t!(node.z)))
    }
}

pub trait TryFromMshElementBlock<C, I>
where
    Self: Sized,
    C: Connectivity + TryConnectivityFromMshElement<C>,
    I: mshio::MshIntT,
{
    fn try_from_element_block(element_block: &mshio::ElementBlock<u64, I>) -> Result<Vec<C>, Box<dyn Error>> {
        let requested_msh_element_type = C::msh_element_type();
        if element_block.element_type != requested_msh_element_type {
            warn!("Warning: Detected connectivity in the MSH file that does not match the requested connectivity. It will be ignored.");
            return Ok(Vec::new());
        /*
        return Err(Box::from(
            "Connectivity in the MSH file does not match the requested connectivity.",
        ));
        */
        } else {
            let mut connectivity = Vec::with_capacity(element_block.elements.len());
            let requested_nodes = requested_msh_element_type
                .nodes()
                .map_err(|_| "Unimplemented element type requested")?;

            for element in &element_block.elements {
                if element.nodes.len() < requested_nodes {
                    return Err(Box::from("Not enough nodes to initialize connectivity."));
                }
                connectivity.push(C::try_connectivity_from_msh_element(element)?);
            }

            Ok(connectivity)
        }
    }
}

impl<C, I> TryFromMshElementBlock<C, I> for C
where
    C: Connectivity + TryConnectivityFromMshElement<C>,
    I: mshio::MshIntT,
{
}

pub trait TryConnectivityFromMshElement<C>
where
    Self: Sized,
    C: Connectivity,
{
    fn msh_element_type() -> ElementType;

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> Result<C, Box<dyn Error>>;
}

impl TryConnectivityFromMshElement<Tri3d2Connectivity> for Tri3d2Connectivity {
    fn msh_element_type() -> ElementType {
        ElementType::Tri3
    }

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> Result<Self, Box<dyn Error>> {
        Ok(Self([
            (element.nodes[0] - 1).try_into().unwrap(),
            (element.nodes[1] - 1).try_into().unwrap(),
            (element.nodes[2] - 1).try_into().unwrap(),
        ]))
    }
}

impl TryConnectivityFromMshElement<Tri3d3Connectivity> for Tri3d3Connectivity {
    fn msh_element_type() -> ElementType {
        ElementType::Tri3
    }

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> Result<Self, Box<dyn Error>> {
        Ok(Self([
            (element.nodes[0] - 1).try_into().unwrap(),
            (element.nodes[1] - 1).try_into().unwrap(),
            (element.nodes[2] - 1).try_into().unwrap(),
        ]))
    }
}

impl TryConnectivityFromMshElement<Tet4Connectivity> for Tet4Connectivity {
    fn msh_element_type() -> ElementType {
        ElementType::Tet4
    }

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> Result<Self, Box<dyn Error>> {
        Ok(Self([
            (element.nodes[0] - 1).try_into().unwrap(),
            (element.nodes[1] - 1).try_into().unwrap(),
            (element.nodes[2] - 1).try_into().unwrap(),
            (element.nodes[3] - 1).try_into().unwrap(),
        ]))
    }
}

/// Tries to create a Fenris mesh from bytes representing a MSH file
pub fn try_mesh_from_bytes<T, D, C>(msh_bytes: &[u8]) -> Result<Mesh<T, D, C>, Box<dyn std::error::Error>>
where
    T: RealField,
    D: DimName,
    C: Connectivity + TryConnectivityFromMshElement<C>,
    Point<T, D>: TryVertexFromMshNode<T, D, f64>,
    DefaultAllocator: Allocator<T, D>,
{
    let msh = match mshio::parse_msh_bytes(msh_bytes) {
        Ok(msh) => msh,
        Err(e) => return Err(Box::from(format!("Error during MSH parsing ({})", e))),
    };

    try_mesh_from_msh_file(msh)
}

/// Creates a Mesh from a MshFile
pub fn try_mesh_from_msh_file<T, D, C>(msh: MshFile<u64, i32, f64>) -> Result<Mesh<T, D, C>, Box<dyn std::error::Error>>
where
    T: RealField,
    D: DimName,
    C: Connectivity + TryConnectivityFromMshElement<C>,
    Point<T, D>: TryVertexFromMshNode<T, D, f64>,
    DefaultAllocator: Allocator<T, D>,
{
    let mut msh = msh;
    let msh_nodes = msh
        .data
        .nodes
        .take()
        .ok_or("MSH file does not contain nodes")?;
    let msh_elements = msh
        .data
        .elements
        .take()
        .ok_or("MSH file does not contain elements")?;

    let mut vertices = Vec::new();
    let mut connectivity = Vec::new();

    for node_block in &msh_nodes.node_blocks {
        // Ensure that node tags are consecutive
        if node_block.node_tags.is_some() {
            return Err(Box::from("Node block tags are not consecutive"));
        }

        let block_vertices = Point::try_from_node_block(node_block)?;
        vertices.extend(block_vertices);
    }

    for element_block in &msh_elements.element_blocks {
        // Ensure that element tags are consecutive
        if element_block.element_tags.is_some() {
            return Err(Box::from("Element block tags are not consecutive"));
        }

        let block_connectivity = C::try_from_element_block(element_block)?;
        connectivity.extend(block_connectivity);
    }

    Ok(Mesh::from_vertices_and_connectivity(vertices, connectivity))
}
