extern crate ndarray;
extern crate tree_iterators_rs;

use ndarray::prelude::*;
use tree_iterators_rs::prelude::*;

// use pyo3

// TODO add #[repr(u8)], #[derive(FromPrimitive)]
#[derive(Debug)]
pub enum Residue {
    A,
    C,
    G,
    T,
    None,
}

// TODO replace struct with a single u8-sized enum, implement to and from whatever we keep in the ndarray
#[derive(Debug)]
struct ResiduePair(Residue, Residue);

pub fn create_example_binary_tree() -> BinaryTreeNode<Residue> {
    BinaryTreeNode {
        value: Residue::A,
        left: Some(Box::new(BinaryTreeNode {
            value: Residue::C,
            left: Some(Box::new(BinaryTreeNode {
                value: Residue::G,
                left: None,
                right: None,
            })),
            right: Some(Box::new(BinaryTreeNode {
                value: Residue::G,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(BinaryTreeNode {
            value: Residue::C,
            left: Some(Box::new(BinaryTreeNode {
                value: Residue::G,
                left: None,
                right: None,
            })),
            right: Some(Box::new(BinaryTreeNode {
                value: Residue::G,
                left: Some(Box::new(BinaryTreeNode {
                    value: Residue::T,
                    left: None,
                    right: Some(Box::new(BinaryTreeNode {
                        value: Residue::None,
                        left: Some(Box::new(BinaryTreeNode {
                            value: Residue::A,
                            left: None,
                            right: Some(Box::new(BinaryTreeNode {
                                value: Residue::C,
                                left: None,
                                right: None,
                            })),
                        })),
                        right: None,
                    })),
                })),
                right: None,
            })),
        })),
    }
} 
 pub fn main() {
    let root = create_example_binary_tree();

    let result = 
        root.dfs_postorder_iter()
            .map(|val| std::fmt::format(format_args!("{:?}", val)))
            .collect::<Vec<String>>()
            .join(", ");

    println!("{result}");
}