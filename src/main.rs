extern crate ndarray;
extern crate num_enum;
extern crate tree_iterators_rs;

use ndarray::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::convert::TryFrom;
use tree_iterators_rs::prelude::*;

// use pyo3

/* TODO optimize numeric enum */
#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Residue {
    None = 0,
    A,
    C,
    G,
    T,
}

impl Residue {
    /* TODO: magic value */
    const TOTAL: usize = 5;
}

/* TODO #[repr(u8)] */
#[derive(Debug)]
struct ResiduePair(Residue, Residue);

type Entry = Residue;
type Rate = f32;

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

    let tot = Entry::TOTAL;

    let rate_matrix_example = Array::<Rate, _>::eye(tot)
        - (1f32 / (tot - 1) as f32)
            * (Array::<Rate, _>::ones((tot, tot)) - Array::<Rate, _>::eye(tot));
    dbg!(rate_matrix_example);

    let result = root
        .dfs_postorder_iter()
        .map(|val| std::fmt::format(format_args!("{:?}", val)))
        .collect::<Vec<String>>()
        .join(", ");

    println!("{result}");

    /* println!("Total: {}", Residue::TOTAL);
    let test_res = Residue::G;
    let test_num = u8::from(test_res);
    let test_num_2 = 2u8;
    let test_res_2 = Residue::try_from(test_num_2).unwrap();
    println!("Entry has size {}, value {:?}", std::mem::size_of_val(&test_res), test_num);
    println!{"test_res_2 = {:?}", test_res_2}; */
}
