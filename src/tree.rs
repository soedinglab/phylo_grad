use crate::data_types::*;
use crate::tree_iterators_rs::prelude::*;

impl<const TOTAL: usize> FelsensteinNodeStd<TOTAL> {
    fn trivial() -> Self {
        FelsensteinNodeStd {
            log_p: [1 as Float; TOTAL],
            distance: 1 as Float,
        }
    }
}

#[derive(Debug)]
struct TreeFixed<Data, const TREE_SIZE: usize> {
    data: [Option<Data>; TREE_SIZE],
}

/* Implement as a trait to have common interface? */
type Id = usize;
impl<const TREE_SIZE: Id, const TOTAL: usize> TreeFixed<FelsensteinNodeStd<TOTAL>, TREE_SIZE> {
    fn left(self, id: Id) -> Option<Id> {
        let left_id = id * 2 + 1;
        if left_id >= TREE_SIZE {
            return None;
        }
        if self.data[id].is_none() {
            return None;
        } // (!) Check that the tree element is non-empty
        Some(id)
    }
    fn right(id: Id) -> Option<Id> {
        let right_id = id * 2 + 2;
        if right_id < TREE_SIZE {
            Some(right_id)
        } else {
            None
        }
    }
    fn trivial() -> Self {
        let data = [Some(FelsensteinNodeStd::<TOTAL>::trivial()); TREE_SIZE];
        Self { data }
    }
}

pub fn test_tree() {
    const TOTAL: usize = Entry::TOTAL;
    const LEAVES: usize = 4;
    const SIZE: usize = 2 * LEAVES - 1;
    let tree: TreeFixed<FelsensteinNodeStd<TOTAL>, SIZE> =
        TreeFixed::<FelsensteinNodeStd<TOTAL>, SIZE>::trivial();
    dbg!(tree.data);
}
/*
impl<'a, T> MutBorrowedTreeNode<'a> for tree_fixed<T> where Self: 'a {
    type MutBorrowedValue = &'a mut tree_fixed<T>;
    type MutBorrowedChildren
}
*/

pub fn create_example_binary_tree<const TOTAL: usize>() -> BinaryTreeNode<FelsensteinNode> {
    BinaryTreeNode {
        value: FelsensteinNode::from(Residue::A),
        left: Some(Box::new(BinaryTreeNode {
            value: FelsensteinNode::from(Residue::C),
            left: Some(Box::new(BinaryTreeNode {
                value: FelsensteinNode::from(Residue::G),
                left: None,
                right: None,
            })),
            right: Some(Box::new(BinaryTreeNode {
                value: FelsensteinNode::from(Residue::G),
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(BinaryTreeNode {
            value: FelsensteinNode::from(Residue::C),
            left: Some(Box::new(BinaryTreeNode {
                value: FelsensteinNode::from(Residue::G),
                left: None,
                right: None,
            })),
            right: Some(Box::new(BinaryTreeNode {
                value: FelsensteinNode::from(Residue::G),
                left: Some(Box::new(BinaryTreeNode {
                    value: FelsensteinNode::from(Residue::T),
                    left: None,
                    right: Some(Box::new(BinaryTreeNode {
                        value: FelsensteinNode::from(Residue::None),
                        left: Some(Box::new(BinaryTreeNode {
                            value: FelsensteinNode::from(Residue::A),
                            left: None,
                            right: Some(Box::new(BinaryTreeNode {
                                value: FelsensteinNode::from(Residue::C),
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
