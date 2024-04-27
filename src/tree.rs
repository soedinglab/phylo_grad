/* Tree data:
- a container with arbitrary access
- holding Felsenstein_data
    Implementation:
// Beware: ownership (need to be able to return both a mutable and an immutable child ref)
// Seems like we won't be able to implement the .left() method on the tree node,
as it requires the tree root, and we don't want to store it in the child. However, this is
a static dispatch problem: can we use the tree as a generic parameter of the node?
*/

/* Choose implementation based on how sparse the input tree is:
- if it is not sparse, use a fixed-size array.
- if it is moderately sparse, use a sparse vector.
- if it is very sparse, use pointers.
*/

/* TODO check how sparse are the tree inputs that Benjamin has */

/* TODO check the constraint on leaves count wrt type size */

use crate::data_types::*;

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
    data: [Data; TREE_SIZE],
}

/* Implement as a trait to have common interface? */
type Id = usize;
impl<const TREE_SIZE: Id, const TOTAL: usize> TreeFixed<FelsensteinNodeStd<TOTAL>, TREE_SIZE> {
    fn left(id: Id) -> Option<Id> {
        let left_id = id * 2 + 1;
        if left_id < TREE_SIZE {
            return Some(left_id);
        } else {
            return None;
        }
    }
    fn right(id: Id) -> Option<Id> {
        let right_id = id * 2 + 2;
        if right_id < TREE_SIZE {
            return Some(right_id);
        } else {
            return None;
        }
    }
    fn trivial() -> Self {
        let data = [FelsensteinNodeStd::<TOTAL>::trivial(); TREE_SIZE];
        Self { data: data }
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
