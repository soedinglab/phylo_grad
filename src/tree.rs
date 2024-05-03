use crate::data_types::*;

/*  Tree root has three children. In our implementaion the root always has
fixed id (0 or -1), and its parent attribute points at the third child.
    The distance for the tree root will be set to -1.0.
*/

#[derive(Debug)]
pub struct TreeNodeId<Id> {
    pub parent: Id,
    pub left: Option<Id>,
    pub right: Option<Id>,
    pub distance: Float,
}

pub type TreeNode = TreeNodeId<Id>;
