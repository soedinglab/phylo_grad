use std::error::Error;

use crate::data_types::*;
use crate::io::*;
use crate::tree::*;

impl FelsensteinError {
    const TOO_MANY_CHILDREN: Self =
        Self::DeserializationError("Incorrect format: a non-root node has more than two children");
    const CYCLES: Self = Self::DeserializationError("There are cycles in the input data");
}

/// Weak preprocessing: without level batching.
///     Assign height to each node.
///     Sort range(id) with `|id| height(node[id])` as key
///     Remap indices
///     Add children indices and create Vec<TreeNode>
pub fn preprocess_weak<F: FloatTrait>(
    parents: Vec<i32>,
    num_leaves: u32,
) -> Result<(Vec<TreeNode>, Vec<u32>), Box<dyn Error>> {
    let num_nodes = parents.len();

    let mut height = vec![0_u32; num_nodes];

    // Assign min height to each node. Worst case complexity: O(n^2), but more like N*log(N)
    for i in 0..num_leaves {
        let mut node = i as usize;
        let mut h = 0;
        while parents[node] >= 0 {
            h += 1;
            node = parents[node] as usize;
            height[node] = height[node].min(h);
        }
    }

    let mut indices = (0..num_nodes as u32).collect::<Vec<u32>>();
    indices.sort_by_key(|&id| height[id as usize]);

    let mut new_parents = vec![TreeNode{parent : 0, left : None, right: None}; num_nodes];
    let mut childs: Vec<Vec<u32>> = vec![vec![]; num_nodes];
    let mut root_id = 0;
    for (new_id, &old_id) in indices.iter().enumerate() {
        if parents[old_id as usize] < 0 {
            root_id = new_id;
        } else {
            let parent_id = parents[old_id as usize] as u32;
            new_parents[new_id].parent = parent_id;
            childs[parent_id as usize].push(new_id as u32);
        }
    }

    

    todo!()

}
