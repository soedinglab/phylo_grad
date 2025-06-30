use std::collections::HashMap;

use crate::data_types::*;
use crate::tree::*;

#[derive(Debug)]
pub enum TreeError {
    NotBinary,
    RootNotThreeChildren,
}

/// Weak preprocessing: without level batching.
///     Assign height to each node.
///     Sort range(id) with `|id| height(node[id])` as key
///     Remap indices
///     Add children indices and create Vec<TreeNode>
pub fn topological_preprocess<F: FloatTrait>(
    parents: Vec<i32>,
    dist: Vec<F>,
    num_leaves: u32,
) -> Result<(Vec<TreeNode>, Vec<F>), TreeError> {
    let num_nodes = parents.len();

    let mut height = vec![0u32; num_nodes];

    // Assign min height to each node. Worst case complexity: O(n^2), but more like N*log(N)
    for i in 0..num_leaves {
        let mut node = i as usize;
        let mut h = 0;
        while parents[node] >= 0 {
            h += 1;
            node = parents[node] as usize;
            height[node] = height[node].max(h);
        }
    }

    // Sort the internal nodes by height, make sure the sort is stable in the leaf nodes
    let mut indices = (num_leaves..num_nodes as u32).collect::<Vec<u32>>();
    indices.sort_by_key(|&id| height[id as usize]);
    let indices = (0..num_leaves).chain(indices).collect::<Vec<u32>>();
    let rev_mapping = indices
        .iter()
        .enumerate()
        .map(|(i, &x)| (x, i as u32))
        .collect::<HashMap<u32, u32>>();

    // Change parents ids
    let new_parents = parents
        .iter()
        .map(|&x| {
            if x == -1 {
                -1
            } else {
                rev_mapping[&(x as u32)] as i32
            }
        })
        .collect::<Vec<i32>>();
    // Permute parents
    let new_parents = indices
        .iter()
        .map(|&x| new_parents[x as usize])
        .collect::<Vec<i32>>();

    let new_dist = indices
        .iter()
        .map(|&x| dist[x as usize])
        .collect::<Vec<F>>();

    drop(parents);

    let mut tree_nodes = vec![
        TreeNode {
            parent: 0,
            left: None,
            right: None
        };
        num_nodes
    ];
    let mut childs: Vec<Vec<u32>> = vec![vec![]; num_nodes];

    let mut root_id = 0;
    for i in 0..num_nodes {
        if new_parents[i] < 0 {
            root_id = i;
        } else {
            let parent = new_parents[i];
            tree_nodes[i].parent = parent as u32;
            childs[parent as usize].push(i as u32);
        }
    }

    assert!(root_id == num_nodes - 1);

    for (idx, children) in childs.into_iter().enumerate() {
        if idx == root_id {
            if children.len() == 3 {
                tree_nodes[idx].left = Some(children[0]);
                tree_nodes[idx].right = Some(children[1]);
                // We store the third child in the parent field
                tree_nodes[idx].parent = children[2];
            } else {
                return Err(TreeError::RootNotThreeChildren);
            }
        } else {
            #[allow(clippy::collapsible_else_if)]
            if children.len() == 2 {
                tree_nodes[idx].left = Some(children[0]);
                tree_nodes[idx].right = Some(children[1]);
            } else if children.is_empty() {
                // Leaf node
            } else {
                return Err(TreeError::NotBinary);
            }
        }
    }

    Ok((tree_nodes, new_dist))
}
