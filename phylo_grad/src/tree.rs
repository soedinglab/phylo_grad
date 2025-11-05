use std::collections::HashMap;

use crate::FloatTrait;

/// Represents a tree
/// The nodes are numbered from 0 to n-1, where n is the number of nodes.
/// We store the parent of each node, the root node has parent -1.
/// The leaf nodes are the first `num_leaves` nodes in the tree.
/// The all the nodes have to be in topological order, i.e. the parent of a node is always after the node itself in the slice.
/// This means the root node is always the last node in the slice.
#[derive(Debug, Clone)]
pub struct Tree<'a, F> {
    pub parents: &'a [i32],
    pub distances: &'a [F],
    pub num_leaves: usize,
}

impl<'a, F: FloatTrait> Tree<'a, F> {
    pub fn new(parents: &'a [i32], distances: &'a [F], num_leaves: usize) -> Self {
        Tree {
            parents,
            distances,
            num_leaves,
        }
    }
}

pub fn topological_sort<F: FloatTrait>(
    parents: &[i32],
    distances: &[F],
) -> (Vec<i32>, Vec<F>, usize) {
    // Leaves have height 0, the parents of leaves have height 1, the root will have the maximum height.
    let mut heights = vec![0; parents.len()];

    let mut childs = vec![Vec::new(); parents.len()];
    let mut root_id = 0;

    for (child, &parent) in parents.iter().enumerate() {
        if parent >= 0 {
            childs[parent as usize].push(child);
        } else {
            root_id = child;
        }
    }

    dfs(root_id, &childs, &mut heights);

    let num_leaves = heights.iter().filter(|&&h| h == 0).count();

    // Sort the nodes by height, such that the leaves come first
    let mut indices: Vec<usize> = (0..parents.len()).collect();
    indices.sort_by_key(|&i| heights[i]); // This sort is stable

    let rev_mapping = indices
        .iter()
        .enumerate()
        .map(|(i, &x)| (x as u32, i as u32))
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
        .map(|&x| distances[x as usize])
        .collect::<Vec<F>>();

    (new_parents, new_dist, num_leaves)
}

fn dfs(node: usize, childs: &[Vec<usize>], heights: &mut [u32]) -> u32 {
    if childs[node].is_empty() {
        return 0;
    }
    let mut max_height = 0;
    for &child in &childs[node] {
        max_height = max_height.max(dfs(child, childs, heights));
    }
    heights[node] = max_height + 1;
    max_height + 1
}
