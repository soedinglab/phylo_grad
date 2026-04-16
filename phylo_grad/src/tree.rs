use std::collections::HashMap;

/// Represents a tree
/// The nodes are numbered from 0 to n-1, where n is the number of nodes.
/// We store the parent of each node, the root node has parent -1.
/// The leaf nodes are the first `num_leaves` nodes in the tree.
/// The all the nodes have to be in topological order, i.e. the parent of a node is always after the node itself in the slice.
/// This means the root node is always the last node in the slice.
#[derive(Debug, Clone)]
pub struct Tree<'a> {
    pub parents: &'a [i32],
    pub distances: &'a [f64],
    pub num_leaves: usize,
}

impl<'a> Tree<'a> {
    pub fn new(parents: &'a [i32], distances: &'a [f64], num_leaves: usize) -> Self {
        Tree {
            parents,
            distances,
            num_leaves,
        }
    }
}

/// middle is -1 for bifurcations, only at the root node it is one of the children.
#[derive(Debug, Clone)]
pub struct Bifurcation {
    pub left: i32,
    pub right: i32,
    pub middle: i32,
    pub parent: i32,
}

pub fn get_topological_bifurcations(tree: &Tree) -> Vec<Bifurcation> {
    let mut childs = vec![Vec::new(); tree.parents.len()];
    for (child, &parent) in tree.parents.iter().enumerate() {
        if parent >= 0 {
            childs[parent as usize].push(child as i32);
        }
    }

    let mut bifurcations = Vec::new();

    for node in tree.num_leaves..tree.parents.len() {
        let children = &childs[node];
        assert!(children.len() > 0, "Interior Node {} has no children", node);
        
        if children.len() == 2 {
            // bifurcation case
            let left = children[0];
            let right = children[1];
            bifurcations.push(Bifurcation {
                left,
                right,
                parent : node as i32,
                middle: -1,
            });
        } else if children.len() == 3 {
            // trifurcation case, only at the root node
            assert!(node == tree.parents.len() - 1, "Only the root node can have three children, but node {} has three children", node);
            let left = children[0];
            let right = children[1];
            let middle = children[2];
            bifurcations.push(Bifurcation {
                left,
                right,
                middle,
                parent : node as i32,
            });
        } else if children.len() > 3 {
            panic!("Node {} has {} children, but only bifurcations with 2 or 3 children are supported", node, children.len());
        } else if children.len() == 1 {
            panic!("Node {} has only one child, but only bifurcations with 2 or 3 children are supported", node);
        }
    }


    bifurcations
}

///
/// Returns the new parents, new distances, number of leaves and the number of branches below
/// It also return the sorting order. i[new_index] = original_index
pub fn topological_sort(
    parents: &[i32],
    distances: &[f64],
) -> (Vec<i32>, Vec<f64>, usize, Vec<usize>) {
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
        .collect::<Vec<f64>>();

    (new_parents, new_dist, num_leaves, indices)
}

fn dfs(node: usize, childs: &[Vec<usize>], heights: &mut [u32]) -> u32 {
    if childs[node].is_empty() {
        return 0;
    }
    let mut max_height = 0;
    for &child in &childs[node] {
        let child_height = dfs(child, childs, heights);
        max_height = max_height.max(child_height);
    }
    heights[node] = max_height + 1;
    max_height + 1
}
