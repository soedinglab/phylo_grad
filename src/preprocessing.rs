use itertools::process_results;

use std::error::Error;

use crate::data_types::*;
use crate::io::*;
use crate::tree::*;

impl FelsensteinError {
    const SEQ_LENGTH: Self =
        Self::DeserializationError("Tree leaves contain sequences of different lengths");
    const TOO_MANY_CHILDREN: Self =
        Self::DeserializationError("Incorrect format: a non-root node has more than two children");
    const CYCLES: Self = Self::DeserializationError("There are cycles in the input data");
}

pub fn try_residue_sequences_from_strings<Residue>(
    sequences_raw: &[String],
) -> Result<na::DMatrix<Residue>, FelsensteinError>
where
    Residue: ResidueTrait,
{
    let num_leaves = sequences_raw.len();

    let seq_length = sequences_raw[0].len();

    let sequences_flat = process_results(
        sequences_raw.iter().map(|s| {
            if s.len() == seq_length {
                Ok(s)
            } else {
                Err(FelsensteinError::SEQ_LENGTH)
            }
        }),
        |iter| {
            iter.flat_map(|s| Residue::try_deserialize_string_iter(s))
                .collect::<Result<Vec<Residue>, FelsensteinError>>()
        },
    )??;

    let sequences_2d = na::DMatrix::from_vec_storage(na::VecStorage::new(
        na::Dyn(seq_length),
        na::Dyn(num_leaves),
        sequences_flat,
    ))
    .transpose();
    Ok(sequences_2d)
}

/// Weak preprocessing: without level batching.
///     Assign height to each node.
///     Sort range(id) with `|id| height(node[id])` as key
///     Remap indices
///     Add children indices and create Vec<TreeNode>
pub fn preprocess_weak(
    raw_tree: &[RawInputRecord],
) -> Result<(Vec<TreeNode>, Vec<Float>, Vec<String>), Box<dyn Error>> {
    /* TODO linear algorithm via swap-remove and dynamic old_to_new index*/

    let num_nodes = raw_tree.len();

    let root_id = {
        let mut depth: usize = 0;
        let mut id: usize = 0;
        while raw_tree[id].parent >= 0 {
            id = raw_tree[id].parent as usize;
            depth += 1;
            if depth > num_nodes {
                return Err(Box::new(FelsensteinError::CYCLES));
            }
        }
        id
    };

    /* Initialization: initialize children counters with 0, then go once through the vector, incrementing the child counter for the parent of the current node.
    Iteration:
        Find all nodes with counter = 0
        Decrement the counters of their parents
        Pop the chosen nodes from the tree and add them to the resulting sequence. (*)
    For now, popping the children in the (*) step is replaced by setting children counter for the chosen nodes to -100.
    */

    /* Initialization */
    let mut num_children: Vec<i32> = vec![0_i32; num_nodes];

    for node in raw_tree.iter() {
        if node.parent >= 0 {
            num_children[node.parent as usize] += 1;
        }
    }

    let num_leaves = num_children.iter().filter(|num| **num == 0).count();

    /* Step */
    let mut ordered_ids = Vec::<usize>::with_capacity(num_nodes);
    while num_children[root_id] > 0 {
        let leaves: Vec<usize> = (0..num_nodes)
            .filter(|node_id| num_children[*node_id] == 0)
            .collect();
        for node_id in leaves.iter() {
            let parent_id = raw_tree[*node_id].parent as usize;
            num_children[parent_id] -= 1;
            num_children[*node_id] = -100;
        }
        ordered_ids.extend(leaves);
    }
    ordered_ids.push(root_id);

    /* Index map */
    let mut new_from_old = vec![1_000_000_usize; num_nodes];
    for (new_id, old_id) in ordered_ids.iter().enumerate() {
        new_from_old[*old_id] = new_id;
    }

    /* Children ids */

    let mut left = vec![None as Option<usize>; num_nodes];
    let mut right = vec![None as Option<usize>; num_nodes];
    let mut children_of_root = Vec::<usize>::with_capacity(3);
    /* Want: "left[new_id] = new_from_old[raw_tree[old_from_new[new_id]].left]",
    that is raw_tree[old_from_new[left[new_id]]].parent = old_from_new[new_id]
    */
    for (old_id, node) in raw_tree.iter().enumerate() {
        if node.parent == -1 {
            continue;
        }
        if node.parent as usize == root_id {
            children_of_root.push(new_from_old[old_id]);
            continue;
        }
        let new_id = new_from_old[old_id];
        let parent_old_id = node.parent as usize;
        let parent_new_id = new_from_old[parent_old_id];
        if left[parent_new_id].is_none() {
            left[parent_new_id] = Some(new_id);
        } else if right[parent_new_id].is_none() {
            right[parent_new_id] = Some(new_id);
        } else {
            dbg!(parent_old_id);
            dbg!(parent_new_id);
            return Err(Box::new(FelsensteinError::TOO_MANY_CHILDREN));
        }
    }

    /* Remapping */

    let mut tree = Vec::<TreeNode>::with_capacity(num_nodes);
    let mut distances = Vec::<Float>::with_capacity(num_nodes);
    let mut sequences = Vec::<String>::with_capacity(num_leaves);

    /* TODO deserialize sequences into a separate vector to avoid copy */
    for old_id in ordered_ids[0..num_leaves].iter() {
        let seq = raw_tree[*old_id].sequence.as_deref();
        sequences.push(String::from(seq.unwrap()));
    }

    for (new_id, old_id) in ordered_ids[..num_nodes - 1].iter().enumerate() {
        let distance = raw_tree[*old_id].distance.unwrap();
        distances.push(distance);

        let parent_old_id = raw_tree[*old_id].parent as usize;
        let parent_new_id = new_from_old[parent_old_id];
        tree.push(TreeNode {
            parent: parent_new_id,
            left: left[new_id],
            right: right[new_id],
        });
    }
    /* root */
    distances.push(Float::NEG_INFINITY);
    tree.push(TreeNodeId {
        parent: children_of_root[0],
        left: children_of_root.get(1).copied(),
        right: children_of_root.get(2).copied(),
    });

    Ok((tree, distances, sequences))
}
