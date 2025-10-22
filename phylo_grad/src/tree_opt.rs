use lazy_static::lazy_static;
use nalgebra::SVector;
use nalgebra as na;
use num_traits::Float;
use stochastic_optimizers::Optimizer;
use core::f64;
use std::collections::HashMap;

use crate::{forward::{compute_param_data, diag_times_assign, times_diag_assign, ParamPrecomp}, FelsensteinTree, FloatTrait};

lazy_static! {
    static ref AMINO_MAPPING: HashMap<u8, u8> = {
        let mut m = HashMap::new();
        m.insert(b'A', 0);
        m.insert(b'R', 1);
        m.insert(b'N', 2);
        m.insert(b'D', 3);
        m.insert(b'C', 4);
        m.insert(b'Q', 5);
        m.insert(b'E', 6);
        m.insert(b'G', 7);
        m.insert(b'H', 8);
        m.insert(b'I', 9);
        m.insert(b'L', 10);
        m.insert(b'K', 11);
        m.insert(b'M', 12);
        m.insert(b'F', 13);
        m.insert(b'P', 14);
        m.insert(b'S', 15);
        m.insert(b'T', 16);
        m.insert(b'W', 17);
        m.insert(b'Y', 18);
        m.insert(b'V', 19);
        m.insert(b'-', 20);
        m.insert(b'X', 20);
        m
    };
}

pub fn seq2pll<F: FloatTrait>(seq: impl Iterator<Item = u8>) -> Vec<SVector<F, 20>> {
    seq
        .map(|c| *AMINO_MAPPING.get(&c).unwrap_or(&20))
        .map(|idx| {
            let mut v = SVector::<F, 20>::from_element(<F as FloatTrait>::from_f64(f64::NEG_INFINITY));
            if idx < 20 {
                v[idx as usize] = F::zero();
            } else {
                for i in 0..20 {
                    v[i] = F::zero();
                }
            }
            v
        })
        .collect()
}

pub fn optimize_tree(newick: &str, sequences: &HashMap<String, Vec<u8>>) -> String {
    let tree = phylotree::tree::Tree::from_newick(newick).unwrap();

    let L = sequences.values().next().unwrap().len();

    let level_order = tree.levelorder(&tree.get_root().unwrap()).unwrap();
    
    let mut parents = vec![i32::MAX; level_order.len()];
    let mut distances = vec![f64::NAN; level_order.len() - 1];

    // idx_mapping[tree_idx] = new_idx
    let mut idx_mapping = vec![i32::MAX; level_order.len()];

    let num_leaves = tree.n_leaves();

    let mut leaf_idx = 0usize;
    let mut internal_idx = level_order.len() - 1;

    for node_idx in &level_order {
        let node = tree.get(node_idx).unwrap();
        if node.is_tip() {
            idx_mapping[*node_idx] = leaf_idx as i32;
            parents[leaf_idx] = idx_mapping[node.parent.unwrap()];
            distances[leaf_idx] = node.parent_edge.unwrap();
            leaf_idx += 1;
        } else {
            idx_mapping[*node_idx] = internal_idx as i32;
            if let Some(parent_idx) = node.parent {
                parents[internal_idx] = idx_mapping[parent_idx];
                distances[internal_idx] = node.parent_edge.unwrap();
            } else {
                parents[internal_idx] = -1;
            }
            internal_idx -= 1;
        }
    }

    distances.iter_mut().for_each(|x| *x = x.ln());


    let mut felsenstein = FelsensteinTree::<f64, 20>::new_no_sort(&parents, num_leaves);


    let mut leaf_pll = vec![];

    for i in 0..L {
        let mut column_seq = vec![b'x'; num_leaves];
        
        for node_idx in level_order.iter() {
            let node = tree.get(&node_idx).unwrap();
            if node.is_tip() {
                let new_idx = idx_mapping[*node_idx];
                assert!(new_idx < num_leaves as i32);
                let seq = sequences.get(node.name.as_ref().unwrap()).unwrap();
                column_seq[new_idx as usize] = seq[i];
            }
        }
        leaf_pll.push(seq2pll::<f64>(column_seq.into_iter()));
    }

    felsenstein.bind_leaf_log_p(leaf_pll);

    let param = read_model_paml::<f64>("models/lg.paml");
    println!("Initial log likelihood: {}", felsenstein.calculate_branch_gradients(&distances, &param).1.iter().sum::<f64>());

    /*let evaluate = |x : &[f64], g: &mut [f64]| {
        let (gradients, ll) = felsenstein.calculate_branch_gradients(x, &param);
        for (i, grad) in gradients.iter().enumerate() {
            g[i] = -*grad;
        }
        Ok(-ll.iter().sum::<f64>())
    };

    let optimizer = gosh_lbfgs::lbfgs().with_max_iterations(50).minimize(&mut distances, evaluate, |prgr| {
        println!("Iteration {}: {}", prgr.niter, prgr.fx);
        false
    }).expect("Problem optimizing");*/

    let mut optimizer = stochastic_optimizers::Adam::new(distances, 0.001);

    for i in 0..100  {
        let (gradients, ll) = felsenstein.calculate_branch_gradients(optimizer.parameters(), &param);
        let neg_gradients: Vec<f64> = gradients.iter().map(|x| -*x).collect();
        println!("Gradients: {:?}", neg_gradients.iter().sum::<f64>());
        optimizer.step(&neg_gradients);
        println!("Iteration {}: {}", i, ll.iter().sum::<f64>());
    }

    return "".to_string();
}

pub fn read_model_paml<F : FloatTrait>(file: &str) -> ParamPrecomp<F, 20> {
    let content = std::fs::read_to_string(file).expect("Could not read model file");
    let floats: Vec<f64> = content
        .split_ascii_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    let mut R = na::SMatrix::<F, 20, 20>::zeros();
    let mut pi = na::SVector::<F, 20>::zeros();

    let mut idx = 0;
    for i in 0..20 {
        for j in 0..i {
            R[(i, j)] = <F as FloatTrait>::from_f64(floats[idx]);
            R[(j, i)] = R[(i, j)];
            idx += 1;
        }
    }

    for i in 0..20 {
        pi[i] = <F as FloatTrait>::from_f64(floats[idx]);
        idx += 1;
    }

    let sqrt_pi = pi.map(|x| Float::sqrt(x));

    diag_times_assign(R.as_view_mut(), sqrt_pi.iter().copied());
    times_diag_assign(R.as_view_mut(), sqrt_pi.iter().copied());
    

    compute_param_data(R.as_view(), sqrt_pi.as_view()).expect("Could not diagonalize rate matrix or eigenvalues are too big")
}
