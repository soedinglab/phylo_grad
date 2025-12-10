#![allow(non_snake_case)]

//! This code is used in the RaxML IQ-TREE benchmark for optimizing GTR parameters
//! It has a global matrix mode a per column matrix mode, caled "local" here.

use lazy_static::lazy_static;
use nalgebra as na;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::collections::HashMap;

use phylo_grad::{FelsensteinTree, FloatTrait};

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

pub fn seq2pll<F: FloatTrait>(seq: impl Iterator<Item = u8>) -> Vec<na::SVector<F, 20>> {
    seq.map(|c| *AMINO_MAPPING.get(&c).unwrap_or(&20))
        .map(|idx| {
            let mut v =
                na::SVector::<F, 20>::from_element(<F as FloatTrait>::from_f64(f64::NEG_INFINITY));
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

pub fn process_newick_alignment(
    newick: &str,
    sequences: &HashMap<String, Vec<u8>>,
) -> (
    phylo_grad::FelsensteinTree<f64, 20>,
    Vec<Vec<na::SVector<f64, 20>>>,
) {
    let tree = phylotree::tree::Tree::from_newick(newick).unwrap();

    let num_sides = sequences.values().next().unwrap().len();
    let level_order = tree.levelorder(&tree.get_root().unwrap()).unwrap();

    let mut parents = vec![i32::MAX; level_order.len()];
    let mut distances = vec![f64::NAN; level_order.len()];

    let num_leaves = tree.n_leaves();

    // idx_mapping[tree_idx] = new_idx
    let mut idx_mapping = vec![i32::MAX; level_order.len()];

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

    let mut leaf_pll = vec![];

    for i in 0..num_sides {
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

    let felsenstein = FelsensteinTree::<f64, 20>::new(&parents, &distances);
    (felsenstein, leaf_pll)
}

pub fn optimize_gtr_local(newick: &str, sequences: &HashMap<String, Vec<u8>>) -> f64 {
    let (felsenstein, mut leaf_pll) = process_newick_alignment(newick, sequences);

    for log_p in &mut leaf_pll {
        log_p.resize(felsenstein.num_nodes(), na::SVector::<f64, 20>::zeros());
    }

    let ll = leaf_pll
        .par_iter_mut()
        .map(|log_p| optimize_gtr_single_side(&felsenstein, &vec![0.0; 20], log_p))
        .sum::<f64>();

    return ll;
}

fn optimize_gtr_single_side(
    felsenstein: &FelsensteinTree<f64, 20>,
    log_pi_init: &[f64],
    log_p: &mut [na::SVector<f64, 20>],
) -> f64 {
    let evaluate = |x: &[f64], g: &mut [f64]| {
        let log_R = &x[..190];
        let log_pi_unormalized = &x[190..210];

        let rate_matrix_data = rate_matrix(log_R, log_pi_unormalized);

        let result = felsenstein.calculate_gradients_single_side(
            rate_matrix_data.S.as_view(),
            rate_matrix_data.sqrt_pi.as_view(),
            log_p,
        );

        let likelihood: f64 = result.log_likelihood;

        let (grad_log_R, grad_log_pi) =
            rate_matrix_backward(&rate_matrix_data, &result.grad_s, &result.grad_sqrt_pi);

        for i in 0..190 {
            g[i] = -grad_log_R[i];
        }
        for i in 0..20 {
            g[190 + i] = -grad_log_pi[i];
        }

        println!("Likelihood={}", likelihood);

        Ok(-likelihood)
    };

    let mut init = {
        let mut v = vec![0.0; 210];
        for i in 0..190 {
            v[i] = 0.0;
        }
        for i in 0..20 {
            v[190 + i] = log_pi_init[i];
        }
        v
    };

    let mut last_loss = f64::INFINITY;

    let optimizer = gosh_lbfgs::lbfgs()
        .with_max_iterations(0)
        .minimize(&mut init, evaluate, |prgr| {
            println!("Iteration {}: {}", prgr.niter, prgr.fx);
            if (last_loss - prgr.fx).abs() < 1e-6 * prgr.fx && prgr.niter > 10 {
                return true;
            }
            last_loss = prgr.fx;
            false
        })
        .expect("Problem optimizing");

    -optimizer.fx
}

pub fn optimize_gtr_global(newick: &str, sequences: &HashMap<String, Vec<u8>>) -> f64 {
    let (mut felsenstein, leaf_pll) = process_newick_alignment(newick, sequences);
    felsenstein.bind_leaf_log_p(leaf_pll);

    let evaluate = |x: &[f64], g: &mut [f64]| {
        let log_R = &x[..190];
        let log_pi_unormalized = &x[190..210];

        let rate_matrix_data = rate_matrix(log_R, log_pi_unormalized);

        let result = felsenstein
            .calculate_gradients(&vec![rate_matrix_data.S], &vec![rate_matrix_data.sqrt_pi]);

        let likelihood: f64 = result.log_likelihood.iter().sum();

        let (grad_log_R, grad_log_pi) = rate_matrix_backward(
            &rate_matrix_data,
            &result.grad_s[0],
            &result.grad_sqrt_pi[0],
        );

        for i in 0..190 {
            g[i] = -grad_log_R[i];
        }
        for i in 0..20 {
            g[190 + i] = -grad_log_pi[i];
        }

        println!("Likelihood={}", likelihood);

        Ok(-likelihood)
    };

    let amino_counts = {
        let mut counts = vec![0.0; 20];
        for seq in sequences.values() {
            for &c in seq {
                let idx = *AMINO_MAPPING.get(&c).unwrap_or(&20);
                if idx < 20 {
                    counts[idx as usize] += 1.0;
                }
            }
        }
        counts
    };

    let mut init = {
        let mut v = vec![0.0; 210];
        for i in 0..190 {
            v[i] = 0.0;
        }
        for i in 0..20 {
            v[190 + i] = (amino_counts[i] as f64).ln();
        }
        v
    };

    let mut last_loss = f64::INFINITY;

    let optimizer = gosh_lbfgs::lbfgs()
        .with_max_iterations(0)
        .minimize(&mut init, evaluate, |prgr| {
            println!("Iteration {}: {}", prgr.niter, prgr.fx);
            if (last_loss - prgr.fx).abs() < 1e-6 * prgr.fx && prgr.niter > 10 {
                return true;
            }
            last_loss = prgr.fx;
            false
        })
        .expect("Problem optimizing");
    -optimizer.fx
}

fn rate_matrix_backward(
    data: &RateMatrixData,
    s_cotangent: &na::SMatrix<f64, 20, 20>,
    sqrt_pi_cotangent: &na::SVector<f64, 20>,
) -> (Vec<f64>, Vec<f64>) {
    let mut grad_log_R = vec![0.0; 190];
    let mut grad_log_pi = vec![0.0; 20];

    let d_logS = data.S.component_mul(s_cotangent);

    let d_logM = -d_logS.sum();

    let piRpi_max = data.piRpi.max();
    let piRpi_exp: na::SMatrix<f64, 20, 20> = data.piRpi.map(|x| (x - piRpi_max).scalar_exp());
    let piRpi_sum = piRpi_exp.sum();

    let d_piRpi = piRpi_exp.map(|x| x * d_logM / piRpi_sum);

    let d_log_R_mat = d_logS + d_piRpi;

    let mut idx = 0;
    for i in 0..20 {
        for j in 0..i {
            grad_log_R[idx] = d_log_R_mat[(i, j)] + d_log_R_mat[(j, i)];
            idx += 1;
        }
    }

    let d_log_pi = 0.5 * (d_logS.row_sum().transpose() + d_logS.column_sum())
        + d_piRpi.row_sum().transpose()
        + d_piRpi.column_sum();

    let d_log_pi =
        d_log_pi + sqrt_pi_cotangent.component_mul(&data.log_pi.map(|x| 0.5 * (0.5 * x).exp()));

    let softmax_log_pi_unorm = phylo_grad::softmax(&data.log_pi_unormalized);

    for i in 0..20 {
        grad_log_pi[i] = d_log_pi[i] - softmax_log_pi_unorm[i] * d_log_pi.sum();
    }

    (grad_log_R, grad_log_pi)
}

fn rate_matrix(log_R: &[f64], log_pi_unormalized: &[f64]) -> RateMatrixData {
    let log_pi_unormalized: na::SVector<f64, 20> =
        na::SVector::<f64, 20>::from_iterator(log_pi_unormalized.iter().copied());
    let log_pi = log_pi_unormalized.add_scalar(-FloatTrait::logsumexp(log_pi_unormalized.iter()));

    let log_R_mat = {
        let mut mat = na::SMatrix::<f64, 20, 20>::zeros();
        let mut idx = 0;
        for i in 0..20 {
            for j in 0..i {
                mat[(i, j)] = log_R[idx];
                mat[(j, i)] = log_R[idx];
                idx += 1;
            }
        }
        mat
    };

    let piRpi = {
        let mut mat = na::SMatrix::<f64, 20, 20>::zeros();
        for i in 0..20 {
            for j in 0..20 {
                if i != j {
                    mat[(i, j)] = log_pi[i] + log_R_mat[(i, j)] + log_pi[j];
                } else {
                    mat[(i, j)] = f64::NEG_INFINITY;
                }
            }
        }
        mat
    };

    let logM = FloatTrait::logsumexp(piRpi.iter());

    let logS = {
        let mut mat = na::SMatrix::<f64, 20, 20>::zeros();
        for i in 0..20 {
            for j in 0..20 {
                if i != j {
                    mat[(i, j)] = 0.5 * (log_pi[i] + log_pi[j]) + log_R_mat[(i, j)] - logM;
                } else {
                    mat[(i, j)] = 0.0;
                }
            }
        }
        mat
    };

    let sqrt_pi = {
        let mut v = na::SVector::<f64, 20>::zeros();
        for i in 0..20 {
            v[i] = (log_pi[i] * 0.5).scalar_exp();
        }
        v
    };

    let S = {
        let mut mat = na::SMatrix::<f64, 20, 20>::zeros();
        for i in 0..20 {
            for j in 0..20 {
                mat[(i, j)] = logS[(i, j)].scalar_exp();
            }
        }
        mat
    };

    RateMatrixData {
        log_pi_unormalized,
        piRpi,
        log_pi,
        sqrt_pi,
        S,
    }
}
#[derive(Debug)]
struct RateMatrixData {
    log_pi_unormalized: na::SVector<f64, 20>,
    piRpi: na::SMatrix<f64, 20, 20>,
    log_pi: na::SVector<f64, 20>,
    sqrt_pi: na::SVector<f64, 20>,
    S: na::SMatrix<f64, 20, 20>,
}
