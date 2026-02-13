use crate::data_types::*;
use crate::forward::*;
use crate::tree::Bifurcation;

use nalgebra as na;

/// Numerical stable softmax
pub fn softmax<F: FloatTrait, const N: usize>(x: &na::SVector<F, N>) -> na::SVector<F, N> {
    let x_max = x.max();

    let mut result = x.add_scalar(-x_max);

    unsafe {
        F::vec_exp(std::mem::transmute::<&mut [[F; N]; 1], &mut [F; N]>(
            &mut result.data.0,
        ));
    }
    result /= result.sum();
    result
}

fn X<F: FloatTrait, const DIM: usize>(
    eigenvalues: na::SVectorView<F, DIM>,
    t: F,
    exp_t_lambda: &na::SVector<F, DIM>,
) -> na::SMatrix<F, DIM, DIM> {
    na::SMatrix::<F, DIM, DIM>::from_fn(|i, j| {
        let diff = num_traits::Float::abs(eigenvalues[i] - eigenvalues[j]);
        if diff < FloatTrait::from_f64(1e-10) {
            t * exp_t_lambda[i]
        } else if diff > FloatTrait::from_f64(1.0) {
            (exp_t_lambda[i] - exp_t_lambda[j]) / (eigenvalues[i] - eigenvalues[j])
        } else {
            exp_t_lambda[j]
                * (num_traits::Float::exp_m1(t * (eigenvalues[i] - eigenvalues[j]))
                    / (eigenvalues[i] - eigenvalues[j]))
        }
    })
}

/// Backward pass for expm(distance * 1/sqrt_pi @ S @ sqrt_pi)
pub fn d_expm_vjp<F: FloatTrait, const DIM: usize>(
    cotangent_vector: &mut na::SMatrix<F, DIM, DIM>,
    distance: F,
    param: &ParamPrecomp<F, DIM>,
    exp_t_lambda: &na::SVector<F, DIM>,
) {
    /*
    B = V_pi_invT
    B_inv = V_pi_T

    result =
      ((B_inv @ cotangent @ B) \odot X_T(lam, dist))

      we do not do the outer most matrix muls here
    */

    let B = param.V_pi_inv.transpose();
    let B_inv = param.V_pi.transpose();

    let X = X(param.eigenvalues.as_view(), distance, exp_t_lambda);

    *cotangent_vector *= B;
    *cotangent_vector = B_inv * *cotangent_vector;

    cotangent_vector.component_mul_assign(&X);
}

/// Backward pass for rho(W) = 1/sqrt_pi @ S @ sqrt_pi
pub fn d_param<F: FloatTrait, const DIM: usize>(
    cotangent_vector: na::SMatrixView<F, DIM, DIM>,
    param: &ParamPrecomp<F, DIM>,
) -> (na::SMatrix<F, DIM, DIM>, na::SVector<F, DIM>) {
    let sqrt_pi = param.sqrt_pi.clone_owned();
    let sqrt_pi_recip = param.sqrt_pi_recip.clone_owned();
    let symmetric = param.symmetric_matrix.clone_owned();

    /* d_S rho(W) = diag(sqrt_pi)^-1 * W * diag(sqrt_pi) */
    let grad_symmetric = {
        let mut grad_symmetric_pre = cotangent_vector.clone_owned();
        diag_times_assign(
            grad_symmetric_pre.as_view_mut(),
            sqrt_pi_recip.iter().copied(),
        );
        times_diag_assign(grad_symmetric_pre.as_view_mut(), sqrt_pi.iter().copied());
        grad_symmetric_pre
    };

    /* d_delta rho(W) [i, j]:
            0 if i >= j
            grad_S[i, j] + grad_S[j, i] - grad_S[i, i] * pi_j / pi_i - grad_S[j, j] * pi_i / pi_j if i < j
    */

    let mut grad_s = na::SMatrix::<F, DIM, DIM>::zeros();
    for j in 0..DIM {
        for i in 0..j {
            grad_s[(i, j)] = grad_symmetric[(i, j)] + grad_symmetric[(j, i)]
                - grad_symmetric[(i, i)] * sqrt_pi_recip[i] * sqrt_pi[j]
                - grad_symmetric[(j, j)] * sqrt_pi_recip[j] * sqrt_pi[i]
        }
    }

    /* grad_sqrt_pi [j] =
        Sum_{i, i!=j} (
            sqrt_pi_recip[j]
            * S[i, j]
            * (sqrt_pi[j]*sqrt_pi_recip[i] * (w_ij - w_ii)
               -sqrt_pi[i]*sqrt_pi_recip[j] * (w_ji - w_jj))
        )
    */
    let mut grad_sqrt_pi = na::SVector::<F, DIM>::zeros();
    for j in 0..DIM {
        for i in 0..DIM {
            if i != j {
                grad_sqrt_pi[j] += sqrt_pi_recip[j]
                    * symmetric[(i, j)]
                    * (sqrt_pi[j]
                        * sqrt_pi_recip[i]
                        * (cotangent_vector[(i, j)] - cotangent_vector[(i, i)])
                        - sqrt_pi[i]
                            * sqrt_pi_recip[j]
                            * (cotangent_vector[(j, i)] - cotangent_vector[(j, j)]))
            }
        }
    }

    (grad_s, grad_sqrt_pi)
}

pub fn d_log_transition_bifurcation_vjp<F: FloatTrait, const DIM: usize>(
    cotangents: &mut[na::SVector<F, DIM>],
    lin_pl: &[na::SVector<F, DIM>],
    forward: &[ModelEdgeData<F, DIM>],
    param: &ParamPrecomp<F, DIM>,
    d_Q_output: &mut na::SMatrix<F, DIM, DIM>,
    bifurcation: &Bifurcation,
    distances: &[F],
) {
    if bifurcation.middle == -1 {
        // bifurcation case

        let mut d_trans_left = na::SMatrix::<F, DIM, DIM>::zeros();
        let mut d_trans_right = na::SMatrix::<F, DIM, DIM>::zeros();
        let parent_cotangent = &lin_pl[bifurcation.parent as usize];
        for a in 0..DIM {
            let left_contribution = forward[bifurcation.left as usize].transition_T.column(a).dot(&lin_pl[bifurcation.left as usize]);
            let right_contribution = forward[bifurcation.right as usize].transition_T.column(a).dot(&lin_pl[bifurcation.right as usize]);
            for b in 0..DIM {
                cotangents[bifurcation.left as usize][b] = parent_cotangent[b] * right_contribution * forward[bifurcation.left as usize].transition_T[(b, a)];
                cotangents[bifurcation.right as usize][b] = parent_cotangent[b] * left_contribution * forward[bifurcation.right as usize].transition_T[(b, a)];
            }
            d_trans_left.set_row(a, &(lin_pl[bifurcation.left as usize].component_mul(parent_cotangent) * right_contribution).transpose());
            d_trans_right.set_row(a, &(lin_pl[bifurcation.right as usize].component_mul(parent_cotangent) * left_contribution).transpose());
            
        }

        d_expm_vjp(&mut d_trans_left, distances[bifurcation.left as usize], param, &forward[bifurcation.left as usize].exp_t_lambda); 
        d_expm_vjp(&mut d_trans_right, distances[bifurcation.right as usize], param, &forward[bifurcation.right as usize].exp_t_lambda);
        *d_Q_output += d_trans_left;
        *d_Q_output += d_trans_right;
    } else {
        let mut d_trans_left = na::SMatrix::<F, DIM, DIM>::zeros();
        let mut d_trans_right = na::SMatrix::<F, DIM, DIM>::zeros();
        let mut d_trans_middle = na::SMatrix::<F, DIM, DIM>::zeros();
        let parent_cotangent = &lin_pl[bifurcation.parent as usize];
        for a in 0..DIM {
            let left_contribution = forward[bifurcation.left as usize].transition_T.column(a).dot(&lin_pl[bifurcation.left as usize]);
            let right_contribution = forward[bifurcation.right as usize].transition_T.column(a).dot(&lin_pl[bifurcation.right as usize]);
            let middle_contribution = forward[bifurcation.middle as usize].transition_T.column(a).dot(&lin_pl[bifurcation.middle as usize]);
            for b in 0..DIM {
                cotangents[bifurcation.left as usize][b] = (right_contribution * middle_contribution * parent_cotangent[b]) * forward[bifurcation.left as usize].transition_T[(b, a)];
                cotangents[bifurcation.right as usize][b] = (left_contribution * middle_contribution * parent_cotangent[b]) * forward[bifurcation.right as usize].transition_T[(b, a)];
                cotangents[bifurcation.middle as usize][b] = (left_contribution * right_contribution * parent_cotangent[b]) * forward[bifurcation.middle as usize].transition_T[(b, a)];
            }
            d_trans_left.set_row(a, &(lin_pl[bifurcation.left as usize].component_mul(parent_cotangent) * (right_contribution * middle_contribution)).transpose());
            d_trans_right.set_row(a, &(lin_pl[bifurcation.right as usize].component_mul(parent_cotangent) * (left_contribution * middle_contribution)).transpose());
            d_trans_middle.set_row(a, &(lin_pl[bifurcation.middle as usize].component_mul(parent_cotangent) * (left_contribution * right_contribution)).transpose());
        }
        d_expm_vjp(&mut d_trans_left, distances[bifurcation.left as usize], param, &forward[bifurcation.left as usize].exp_t_lambda); 
        d_expm_vjp(&mut d_trans_right, distances[bifurcation.right as usize], param, &forward[bifurcation.right as usize].exp_t_lambda);
        d_expm_vjp(&mut d_trans_middle, distances[bifurcation.middle as usize], param, &forward[bifurcation.middle as usize].exp_t_lambda);
        *d_Q_output += d_trans_left;
        *d_Q_output += d_trans_right;
        *d_Q_output += d_trans_middle;
    };


   
    
}