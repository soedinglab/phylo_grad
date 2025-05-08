use crate::data_types::*;
use crate::forward::*;

use nalgebra as na;

pub struct BackwardData<F, const DIM: usize> {
    pub grad_log_p: na::SVector<F, DIM>,
}

/// Numerical stable softmax
pub fn softmax<F: FloatTrait, const N: usize>(x: &na::SVector<F, N>) -> na::SVector<F, N> {
    let x_max = x.max();

    let mut result = x.add_scalar(-x_max);

    unsafe {
        F::vec_exp(std::mem::transmute::<&mut [[F;N];1], &mut [F; N]>(&mut result.data.0));
    }
    result /= result.sum();
    result
}

pub fn d_ln_vjp<F: FloatTrait, const DIM: usize>(
    cotangent_vector: &mut na::SMatrix<F, DIM, DIM>,
    argument: &na::SMatrix<F, DIM, DIM>,
) {
    for i in 0..DIM {
        for j in 0..DIM {
            cotangent_vector[(i, j)] /= argument[(i, j)];
        }
    }
}

fn X<F: FloatTrait, const DIM: usize>(
    eigenvalues: na::SVectorView<F, DIM>,
    t: F,
    exp_t_lambda: &na::SVector<F, DIM>
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

fn child_input_forward_data<F: FloatTrait, const DIM: usize>(
    log_p: na::SVectorView<F, DIM>,
    log_transition_T: &na::SMatrix<F, DIM, DIM>,
    output: &mut na::SMatrix<F, DIM, DIM>,
) {
    /* result = log_p[None, :] + log_transition */
    for i in 0..DIM {
        for j in 0..DIM {
            output[(i, j)] = log_p[j] + log_transition_T[(j, i)];
        }
    }
}

fn d_broadcast_vjp<F: FloatTrait, const DIM: usize>(
    cotangent_vector: na::SMatrixView<F, DIM, DIM>,
) -> na::SVector<F, DIM> {
    /* sum(cotangent_vector, dim=1) */
    na::SVector::<F, DIM>::from_iterator(cotangent_vector.column_iter().map(|col| col.sum()))
}

pub fn d_log_transition_child_input_vjp<F: FloatTrait, const DIM: usize>(
    cotangent_vector: na::SVectorView<F, DIM>,
    log_p: na::SVectorView<F, DIM>,
    forward: &LogTransitionForwardData<F, DIM>,
    compute_grad_log_p: bool,
    output: &mut na::SMatrix<F, DIM, DIM>,
) -> Option<na::SVector<F, DIM>> {
    child_input_forward_data(log_p, &forward.log_transition_T, output);

    /* d_lse */
    for mut row in output.row_iter_mut() {
        row.copy_from(&softmax(&row.transpose()).transpose());
    }
    diag_times_assign(output.as_view_mut(), cotangent_vector.iter().copied());

    let grad_log_p = if compute_grad_log_p {
        Some(d_broadcast_vjp(output.as_view()))
    } else {
        None
    };

    grad_log_p
}

pub fn d_child_input_param<F: FloatTrait, const DIM: usize>(
    cotangent_vector: na::SVectorView<F, DIM>,
    distance: F,
    param: &ParamPrecomp<F, DIM>,
    log_p: na::SVectorView<F, DIM>,
    forward: &LogTransitionForwardData<F, DIM>,
    compute_grad_log_p: bool,
    output: &mut na::SMatrix<F, DIM, DIM>,
) -> Option<na::SVector<F, DIM>> {
    let grad_log_p = d_log_transition_child_input_vjp(
        cotangent_vector,
        log_p,
        forward,
        compute_grad_log_p,
        output,
    );
    d_ln_vjp(output, &forward.matrix_exp);

    d_expm_vjp(output, distance, param, &forward.exp_t_lambda);

    grad_log_p
}
