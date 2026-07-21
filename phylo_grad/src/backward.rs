use crate::forward::*;
use crate::tree::Bifurcation;

use nalgebra as na;

fn X<const DIM: usize>(
    eigenvalues: na::SVectorView<f64, DIM>,
    t: f64,
    exp_t_lambda: &na::SVector<f64, DIM>,
) -> na::SMatrix<f64, DIM, DIM> {
    na::SMatrix::<f64, DIM, DIM>::from_fn(|i, j| {
        let diff = f64::abs(eigenvalues[i] - eigenvalues[j]);
        if diff < 1e-10 {
            t * exp_t_lambda[i]
        } else if diff > 1.0 {
            (exp_t_lambda[i] - exp_t_lambda[j]) / (eigenvalues[i] - eigenvalues[j])
        } else {
            exp_t_lambda[j]
                * (f64::exp_m1(t * (eigenvalues[i] - eigenvalues[j]))
                    / (eigenvalues[i] - eigenvalues[j]))
        }
    })
}

/// Backward pass for expm(distance * Q)x
/// It gets the cotangent of expm(distance * Q)x and accumulates the cotangent of Q
pub fn d_expm_vjp<const DIM: usize>(
    Q_cotangent_vector: &mut na::SMatrix<f64, DIM, DIM>,
    distance: f64,
    param: &ParamPrecomp<DIM>,
    exp_t_lambda: &na::SVector<f64, DIM>,
    x: &na::SVector<f64, DIM>,
    cotangent_vector: &na::SVector<f64, DIM>,
) {
    let A_t = param.V_pi.transpose();

    // A^Ty
    let Aty = A_t * cotangent_vector;

    // x^TA^-T
    let xtAinv_t = x.transpose() * param.V_pi_inv.transpose();

    let mut X = X(param.eigenvalues.as_view(), distance, exp_t_lambda);

    X.component_mul_assign(&(Aty * xtAinv_t));

    *Q_cotangent_vector += X;
}

/// Backward pass for rho(W) = 1/sqrt_pi @ S @ sqrt_pi
pub fn d_param<const DIM: usize>(
    cotangent_vector: na::SMatrixView<f64, DIM, DIM>,
    param: &ParamPrecomp<DIM>,
) -> (na::SMatrix<f64, DIM, DIM>, na::SVector<f64, DIM>) {
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

    let mut grad_s = na::SMatrix::<f64, DIM, DIM>::zeros();
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
    let mut grad_sqrt_pi = na::SVector::<f64, DIM>::zeros();
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

pub fn d_log_transition_bifurcation_vjp<const DIM: usize>(
    cotangents: &mut [na::SVector<f64, DIM>],
    lin_pl: &[na::SVector<f64, DIM>],
    lin_child_contributions: &[na::SVector<f64, DIM>],
    forward: &[ModelEdgeData<DIM>],
    param: &ParamPrecomp<DIM>,
    d_Q_output: &mut na::SMatrix<f64, DIM, DIM>,
    bifurcation: &Bifurcation,
    distances: &[f64],
    offsets: &[u32],
    d_edge_lengths: Option<&mut [f64]>,
) {
    let scaler = if offsets[bifurcation.parent as usize] == 0 {
        1.0
    } else {
        f64::powi(2.0, offsets[bifurcation.parent as usize] as i32)
    };

    fn single_edge<const DIM: usize>(
        d_Q_output: &mut na::SMatrix<f64, DIM, DIM>,
        cotangent_vector: &na::SVector<f64, DIM>,
        distance: f64,
        param: &ParamPrecomp<DIM>,
        exp_t_lambda: &na::SVector<f64, DIM>,
        child_pl: &na::SVector<f64, DIM>,
        child_cotangent: &mut na::SVector<f64, DIM>,
    ) {
        d_expm_vjp(
            d_Q_output,
            distance,
            param,
            exp_t_lambda,
            child_pl,
            cotangent_vector,
        );

        // The child cotangent is exp(tQ)^T * cotangent_vector
        // This is the same as V_pi_inv^T * diag(exp_t_lambda) * V_pi T * cotangent_vector
        *child_cotangent = param.V_pi.transpose() * cotangent_vector;
        child_cotangent.component_mul_assign(exp_t_lambda);
        *child_cotangent = param.V_pi_inv.transpose() * *child_cotangent;
    }

    if bifurcation.middle == -1 {
        let parent_cotangent = cotangents[bifurcation.parent as usize] * scaler;

        let left_cotangent =
            parent_cotangent.component_mul(&lin_child_contributions[bifurcation.right as usize]);
        let right_cotangent =
            parent_cotangent.component_mul(&lin_child_contributions[bifurcation.left as usize]);

        single_edge(
            d_Q_output,
            &left_cotangent,
            distances[bifurcation.left as usize],
            param,
            &forward[bifurcation.left as usize].exp_t_lambda,
            &lin_pl[bifurcation.left as usize],
            &mut cotangents[bifurcation.left as usize],
        );
        single_edge(
            d_Q_output,
            &right_cotangent,
            distances[bifurcation.right as usize],
            param,
            &forward[bifurcation.right as usize].exp_t_lambda,
            &lin_pl[bifurcation.right as usize],
            &mut cotangents[bifurcation.right as usize],
        );

        if let Some(d_edge_lengths) = d_edge_lengths {
            todo!("Do edge gradients");
        }
    } else {
        // trifurcation case
        let parent_cotangent = cotangents[bifurcation.parent as usize] * scaler;

        let left_cotangent = parent_cotangent
            .component_mul(&lin_child_contributions[bifurcation.middle as usize])
            .component_mul(&lin_child_contributions[bifurcation.right as usize]);
        let middle_cotangent = parent_cotangent
            .component_mul(&lin_child_contributions[bifurcation.left as usize])
            .component_mul(&lin_child_contributions[bifurcation.right as usize]);
        let right_cotangent = parent_cotangent
            .component_mul(&lin_child_contributions[bifurcation.left as usize])
            .component_mul(&lin_child_contributions[bifurcation.middle as usize]);



        single_edge(
            d_Q_output,
            &left_cotangent,
            distances[bifurcation.left as usize],
            param,
            &forward[bifurcation.left as usize].exp_t_lambda,
            &lin_pl[bifurcation.left as usize],
            &mut cotangents[bifurcation.left as usize],
        );
        single_edge(
            d_Q_output,
            &middle_cotangent,
            distances[bifurcation.middle as usize],
            param,
            &forward[bifurcation.middle as usize].exp_t_lambda,
            &lin_pl[bifurcation.middle as usize],
            &mut cotangents[bifurcation.middle as usize],
        );
        single_edge(
            d_Q_output,
            &right_cotangent,
            distances[bifurcation.right as usize],
            param,
            &forward[bifurcation.right as usize].exp_t_lambda,
            &lin_pl[bifurcation.right as usize],
            &mut cotangents[bifurcation.right as usize],
        );

        if let Some(d_edge_lengths) = d_edge_lengths {
            todo!("Do edge gradients");
        }
    };
}
