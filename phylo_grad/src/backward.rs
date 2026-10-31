use crate::forward::*;

use nalgebra as na;

/// Below this |t * (lambda_i - lambda_j)| the divided difference is evaluated with a Taylor series,
/// above it (exp(t lambda_i) - exp(t lambda_j)) / (lambda_i - lambda_j) has a relative error of at most ~eps / 1e-3.
const DIVIDED_DIFFERENCE_TAYLOR_THRESHOLD: f64 = 1e-3;

/// Backward pass for expm(distance * Q)x
/// It gets the cotangent of expm(distance * Q)x and accumulates the cotangent of Q
///
/// Q_cotangent += X ∘ (A^T y)(A^-1 x)^T with the divided differences
/// X_ij = (exp(t lambda_i) - exp(t lambda_j)) / (lambda_i - lambda_j), X_ii = t exp(t lambda_i)
/// X is never materialized, so this is a single pass over Q_cotangent without divisions or transcendental functions.
/// Aty = A^T y and Ainv_x = A^-1 x are passed in, because the caller needs them as well.
pub fn d_expm_vjp<const DIM: usize>(
    Q_cotangent_vector: &mut na::SMatrix<f64, DIM, DIM>,
    distance: f64,
    param: &ParamPrecomp<DIM>,
    exp_t_lambda: &na::SVector<f64, DIM>,
    Aty: &na::SVector<f64, DIM>,
    Ainv_x: &na::SVector<f64, DIM>,
) {
    let t = distance;
    for j in 0..DIM {
        let exp_j = exp_t_lambda[j];
        let t_exp_j = t * exp_j;
        let Ainv_x_j = Ainv_x[j];
        let diff = param.eigenvalue_diff.column(j);
        let diff_recip = param.eigenvalue_diff_recip.column(j);
        let mut Q_cotangent_j = Q_cotangent_vector.column_mut(j);
        for i in 0..DIM {
            let z = t * diff[i];
            let difference_quotient = (exp_t_lambda[i] - exp_j) * diff_recip[i];
            // t exp(t lambda_j) expm1(z) / z
            let taylor = t_exp_j
                * (1.0 + z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z * (1.0 / 120.0)))));
            let x_ij = if z.abs() > DIVIDED_DIFFERENCE_TAYLOR_THRESHOLD {
                difference_quotient
            } else {
                taylor
            };
            Q_cotangent_j[i] += x_ij * Aty[i] * Ainv_x_j;
        }
    }
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

/// Backward pass for the contribution of a single edge to its parent: exp(tQ) child_pl
/// Ainv_x is V_pi_inv * child_pl.
/// cotangent_vector is the cotangent of this contribution, it accumulates the cotangent of Q.
/// If needed, it writes the cotangent of child_pl and adds the derivative with respect to the edge length.
pub fn d_edge_vjp<const DIM: usize>(
    d_Q_output: &mut na::SMatrix<f64, DIM, DIM>,
    cotangent_vector: &na::SVector<f64, DIM>,
    distance: f64,
    param: &ParamPrecomp<DIM>,
    exp_t_lambda: &na::SVector<f64, DIM>,
    Ainv_x: &na::SVector<f64, DIM>,
    child_cotangent: Option<&mut na::SVector<f64, DIM>>,
    d_edge_length: Option<&mut f64>,
) {
    // exp(tQ) = V_pi * diag(exp_t_lambda) * V_pi_inv, so all cotangents only need
    // V_pi^T * cotangent_vector and V_pi_inv * child_pl
    let Aty = param.V_pi.tr_mul(cotangent_vector);

    d_expm_vjp(d_Q_output, distance, param, exp_t_lambda, &Aty, Ainv_x);

    if let Some(child_cotangent) = child_cotangent {
        // The child cotangent is exp(tQ)^T * cotangent_vector = V_pi_inv^T * diag(exp_t_lambda) * V_pi^T * cotangent_vector
        *child_cotangent = param.V_pi_inv.tr_mul(&Aty.component_mul(exp_t_lambda));
    }

    if let Some(d_edge_length) = d_edge_length {
        // d/dt cotangent_vector^T exp(tQ) x = (V_pi^T cotangent_vector)^T diag(lambda_i * exp(t * lambda_i)) V_pi_inv x
        *d_edge_length += Aty
            .component_mul(&param.eigenvalues)
            .component_mul(exp_t_lambda)
            .dot(Ainv_x);
    }
}
