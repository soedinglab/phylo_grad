use crate::data_types::*;
use crate::forward::*;

use na::{Const, DefaultAllocator};

pub struct BackwardData<const DIM: usize> {
    pub grad_log_p: na::SVector<Float, DIM>,
}

pub fn softmax_na<const N: usize>(x: na::SVectorView<Float, N>) -> na::SVector<Float, N> {
    let mut result: na::SVector<Float, N>;
    let x_max = *x
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .expect("Iterator cannot be empty");

    result = x.add_scalar(-x_max);
    result = result.map(Float::exp);

    let scale = result.sum().recip();
    result *= scale;
    result
}

fn d_exp_vjp<const N: usize>(
    cotangent_vector: na::SMatrixView<Float, N, N>,
    argument: na::SMatrixView<Float, N, N>,
) -> na::SMatrix<Float, N, N>
where
    Const<N>: Doubleable,
    TwoTimesConst<N>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, N>,
{
    /* exp([[X^T, 0], [w, X^T]]) = [[exp(X^T), 0], [w (D_X exp), exp(X^T)]] */
    let mut block_triangular = MatrixNNAllocated::<Float, TwoTimesConst<N>>::zeros();
    let argument_transposed = argument.transpose();
    block_triangular
        .index_mut((..N, ..N))
        .copy_from(&argument_transposed);
    block_triangular
        .index_mut((N.., N..))
        .copy_from(&argument_transposed);
    block_triangular
        .index_mut((N.., ..N))
        .copy_from(&cotangent_vector);
    let exp_combined = block_triangular.exp();

    let dexp: na::SMatrix<Float, N, N> = exp_combined.fixed_view::<N, N>(N, 0).clone_owned();
    dexp
}

fn d_map_ln_vjp<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    argument: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /* If w = \sum w_kl dy_kl, then
    map_ln^*(w) = \sum_kl (w_kl / x_kl) dx_kl = map_recip(x) \odot w */
    let mut rec = argument.map(Float::recip);
    rec.component_mul_assign(&cotangent_vector);
    rec
}

fn d_rate_log_transition_vjp<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
    forward: &LogTransitionForwardData<DIM>,
) -> na::SMatrix<Float, DIM, DIM>
where
    Const<DIM>: Doubleable,
    TwoTimesConst<DIM>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, DIM>,
{
    /* log_tr*|y (w) = mul*|rx @ exp*|exp(rx) @ map_ln*|y @ w
     */
    let forward_1 = forward.step_1;
    let forward_2 = forward.step_2;

    let reverse_1 = d_map_ln_vjp(cotangent_vector, forward_2.as_view());
    let reverse_2 = d_exp_vjp(reverse_1.as_view(), forward_1.as_view());
    let result = reverse_2 * distance;
    result
}

/* verified */
fn exprel(x: f64) -> f64 {
    const THRESHOLD: f64 = 2e-3;
    /* LOG_F64_MAX = f64::MAX.log() */
    const LOG_F64_MAX: f64 = 7.0978271289338397e+02;
    /* LOG_F64_MIN = ? */
    const LOG_F64_MIN: f64 = -7.0839641853226408e+02;
    if x < LOG_F64_MIN as f64 {
        -1.0 / x
    } else if x < -THRESHOLD {
        (x.exp() - 1.0) / x
    } else if x < THRESHOLD {
        1.0 + 0.5 * x * (1.0 + x / 3.0 * (1.0 + 0.25 * x * (1.0 + 0.2 * x)))
    } else if x < LOG_F64_MAX as f64 {
        (x.exp() - 1.0) / x
    } else {
        f64::MAX
    }
}

/* verified */
/* TODO simplify */
fn X_transposed<const DIM: usize>(
    eigenvalues: na::SVectorView<Float, DIM>,
    distance: Float,
) -> na::SMatrix<Float, DIM, DIM> {
    /* X = coeff[:, None] * y,
    y = { exprel( d * (lambda[:, None] - lambda[None, :]) ) if i!=j,
        { 1 if i==j
    coeff = exp(d * lambda) * d */

    /* X_T = coeff[None, :] * y_T
    y_T = { 1 if i == j,
          { exprel(d * lambda[None, :] - lambda[:, None] if i!=j) */

    /* Assuming exprel(0) is Nan and not a runtime error. */
    let coeff = (distance * eigenvalues).map(f64::exp) * distance;
    let id_iter = std::iter::zip(
        (0..DIM).flat_map(|n| std::iter::repeat(n).take(DIM)),
        std::iter::repeat(0..DIM).flatten(),
    );
    let arg_1 = na::SMatrix::<Float, DIM, DIM>::from_iterator(
        id_iter.map(|(i, j)| (eigenvalues[j] - eigenvalues[i]) * distance),
    );
    let mut result = arg_1.map(exprel);
    result.fill_diagonal(1 as Float);
    times_diag_assign(result.as_view_mut(), coeff.iter().copied());

    result
}

/* verified */
fn d_transition_mcgibbon_pande<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
    param: &ParamData<DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /*
    B = V_pi_invT
    B_inv = V_pi_T

    result =
      B @ ((B_inv @ cotangent @ B) \odot X_T(lam, dist)) @ B_inv
    */

    let B = param.V_pi_inv.transpose();
    let B_inv = param.V_pi.transpose();

    let X_T = X_transposed(param.eigenvalues.as_view(), distance);

    /* TODO optimize */
    //let result = B * ((B_inv * cotangent_vector * B).component_mul(&X_T)) * B_inv;
    let mut result = B_inv * cotangent_vector;
    result = result * B;
    result.component_mul_assign(&X_T);
    result = B * result;
    result = result * B_inv;

    result
}

pub fn d_param<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    param: &ParamData<DIM>,
) -> (na::SMatrix<Float, DIM, DIM>, na::SVector<Float, DIM>) {
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

    let mut grad_delta = na::SMatrix::<Float, DIM, DIM>::zeros();
    for j in 0..DIM {
        for i in 0..j {
            grad_delta[(i, j)] = grad_symmetric[(i, j)] + grad_symmetric[(j, i)]
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
    let mut grad_sqrt_pi = na::SVector::<Float, DIM>::zeros();
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

    (grad_delta, grad_sqrt_pi)
}

fn child_input_forward_data<const DIM: usize>(
    log_p: na::SVectorView<Float, DIM>,
    /* TODO get by value */
    log_transition: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /* result = log_p[:, None] + log_transition */
    let mut result = na::SMatrix::<Float, DIM, DIM>::from_iterator(
        log_p.iter().flat_map(|&x| std::iter::repeat(x).take(DIM)),
    );
    result += log_transition;
    result
}

fn d_broadcast_vjp<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
) -> na::SVector<Float, DIM> {
    /* sum(cotangent_vector, dim=1) */
    na::SVector::<Float, DIM>::from_iterator(cotangent_vector.column_iter().map(|col| col.sum()))
}

fn d_log_transition_child_input_vjp<const DIM: usize>(
    cotangent_vector: na::SVectorView<Float, DIM>,
    log_p: na::SVectorView<Float, DIM>,
    forward: &LogTransitionForwardData<DIM>,
    compute_grad_log_p: bool,
) -> (
    na::SMatrix<Float, DIM, DIM>,
    Option<na::SVector<Float, DIM>>,
) {
    let log_transition = forward.log_transition;
    let mut forward_1 = child_input_forward_data(log_p, log_transition.as_view());

    /* d_lse */
    for mut row in forward_1.row_iter_mut() {
        row.copy_from(&softmax_na(row.transpose().as_view()).transpose());
    }
    diag_times_assign(forward_1.as_view_mut(), cotangent_vector.iter().copied());

    let grad_log_p = if compute_grad_log_p {
        Some(d_broadcast_vjp(forward_1.as_view()))
    } else {
        None
    };

    let grad_log_transition = forward_1;

    (grad_log_transition, grad_log_p)
}

pub fn d_child_input_vjp<const DIM: usize>(
    cotangent_vector: na::SVectorView<Float, DIM>,
    distance: Float,
    log_p: na::SVectorView<Float, DIM>,
    forward: &LogTransitionForwardData<DIM>,
    compute_grad_log_p: bool,
) -> (
    na::SMatrix<Float, DIM, DIM>,
    Option<na::SVector<Float, DIM>>,
)
where
    Const<DIM>: Doubleable,
    TwoTimesConst<DIM>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, DIM>,
{
    let (grad_log_transition, grad_log_p) =
        d_log_transition_child_input_vjp(cotangent_vector, log_p, forward, compute_grad_log_p);

    let grad_rate = d_rate_log_transition_vjp(grad_log_transition.as_view(), distance, forward);

    (grad_rate, grad_log_p)
}

pub fn d_child_input_param<const DIM: usize>(
    cotangent_vector: na::SVectorView<Float, DIM>,
    distance: Float,
    param: &ParamData<DIM>,
    log_p: na::SVectorView<Float, DIM>,
    forward: &LogTransitionForwardData<DIM>,
    compute_grad_log_p: bool,
) -> (
    na::SMatrix<Float, DIM, DIM>,
    Option<na::SVector<Float, DIM>>,
) {
    let (grad_log_transition, grad_log_p) =
        d_log_transition_child_input_vjp(cotangent_vector, log_p, forward, compute_grad_log_p);

    let transition = forward.step_2;
    let grad_transition = d_map_ln_vjp(grad_log_transition.as_view(), transition.as_view());

    let grad_rate = d_transition_mcgibbon_pande(grad_transition.as_view(), distance, param);

    (grad_rate, grad_log_p)
}
