use crate::data_types::*;
use crate::forward::*;

use na::{Const, DefaultAllocator};

pub struct BackwardData<const DIM: usize> {
    pub grad_log_p: [Float; DIM],
}

fn softmax_na<const N: usize>(x: na::SVectorView<Float, N>) -> na::SVector<Float, N> {
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

pub fn softmax_inplace(x: &mut [Float]) {
    let x_max = *x
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .expect("Iterator cannot be empty");

    for item in x.iter_mut() {
        *item -= x_max;
    }
    for item in x.iter_mut() {
        *item = item.exp();
    }
    let scale = x.iter().sum::<Float>().recip();
    for item in x.iter_mut() {
        *item *= scale;
    }
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
    let mut result = B_inv * cotangent_vector * B;
    result.component_mul_assign(&X_T);
    result = B * cotangent_vector * B_inv;

    result
}

/* TODO! avoid using diagonal entries of S. */
/* TODO! use an index iterator to only compute the entries above the diagonal */
pub fn d_param<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    symmetric: na::SMatrixView<Float, DIM, DIM>,
    sqrt_pi: na::SVectorView<Float, DIM>,
) -> (na::SMatrix<Float, DIM, DIM>, na::SVector<Float, DIM>) {
    let sqrt_pi_recip = sqrt_pi.map(Float::recip);

    /* d_S rho(W) = diag(sqrt_pi)^-1 * W * diag(sqrt_pi) */
    let mut grad_symmetric = cotangent_vector.clone_owned();
    diag_times_assign(grad_symmetric.as_view_mut(), sqrt_pi_recip.iter().copied());
    times_diag_assign(grad_symmetric.as_view_mut(), sqrt_pi.iter().copied());

    /* d_pi rho(W) [i] = 0.5 / sqrt_pi * (s_ki * (w_ki * sqrt_pi_recip - w_ik * sqrt_pi * pi_i_recip)).sum() */
    let mut grad_pi = 0.5 * sqrt_pi_recip;
    for i in 0..DIM {
        let s_ki = symmetric.column(i);
        let w_ki = cotangent_vector.column(i);
        let w_ik = cotangent_vector.row(i).transpose();
        let pi_i_recip = sqrt_pi_recip[i] * sqrt_pi_recip[i];
        let mut summands =
            w_ki.component_mul(&sqrt_pi_recip) - w_ik.component_mul(&sqrt_pi) * pi_i_recip;
        summands[i] = 0.0 as Float;
        summands.component_mul_assign(&s_ki);
        grad_pi[i] *= summands.sum();
    }

    (grad_symmetric, grad_pi)
}

/* TODO extract iterator, use it to compute d_broadcast (d_lse) */
fn child_input_forward_data<const DIM: usize>(
    log_p: &[Float; DIM],
    /* TODO get by value */
    log_transition: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    let mut res =
        na::SMatrix::<Float, DIM, DIM>::from_iterator((0..DIM).flat_map(|_| log_p.iter().copied()));
    res += log_transition;
    for mut col in res.column_iter_mut() {
        col.copy_from(&softmax_na(col.as_view()));
    }
    res
}

fn d_broadcast_vjp<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
) -> [Float; DIM] {
    let mut result_iter = cotangent_vector.row_iter().map(|row| row.iter().sum());
    let mut result = [0.0 as Float; DIM];
    for i in 0..DIM {
        result[i] = result_iter.next().unwrap();
    }
    result
}

pub fn d_child_input_vjp<const DIM: usize>(
    cotangent_vector: [Float; DIM],
    distance: Float,
    log_p: &[Float; DIM],
    forward: &LogTransitionForwardData<DIM>,
    compute_grad_log_p: bool,
) -> (na::SMatrix<Float, DIM, DIM>, Option<[Float; DIM]>)
where
    Const<DIM>: Doubleable,
    TwoTimesConst<DIM>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, DIM>,
{
    let log_transition = forward.log_transition();
    let mut forward_1 = child_input_forward_data(log_p, log_transition.as_view());

    for (mut col, wt) in std::iter::zip(forward_1.column_iter_mut(), cotangent_vector.into_iter()) {
        col *= wt;
    }

    let grad_log_p = if compute_grad_log_p {
        Some(d_broadcast_vjp(forward_1.as_view()))
    } else {
        None
    };

    let grad_rate = d_rate_log_transition_vjp(forward_1.as_view(), distance, forward);

    (grad_rate, grad_log_p)
}

pub fn d_child_input_param<const DIM: usize>(
    cotangent_vector: [Float; DIM],
    distance: Float,
    param: &ParamData<DIM>,
    log_p: &[Float; DIM],
    forward: &LogTransitionForwardData<DIM>,
    compute_grad_log_p: bool,
) -> (na::SMatrix<Float, DIM, DIM>, Option<[Float; DIM]>) {
    let log_transition = forward.log_transition();
    let reverse_1 = {
        let mut forward_3 = child_input_forward_data(log_p, log_transition.as_view());
        for (mut col, wt) in
            std::iter::zip(forward_3.column_iter_mut(), cotangent_vector.into_iter())
        {
            col *= wt;
        }
        forward_3
    };

    let grad_log_p = if compute_grad_log_p {
        Some(d_broadcast_vjp(reverse_1.as_view()))
    } else {
        None
    };

    let forward_2 = forward.step_2;
    let reverse_2 = d_map_ln_vjp(reverse_1.as_view(), forward_2.as_view());

    let grad_rate = d_transition_mcgibbon_pande(reverse_2.as_view(), distance, param);

    (grad_rate, grad_log_p)
}
