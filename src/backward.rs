use crate::data_types::*;
use crate::forward::*;

use na::{Const, DefaultAllocator};

pub struct BackwardData<const DIM: usize> {
    pub grad_log_p: [Float; DIM],
    /* TODO remove: we probably don't need to store grad_rates */
    //pub grad_rate: na::SMatrix<Float, DIM, DIM>,
}

fn softmax<const N: usize>(x: &[Float; N]) -> [Float; N] {
    let mut result = [0 as Float; N];
    let x_max = *x
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .expect("Iterator cannot be empty");

    for i in 0..N {
        result[i] = x[i] - x_max;
    }
    for i in 0..N {
        result[i] = result[i].exp();
    }
    let scale = result.iter().sum::<Float>().recip();
    for i in 0..N {
        result[i] *= scale;
    }
    result
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
    DefaultAllocator: ViableAllocator<N>,
{
    /* exp([[X^T, 0], [w, X^T]]) = [[exp(X^T), 0], [w (D_X exp), exp(X^T)]] */
    let mut block_triangular = MatrixNNAllocated::<TwoTimesConst<N>>::zeros();
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
    DefaultAllocator: ViableAllocator<DIM>,
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

/* TODO! rewrite: broadcast (let broadcast, broadcast.column_iter.copy_from) and add */
/* TODO extract iterator, use it to compute d_broadcast (d_lse) */
fn child_input_forward_data<const DIM: usize>(
    log_p: &[Float; DIM],
    /* TODO get by value */
    log_transition: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /* This is softmax(log_p[:, -1] + log_transition, dim=0) */
    let iter = (0..DIM).map(|a| {
        let mut col_a: na::SVector<Float, DIM> = log_transition.column(a).into();
        (0..DIM).for_each(|b| (col_a[b] += log_p[b]));
        na::SVector::from(softmax_na(col_a.as_view()))
    });

    let mut result = na::SMatrix::<Float, DIM, DIM>::zeros();
    for (a, column) in iter.enumerate() {
        result.column_mut(a).copy_from(&column);
    }

    result
}

fn d_broadcast_vjp<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
) -> [Float; DIM] {
    let mut result_iter = cotangent_vector.row_iter().map(|row| row.iter().sum());
    //na::SVector::<Float, DIM>::from_iterator(result_iter)
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
    DefaultAllocator: ViableAllocator<DIM>,
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
