use crate::data_types::*;
use crate::forward::*;

const DIM: Id = Entry::DIM;

pub struct BackwardData<const DIM: usize> {
    pub grad_log_p: [Float; DIM],
    /* TODO remove: we probably don't need to store grad_rates */
    //pub grad_rate: na::SMatrix<Float, DIM, DIM>,
}

pub fn softmax<const N: usize>(x: &[Float; N]) -> [Float; N] {
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
    let mut result: na::SVector<Float, N> = x.into();
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

/* TODO! stability */
pub fn d_exp_jvp(
    direction: na::SMatrixView<Float, DIM, DIM>,
    argument: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    const N: usize = DIM;
    const TWICE_N: usize = 2 * DIM;

    let mut block_triangular = na::SMatrix::<Float, TWICE_N, TWICE_N>::zeros();

    block_triangular.index_mut((..N, ..N)).copy_from(&argument);
    block_triangular.index_mut((N.., N..)).copy_from(&argument);
    block_triangular.index_mut((..N, N..)).copy_from(&direction);

    let exp_combined = block_triangular.exp();

    /* TODO accept &mut result and use copy_from? */
    let dexp: na::SMatrix<Float, N, N> = exp_combined.fixed_view::<N, N>(0, N).clone_owned();
    dexp
}

pub fn d_exp_vjp(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    argument: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /* exp([[X^T, 0], [w, X^T]]) = [[exp(X^T), 0], [w (D_X exp), exp(X^T)]] */
    const N: usize = DIM;
    const TWICE_N: usize = 2 * DIM;

    let mut block_triangular = na::SMatrix::<Float, TWICE_N, TWICE_N>::zeros();

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

    /* TODO accept &mut result and use copy_from? */
    let dexp: na::SMatrix<Float, N, N> = exp_combined.fixed_view::<N, N>(N, 0).clone_owned();
    dexp
}

/* VJP same as JVP (this is a diagonal map) */
fn d_map_ln_jvp<const DIM: usize>(
    direction: na::SMatrixView<Float, DIM, DIM>,
    argument: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /* D_map_ln(argument, direction) = map_recip (argument) \odot direction */
    let mut rec = argument.map(Float::recip);
    rec.component_mul_assign(&direction);
    rec
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

fn d_rate_log_transition_jvp(
    direction: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
    forward: &LogTransitionForwardData<DIM>,
) -> RateType {
    /* result := D_R(log_transition(R, t)) at R=rate, evaluated on 'direction'.
    Let D = D_rate (since t is constant, we don't care about D_t).
    (rate, t, step_2) = forward
    step_1 = forward.step_1()

    backward_1 = D_mul(argument=(rate, t), direction = direction) = t * direction
    backward_2 = D_exp(argument = step_1, direction = backward_1)
    result = D_map_ln(argument = step_2, direction = backward_2) */
    let forward_1 = forward.step_1;
    let forward_2 = forward.step_2;

    let backward_1 = direction * distance;
    let backward_2 = d_exp_jvp(backward_1.as_view(), forward_1.as_view());
    let result = d_map_ln_jvp(backward_2.as_view(), forward_2.as_view());
    result
}

fn d_rate_log_transition_vjp(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
    forward: &LogTransitionForwardData<DIM>,
) -> RateType {
    /* log_tr*|y (w) = mul*|rx @ exp*|exp(rx) @ map_ln*|y @ w
     */
    let forward_1 = forward.step_1;
    let forward_2 = forward.step_2;

    let reverse_1 = d_map_ln_vjp(cotangent_vector, forward_2.as_view());
    let reverse_2 = d_exp_vjp(reverse_1.as_view(), forward_1.as_view());
    let result = reverse_2 * distance;
    result
}

/* TODO! rewrite */
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

pub fn d_child_input_vjp(
    cotangent_vector: [Float; DIM],
    distance: Float,
    log_p: &[Float; DIM],
    forward: &LogTransitionForwardData<DIM>,
    compute_grad_log_p: bool,
) -> (na::SMatrix<Float, DIM, DIM>, Option<[Float; DIM]>) {
    let log_transition = forward.log_transition();
    let forward_1 = child_input_forward_data(log_p, log_transition.as_view());
    let grad_log_p = if compute_grad_log_p {
        Some(d_broadcast_vjp(forward_1.as_view()))
    } else {
        None
    };
    /* TODO! check if this is correct (especially the use of x_column) */
    let reverse_1_iterator = (0..DIM)
        .map(|i| {
            let x_column = forward_1.column(i);
            (0..DIM).map(move |k| &cotangent_vector[i] * x_column[k])
        })
        .flatten();
    let d_log_transition_result = na::SMatrix::<Float, DIM, DIM>::from_iterator(reverse_1_iterator);

    let grad_rate = d_rate_log_transition_vjp(d_log_transition_result.as_view(), distance, forward);

    (grad_rate, grad_log_p)
}
