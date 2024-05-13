use crate::data_types::*;
use crate::forward::*;

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

pub fn softmax_na<const N: usize>(x: na::SVectorView<Float, N>) -> na::SVector<Float, N> {
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

/* TODO! optimize 'direction' away by replacing it with an index (i, j) */
/* TODO stability */
pub fn d_exp_jvp(
    argument: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
    direction: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }> {
    const N: usize = Entry::DIM;
    const TWICE_N: usize = 2 * Entry::DIM;

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
    argument: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
    cotangent_vector: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }> {
    /* exp([[X^T, 0], [w, X^T]]) = [[exp(X^T), 0], [w (D_X exp), exp(X^T)]] */
    const N: usize = Entry::DIM;
    const TWICE_N: usize = 2 * Entry::DIM;

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
fn d_map_ln_jvp(
    argument: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
    direction: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }> {
    /* D_map_ln(argument, direction) = map_recip (argument) \odot direction */
    let mut rec = argument.map(Float::recip);
    rec.component_mul_assign(&direction);
    rec
}

fn d_map_ln_vjp<const DIM: usize>(
    argument: na::SMatrixView<Float, DIM, DIM>,
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
) -> na::SMatrix<Float, DIM, DIM> {
    /* If w = \sum w_kl dy_kl, then
    map_ln^*(w) = \sum_kl (w_kl / x_kl) dx_kl = map_recip(x) \odot w */
    let mut rec = argument.map(Float::recip);
    rec.component_mul_assign(&cotangent_vector);
    rec
}

fn d_rate_log_transition_jvp(
    forward: &LogTransitionForwardData<{ Entry::DIM }>,
    distance: Float,
    direction: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> RateType {
    /* result := D_R(log_transition(R, t)) at R=rate, evaluated on 'direction'.
    Let D = D_rate (since t is constant, we don't care about D_t).
    (rate, t, step_2) = forward
    step_1 = forward.step_1()

    backward_1 = D_mul(argument=(rate, t), direction = direction) = t * direction
    backward_2 = D_exp(argument = step_1, direction = backward_1)
    result = D_map_ln(argument = step_2, direction = backward_2) */
    let step_1 = forward.step_1;
    let step_2 = forward.step_2;

    let backward_1 = direction * distance;
    let backward_2 = d_exp_jvp(step_1.as_view(), backward_1.as_view());
    let result = d_map_ln_jvp(step_2.as_view(), backward_2.as_view());
    result
}

fn d_rate_log_transition_vjp(
    forward: &LogTransitionForwardData<{ Entry::DIM }>,
    distance: Float,
    cotangent_vector: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> RateType {
    /* log_tr*|y (w) = mul*|rx @ exp*|exp(rx) @ map_ln*|y @ w
     */
    let forward_1 = forward.step_1;
    let forward_2 = forward.step_2;

    let reverse_1 = d_map_ln_vjp(forward_2.as_view(), cotangent_vector);
    let reverse_2 = d_exp_vjp(forward_1.as_view(), reverse_1.as_view());
    let result = reverse_2 * distance;
    result
}

/* TODO! rewrite */
/* TODO extract iterator, use it to compute d_broadcast (d_lse) */
fn child_input_forward_data(
    log_p: &[Float; Entry::DIM],
    /* TODO get by value */
    log_transition: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }> {
    let iter = (0..Entry::DIM).map(|a| {
        let mut col_a: na::SVector<Float, { Entry::DIM }> = log_transition.column(a).into();
        (0..{ Entry::DIM }).for_each(|b| (col_a[b] += log_p[b]));
        na::SVector::from(softmax_na(col_a.as_view()))
    });

    let mut result: na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }> = log_transition.into();
    for (a, column) in iter.enumerate() {
        result.column_mut(a).copy_from(&column);
    }

    result
}

fn d_broadcast_vjp<const DIM: usize>(
    cotangent_vector: na::SMatrixView<Float, DIM, DIM>,
) -> na::SVector<Float, DIM> {
    let result_iter = cotangent_vector.row_iter().map(|row| row.iter().sum());
    na::SVector::<Float, DIM>::from_iterator(result_iter)
}

fn d_child_input_vjp(
    id: Id,
    forward: &ForwardData<{ Entry::DIM }>,
    //backward: &BackwardData<DIM>,
    cotangent_vector: [Float; Entry::DIM],
    distance: &[Float],
    log_p: &[Float; Entry::DIM],
    log_transition: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> (
    na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>,
    na::SVector<Float, { Entry::DIM }>,
) {
    let forward_1 = child_input_forward_data(log_p, log_transition);
    /* TODO! check if this is correct (especially the use of x_column) */
    let reverse_1_iterator = (0..Entry::DIM)
        .map(|i| {
            let x_column = forward_1.column(i);
            (0..Entry::DIM).map(move |k| &cotangent_vector[i] * x_column[k])
        })
        .flatten();
    let d_log_transition_result =
        na::SMatrix::<Float, { Entry::DIM }, { Entry::DIM }>::from_iterator(reverse_1_iterator);
    let d_log_p_result = d_broadcast_vjp(forward_1.as_view());

    let d_rate_result = d_rate_log_transition_vjp(
        &forward.log_transition[id],
        distance[id],
        d_log_transition_result.as_view(),
    );

    (d_rate_result, d_log_p_result)
}
