use std::cmp;

use crate::data_types::*;
use crate::forward::*;
use crate::na::{Const, DefaultAllocator, Dim, DimAdd, DimMin, DimName, DimSum, ToTypenum}; //SMatrix, SMatrixView

pub fn softmax<const N: usize>(x: &[Float; N]) -> [Float; N] {
    let mut result = [0 as Float; N];
    let x_max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    for i in (0..N) {
        result[i] -= x[i] - x_max;
    }
    for i in (0..N) {
        result[i] = x[i].exp();
    }
    let scale = result.iter().sum::<Float>().recip();
    for i in (0..N) {
        result[i] *= scale;
    }
    result
}

pub fn softmax_inplace<const N: usize>(x: &mut [Float; N]) {
    let x_max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    for i in (0..N) {
        x[i] -= x_max;
    }
    for i in (0..N) {
        x[i] = x[i].exp();
    }
    let scale = x.iter().sum::<Float>().recip();
    for i in (0..N) {
        x[i] *= scale
    }
}

/* TODO! optimize 'direction' away by replacing it with an index (i, j) */
/* TODO stability */
pub fn d_exp(
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

pub fn d_map_ln(
    argument: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
    direction: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }> {
    /* D_map_ln(argument, direction) = map_recip (argument) \odot direction */
    let mut rec = argument.map(Float::recip);
    rec.component_mul_assign(&direction);
    rec
}

pub fn d_rate_log_transition(
    forward: &LogTransitionForwardData<{ Entry::DIM }>,
    direction: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) -> RateType {
    /* result := D_R(log_transition(R, t)) at R=rate, evaluated on 'direction'.
    Let D = D_rate (since t is constant, we don't care about D_t).
    (rate, t, step_2) = forward
    step_1 = forward.step_1()

    backward_1 = D_mul(argument=(rate, t), direction = direction) = t * direction
    backward_2 = D_exp(argument = step_1, direction = backward_1)
    result = D_map_ln(argument = step_2, direction = backward_2) */
    let distance = forward.distance;
    let step_1 = forward.step_1;
    let step_2 = forward.step_2;

    let backward_1 = direction * distance;
    let backward_2 = d_exp(step_1.as_view(), backward_1.as_view());
    let result = d_map_ln(step_2.as_view(), backward_2.as_view());
    result
}
