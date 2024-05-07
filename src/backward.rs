use std::cmp;

use crate::data_types::*;
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

/* It can't be hard to make this generic over N, right? */

fn d_exp_4x4(
    argument: na::SMatrixView<Float, 4, 4>,
    direction: na::SMatrixView<Float, 4, 4>,
) -> na::SMatrix<Float, 4, 4> {
    const N: usize = 4;
    let mut block_triangular = na::SMatrix::<Float, { 2 * N }, { 2 * N }>::zeros();

    block_triangular.index_mut((..N, ..N)).copy_from(&argument);
    block_triangular.index_mut((N.., N..)).copy_from(&argument);
    block_triangular.index_mut((..N, N..)).copy_from(&direction);

    let exp_combined = block_triangular.exp();

    let dexp: na::SMatrix<Float, N, N> = exp_combined.fixed_view::<N, N>(0, N).clone_owned();
    dexp
}

/* Wrong! */

pub fn d_exp<const N: usize>(
    argument: na::SMatrix<Float, N, N>,
    direction: na::SMatrix<Float, N, N>,
) -> na::SMatrix<Float, N, N>
where
    [(); {N + N}]:,
    na::Const<{N + N}> : ToTypenum,
na::Const<{N + N}> : DimMin<na::Const<{N + N}>, Output = na::Const<{N + N}>>,
    
{
    let mut block_triangular = na::SMatrix::<Float, { N + N }, { N + N }>::zeros();
    block_triangular.index_mut((..N, ..N)).copy_from(&argument);
    block_triangular.index_mut((N.., N..)).copy_from(&argument);
    block_triangular.index_mut((..N, N..)).copy_from(&direction);
    let exp_combined = block_triangular.exp();
    /* TODO accept &mut result and use copy_from? */
    let mut dexp = na::SMatrix::<
        Float,
        N,
        N,
    >::zeros();
    dexp.copy_from(&exp_combined.index((..N, N..)));
    dexp
}
