use std::cmp;

use crate::data_types::*;

pub fn softmax<const N: usize>(x: &[Float; N]) -> [Float; N] {
    /* TODO accuracy? */
    let mut result = [0.0 as Float; N];
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

//fn log_transition_backward<const DIM: usize>;
