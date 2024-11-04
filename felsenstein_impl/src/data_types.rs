use std::{iter::Sum, simd::num::SimdFloat};

use logsumexp::LogSumExp;

use std::simd;

pub trait FloatTrait
where
    Self: num_traits::Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + na::Scalar
        + std::marker::Sync
        + Into<f64>
        + Sum
        + na::RealField
        + nalgebra_lapack::SymmetricEigenScalar,
{
    const EPS_LOG: Self;
    const EPS_DIV: Self; // Minimum value for sqrt_pi
    fn logsumexp<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self;
    fn from_f64(f: f64) -> Self;
    fn scalar_exp(self) -> Self;
    fn vec_exp<const N: usize>(x: &mut [Self; N]);
    fn vec_logsumexp<const N: usize>(x: &[Self; N]) -> Self;
}

impl FloatTrait for f32 {
    fn logsumexp<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        LogSumExp::ln_sum_exp(iter)
    }
    fn from_f64(f: f64) -> Self {
        f as f32
    }
    const EPS_LOG: Self = 1e-15;
    const EPS_DIV: Self = 1e-10;
    fn scalar_exp(self) -> Self {
        sleef::f32::exp_u10(self)
    }
    fn vec_exp<const N: usize>(x: &mut [Self; N]) {
        let blocks = N / 8;

        for i in 0..blocks {
            let a = simd::f32x8::from_slice(&x[i * 8..]);
            let b = sleef::f32x::exp_u10(a);
            simd::f32x8::copy_to_slice(b, &mut x[i * 8..]);
        }

        for i in blocks * 8..N {
            x[i] = x[i].scalar_exp();
        }
    }
    fn vec_logsumexp<const N: usize>(_x: &[Self; N]) -> Self {
        panic!("Not implemented");
    }
}
impl FloatTrait for f64 {
    fn logsumexp<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        LogSumExp::ln_sum_exp(iter)
    }
    fn from_f64(f: f64) -> Self {
        f
    }
    const EPS_LOG: Self = 1e-100;
    const EPS_DIV: Self = 1e-10;
    fn scalar_exp(self) -> Self {
        sleef::f64::exp_u10(self)
    }
    fn vec_exp<const N: usize>(x: &mut [Self; N]) {
        let blocks = N / 4;

        for i in 0..blocks {
            let a = simd::f64x4::from_slice(&x[i * 4..]);
            let b = sleef::f64x::exp_u10(a);
            simd::f64x4::copy_to_slice(b, &mut x[i * 4..]);
        }

        for i in blocks * 4..N {
            x[i] = x[i].scalar_exp();
        }
    }
    fn vec_logsumexp<const N: usize>(x: &[Self; N]) -> Self {
        let blocks = N / 4;

        let mut max = simd::f64x4::splat(f64::NEG_INFINITY);
        for i in 0..blocks {
            let a = simd::f64x4::from_slice(&x[i * 4..]);
            max = max.simd_max(a);
        }
        let last_elements = simd::f64x4::load_or(&x[blocks * 4..], simd::f64x4::splat(f64::NEG_INFINITY));
        max = max.simd_max(last_elements);

        let max = max.reduce_max();

        let mut sum = simd::f64x4::splat(0.0);
        for i in 0..blocks {
            let a = simd::f64x4::from_slice(&x[i * 4..]);
            let b = a - simd::f64x4::splat(max);
            let c = sleef::f64x::exp_u10(b);
            sum += c;
        }

        let last_elements = simd::f64x4::load_or_default(&x[blocks * 4..]);
        sum += sleef::f64x::exp_u10(last_elements - simd::f64x4::splat(max));

        return max + (sum.reduce_sum()).ln();
    }
}
