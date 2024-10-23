use std::iter::Sum;

use logsumexp::LogSumExp;

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
}
