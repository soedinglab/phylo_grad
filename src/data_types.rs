use std::iter::Sum;

use logsumexp::LogSumExp;

pub type Float = f64;
//pub type ColumnId = usize;
//pub type Id = usize;

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
        + nalgebra_lapack::SymmetricEigenScalar
        + numpy::Element,
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

#[derive(Debug)]
pub enum FelsensteinError {
    LogicError(&'static str),
    DeserializationError(&'static str),
}

impl std::fmt::Display for FelsensteinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Self::LogicError(str) => str,
            Self::DeserializationError(str) => str,
        };
        write!(f, "{}", str)
    }
}
impl std::error::Error for FelsensteinError {}
