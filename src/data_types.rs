use std::{collections::HashMap, convert::TryFrom, iter::Sum};

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
{
    const EPS_LOG : Self;
    const EPS_DIV: Self; // Minimum value for sqrt_pi
    fn logsumexp<'a, I : Iterator<Item = &'a Self>>(iter: I) -> Self;
    fn from_f64(f: f64) -> Self;
}
impl FloatTrait for f32 {
    fn logsumexp<'a, I : Iterator<Item = &'a Self>>(iter: I) -> Self {
        LogSumExp::ln_sum_exp(iter)
    }
    fn from_f64(f: f64) -> Self {
        f as f32
    }
    const EPS_LOG : Self = 1e-15;
    const EPS_DIV: Self = 1e-10;
}
impl FloatTrait for f64 {
    fn logsumexp<'a, I : Iterator<Item = &'a Self>>(iter: I) -> Self {
        LogSumExp::ln_sum_exp(iter)
    }
    fn from_f64(f: f64) -> Self {
        f
    }
    const EPS_LOG : Self = 1e-100;
    const EPS_DIV: Self = 1e-10;
}

pub trait EntryTrait: Sized + Copy {
    const TOTAL: usize;
    const CHARS: usize;
}
pub trait ResidueTrait:
    EntryTrait + TryFrom<char, Error = FelsensteinError> + na::Scalar + std::marker::Sync
{
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        input.chars().map(Self::try_from)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Residue {
    A,
    C,
    G,
    T,
    None,
    R,
    Y,
    S,
    W,
    K,
    M,
    B,
    D,
    H,
    V,
}
impl Residue {
    pub const U: Self = Self::T;
}

impl TryFrom<char> for Residue {
    type Error = FelsensteinError;
    fn try_from(char: char) -> Result<Self, Self::Error> {
        use Residue::*;
        let res: Residue = match char.to_ascii_uppercase() {
            'A' => A,
            'C' => C,
            'G' => G,
            'T' => T,
            'R' => R,
            'Y' => Y,
            'S' => S,
            'W' => W,
            'K' => K,
            'M' => M,
            'B' => B,
            'D' => D,
            'H' => H,
            'V' => V,
            'U' => Residue::U,
            '-' => Residue::None,
            '.' => Residue::None,
            'N' => Residue::None,
            _ => return Err(FelsensteinError::INVALID_CHAR),
        };
        Ok(res)
    }
}
impl Into<char> for Residue {
    fn into(self) -> char {
        use Residue::*;
        match self {
            A => 'A',
            C => 'C',
            G => 'G',
            T => 'T',
            Self::None => 'N',
            R => 'R',
            Y => 'Y',
            S => 'S',
            W => 'W',
            K => 'K',
            M => 'M',
            B => 'B',
            D => 'D',
            H => 'H',
            V => 'V',
        }
    }
}

impl Into<String> for Residue {
    fn into(self) -> String {
        let ch: char = self.into();
        ch.into()
    }
}

impl EntryTrait for Residue {
    const TOTAL: usize = 15;
    const CHARS: usize = 1;
}

impl ResidueTrait for Residue {}

pub trait Distribution<E, F>
where
    Self: std::marker::Sync,
    E: EntryTrait,
    F: FloatTrait,
{
    type Value;
    fn log_p(&self, entry: E) -> Self::Value;
}

pub struct Dist<F, const DIM: usize> {
    pub log_p: HashMap<String, na::SVector<F, DIM>>,
}
impl<R, F, const DIM: usize> Distribution<R, F> for Dist<F, DIM>
where
    R: ResidueTrait + Into<String>,
    F: FloatTrait,
{
    type Value = na::SVector<F, DIM>;
    fn log_p(&self, entry: R) -> Self::Value {
        let code: String = entry.into();
        self.log_p.get(&code).unwrap().clone_owned()
    }
}
#[derive(Clone)]
pub struct DistNoGaps<const N: usize> {
    pub p_none: Option<na::SVector<Float, N>>,
}
impl Distribution<Residue, Float> for DistNoGaps<4> {
    type Value = na::SVector<Float, 4>;
    fn log_p(&self, entry: Residue) -> Self::Value {
        use Residue::*;
        let prob: na::SVector<Float, 4> = match entry {
            A => na::SVector::<Float, 4>::new(1.0, 0.0, 0.0, 0.0),
            C => na::SVector::<Float, 4>::new(0.0, 1.0, 0.0, 0.0),
            G => na::SVector::<Float, 4>::new(0.0, 0.0, 1.0, 0.0),
            T => na::SVector::<Float, 4>::new(0.0, 0.0, 0.0, 1.0),
            //Residue::None => na::SVector::<Float, 4>::new(0.25, 0.25, 0.25, 0.25),
            Residue::None => match self.p_none {
                Some(vector) => vector.clone_owned(),
                Option::<na::SVector<Float, 4>>::None => {
                    na::SVector::<Float, 4>::new(0.25, 0.25, 0.25, 0.25)
                }
            },
            R => na::SVector::<Float, 4>::new(0.5, 0.0, 0.5, 0.0),
            Y => na::SVector::<Float, 4>::new(0.0, 0.5, 0.0, 0.5),
            S => na::SVector::<Float, 4>::new(0.0, 0.5, 0.5, 0.0),
            W => na::SVector::<Float, 4>::new(0.5, 0.0, 0.0, 0.5),
            K => na::SVector::<Float, 4>::new(0.0, 0.0, 0.5, 0.5),
            M => na::SVector::<Float, 4>::new(0.5, 0.5, 0.0, 0.0),
            B => na::SVector::<Float, 4>::new(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            D => na::SVector::<Float, 4>::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
            H => na::SVector::<Float, 4>::new(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0),
            V => na::SVector::<Float, 4>::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        };
        prob.map(Float::ln)
    }
}

#[derive(Clone, Copy)]
pub struct DistGaps {}
impl Distribution<Residue, Float> for DistGaps {
    type Value = na::SVector<Float, 5>;
    fn log_p(&self, entry: Residue) -> Self::Value {
        use Residue::*;
        let prob: na::SVector<Float, 5> = match entry {
            A => na::SVector::<Float, 5>::new(1.0, 0.0, 0.0, 0.0, 0.0),
            C => na::SVector::<Float, 5>::new(0.0, 1.0, 0.0, 0.0, 0.0),
            G => na::SVector::<Float, 5>::new(0.0, 0.0, 1.0, 0.0, 0.0),
            T => na::SVector::<Float, 5>::new(0.0, 0.0, 0.0, 1.0, 0.0),
            Residue::None => na::SVector::<Float, 5>::new(0.0, 0.0, 0.0, 0.0, 1.0),
            R => na::SVector::<Float, 5>::new(0.5, 0.0, 0.5, 0.0, 0.0),
            Y => na::SVector::<Float, 5>::new(0.0, 0.5, 0.0, 0.5, 0.0),
            S => na::SVector::<Float, 5>::new(0.0, 0.5, 0.5, 0.0, 0.0),
            W => na::SVector::<Float, 5>::new(0.5, 0.0, 0.0, 0.5, 0.0),
            K => na::SVector::<Float, 5>::new(0.0, 0.0, 0.5, 0.5, 0.0),
            M => na::SVector::<Float, 5>::new(0.5, 0.5, 0.0, 0.0, 0.0),
            B => na::SVector::<Float, 5>::new(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            D => na::SVector::<Float, 5>::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            H => na::SVector::<Float, 5>::new(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0),
            V => na::SVector::<Float, 5>::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0),
        };
        prob.map(Float::ln)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResiduePair<R>(pub R, pub R);

impl<R> EntryTrait for ResiduePair<R>
where
    R: EntryTrait,
{
    const TOTAL: usize = R::TOTAL * R::TOTAL;
    const CHARS: usize = R::CHARS * 2;
}

impl<R> Into<String> for ResiduePair<R>
where
    R: Into<String>,
{
    fn into(self) -> String {
        let mut res: String = self.0.into();
        let right: String = self.1.into();
        res.push_str(&right);
        res
    }
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

impl FelsensteinError {
    pub const INVALID_CHAR: Self =
        Self::DeserializationError("Invalid character in residue sequence");
    /*
    pub fn invalid_residue(s: &str) -> Self {
        Self::DeserializationError(format!("Invalid residue:{}", s))
    }*/
}