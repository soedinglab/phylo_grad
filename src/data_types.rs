use std::{collections::HashMap, convert::TryFrom};

use na::{allocator::Allocator, Const, DimAdd, DimMin, DimName, ToTypenum};
use num_enum::{IntoPrimitive, TryFromPrimitive};

pub type Float = f64;
//pub type ColumnId = usize;
pub type Id = usize;

pub const EPS_DIV: f64 = 1e-10;
pub const EPS_LOG: f64 = 1e-100;

pub trait FloatTrait
where
    Self: num_traits::Float
        + std::ops::AddAssign
        + std::ops::MulAssign
        + na::Scalar
        + std::marker::Sync,
{
}
impl<F> FloatTrait for F where
    F: num_traits::Float
        + std::ops::AddAssign
        + std::ops::MulAssign
        + na::Scalar
        + std::marker::Sync
{
}

pub trait EntryTrait: Sized + Copy {
    const TOTAL: usize;
    const CHARS: usize;
}
pub trait ResidueTrait: EntryTrait + TryFrom<char> + na::Scalar + std::marker::Sync {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>>;
}

/* TODO! either remove the u8-stuff or make it work */
#[derive(Debug, Copy, Clone, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
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

impl ResidueTrait for Residue {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        input.chars().map(Self::try_from)
    }
}

pub trait Distribution<E, F, const DIM: usize>: std::marker::Sync
where
    E: EntryTrait,
    F: FloatTrait,
{
    fn log_p(&self, entry: E) -> na::SVector<F, DIM>;
}

pub struct Dist<F, const DIM: usize> {
    pub log_p: HashMap<String, na::SVector<F, DIM>>,
}
impl<R, F, const DIM: usize> Distribution<R, F, DIM> for Dist<F, DIM>
where
    R: ResidueTrait + Into<String>,
    F: FloatTrait,
{
    fn log_p(&self, entry: R) -> na::SVector<F, DIM> {
        let code: String = entry.into();
        self.log_p.get(&code).unwrap().clone_owned()
    }
}
#[derive(Clone)]
pub struct DistNoGaps {
    pub p_none: Option<na::SVector<f64, 4>>,
}
impl Distribution<Residue, f64, 4> for DistNoGaps {
    fn log_p(&self, entry: Residue) -> na::SVector<f64, 4> {
        use Residue::*;
        let prob: na::SVector<f64, 4> = match entry {
            A => na::SVector::<f64, 4>::new(1.0, 0.0, 0.0, 0.0),
            C => na::SVector::<f64, 4>::new(0.0, 1.0, 0.0, 0.0),
            G => na::SVector::<f64, 4>::new(0.0, 0.0, 1.0, 0.0),
            T => na::SVector::<f64, 4>::new(0.0, 0.0, 0.0, 1.0),
            //Residue::None => na::SVector::<f64, 4>::new(0.25, 0.25, 0.25, 0.25),
            Residue::None => match self.p_none {
                Some(vector) => vector.clone_owned(),
                Option::<na::SVector<f64, 4>>::None => {
                    na::SVector::<f64, 4>::new(0.25, 0.25, 0.25, 0.25)
                }
            },
            R => na::SVector::<f64, 4>::new(0.5, 0.0, 0.5, 0.0),
            Y => na::SVector::<f64, 4>::new(0.0, 0.5, 0.0, 0.5),
            S => na::SVector::<f64, 4>::new(0.0, 0.5, 0.5, 0.0),
            W => na::SVector::<f64, 4>::new(0.5, 0.0, 0.0, 0.5),
            K => na::SVector::<f64, 4>::new(0.0, 0.0, 0.5, 0.5),
            M => na::SVector::<f64, 4>::new(0.5, 0.5, 0.0, 0.0),
            B => na::SVector::<f64, 4>::new(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            D => na::SVector::<f64, 4>::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
            H => na::SVector::<f64, 4>::new(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0),
            V => na::SVector::<f64, 4>::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        };
        prob.map(f64::ln)
    }
}

#[derive(Clone)]
pub struct DistGaps {}
impl Distribution<Residue, f64, 5> for DistGaps {
    fn log_p(&self, entry: Residue) -> na::SVector<f64, 5> {
        use Residue::*;
        let prob: na::SVector<f64, 5> = match entry {
            A => na::SVector::<f64, 5>::new(1.0, 0.0, 0.0, 0.0, 0.0),
            C => na::SVector::<f64, 5>::new(0.0, 1.0, 0.0, 0.0, 0.0),
            G => na::SVector::<f64, 5>::new(0.0, 0.0, 1.0, 0.0, 0.0),
            T => na::SVector::<f64, 5>::new(0.0, 0.0, 0.0, 1.0, 0.0),
            Residue::None => na::SVector::<f64, 5>::new(0.0, 0.0, 0.0, 0.0, 1.0),
            R => na::SVector::<f64, 5>::new(0.5, 0.0, 0.5, 0.0, 0.0),
            Y => na::SVector::<f64, 5>::new(0.0, 0.5, 0.0, 0.5, 0.0),
            S => na::SVector::<f64, 5>::new(0.0, 0.5, 0.5, 0.0, 0.0),
            W => na::SVector::<f64, 5>::new(0.5, 0.0, 0.0, 0.5, 0.0),
            K => na::SVector::<f64, 5>::new(0.0, 0.0, 0.5, 0.5, 0.0),
            M => na::SVector::<f64, 5>::new(0.5, 0.5, 0.0, 0.0, 0.0),
            B => na::SVector::<f64, 5>::new(0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            D => na::SVector::<f64, 5>::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
            H => na::SVector::<f64, 5>::new(1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0),
            V => na::SVector::<f64, 5>::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0),
        };
        prob.map(f64::ln)
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

impl<F, R, const DIM: usize, const DIM_SQ: usize> Distribution<ResiduePair<R>, F, DIM_SQ>
    for Dist<F, DIM>
where
    F: FloatTrait,
    R: ResidueTrait,
    Dist<F, DIM>: Distribution<R, F, DIM>,
    Const<DIM>: na::DimMul<Const<DIM>, Output = Const<DIM_SQ>>,
{
    fn log_p(&self, entry: ResiduePair<R>) -> na::SVector<F, DIM_SQ> {
        let mut result = na::SVector::<F, DIM_SQ>::zeros();
        let (first, second) = (entry.0, entry.1);
        let log_p_first = self.log_p(first);
        let log_p_second = self.log_p(second);
        for a in 0..DIM {
            for b in 0..DIM {
                result[DIM * a + b] = log_p_first[a] + log_p_second[b];
            }
        }
        result
    }
}

impl<F, R> Distribution<ResiduePair<R>, F, 16> for DistNoGaps
where
    F: FloatTrait,
    R: ResidueTrait,
    DistNoGaps: Distribution<R, F, 4>,
{
    fn log_p(&self, entry: ResiduePair<R>) -> na::SVector<F, 16> {
        let mut result = na::SVector::<F, 16>::zeros();
        let (first, second) = (entry.0, entry.1);
        let log_p_first = self.log_p(first);
        let log_p_second = self.log_p(second);
        for a in 0..4 {
            for b in 0..4 {
                result[4 * a + b] = log_p_first[a] + log_p_second[b];
            }
        }
        result
    }
}

impl<F, R> Distribution<ResiduePair<R>, F, 25> for DistGaps
where
    F: FloatTrait,
    R: ResidueTrait,
    DistGaps: Distribution<R, F, 5>,
{
    fn log_p(&self, entry: ResiduePair<R>) -> na::SVector<F, 25> {
        let mut result = na::SVector::<F, 25>::zeros();
        let (first, second) = (entry.0, entry.1);
        let log_p_first = self.log_p(first);
        let log_p_second = self.log_p(second);
        for a in 0..5 {
            for b in 0..5 {
                result[5 * a + b] = log_p_first[a] + log_p_second[b];
            }
        }
        result
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

pub type TwoTimesConst<const N: usize> = TwoTimes<Const<N>>;
pub type TwoTimes<N> = <N as DimAdd<N>>::Output;
pub trait Exponentiable: ToTypenum + DimName + DimMin<Self, Output = Self> {}
impl<T: ToTypenum + DimName + DimMin<Self, Output = Self>> Exponentiable for T {}

pub trait Doubleable
where
    Self: ToTypenum + DimAdd<Self>,
{
}
impl<T> Doubleable for T where T: ToTypenum + DimAdd<T> {}

pub trait ViableAllocator<T, const N: usize>
where
    Self: Allocator<T, TwoTimesConst<N>, TwoTimesConst<N>>
        + Allocator<T, Const<N>, Const<N>, Buffer = na::ArrayStorage<Float, N, N>>
        + Allocator<T, TwoTimesConst<N>>
        + Allocator<(usize, usize), TwoTimesConst<N>>,
    Const<N>: Doubleable,
{
}
impl<A, T, const N: usize> ViableAllocator<T, N> for A
where
    A: Allocator<T, TwoTimesConst<N>, TwoTimesConst<N>>
        + Allocator<(usize, usize), TwoTimesConst<N>>
        + Allocator<T, Const<N>, Const<N>, Buffer = na::ArrayStorage<Float, N, N>>
        + Allocator<T, TwoTimesConst<N>>,
    Const<N>: Doubleable,
{
}
