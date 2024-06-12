use std::{collections::HashMap, convert::TryFrom};

use na::{allocator::Allocator, Const, DimAdd, DimMin, DimName, ToTypenum};

pub type Float = f64;
//pub type ColumnId = usize;
//pub type Id = usize;

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
impl FloatTrait for f32 {}
impl FloatTrait for f64 {}

pub trait EntryTrait: Sized + Copy {
    const TOTAL: usize;
    const CHARS: usize;
}
pub trait ResidueTrait: EntryTrait + TryFrom<char> + na::Scalar + std::marker::Sync {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>>;
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

impl ResidueTrait for Residue {
    fn try_deserialize_string_iter(
        input: &str,
    ) -> impl Iterator<Item = Result<Self, FelsensteinError>> {
        input.chars().map(Self::try_from)
    }
}

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
pub struct DistNoGaps {
    pub p_none: Option<na::SVector<Float, 4>>,
}
impl Distribution<Residue, Float> for DistNoGaps {
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

impl<F, R, DLeft, DRight, Dim> Distribution<ResiduePair<R>, F> for (&DLeft, &DRight)
where
    F: FloatTrait,
    R: ResidueTrait,
    DLeft: Distribution<R, F, Value = na::OVector<F, Dim>>,
    DRight: Distribution<R, F, Value = na::OVector<F, Dim>>,
    Dim: DimName + Squareable,
    Squared<Dim>: DimName,
    na::DefaultAllocator: SquareableAllocator<Dim, F>,
{
    type Value = na::OVector<F, Squared<Dim>>;
    fn log_p(&self, entry: ResiduePair<R>) -> Self::Value {
        let mut result = na::OVector::<F, Squared<Dim>>::zeros();
        let (left, right) = (entry.0, entry.1);
        let log_p_left = self.0.log_p(left);
        let log_p_right = self.1.log_p(right);
        for a in 0..Dim::dim() {
            for b in 0..Dim::dim() {
                result[Dim::dim() * a + b] = log_p_left[a] + log_p_right[b];
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
type Squared<N> = <N as na::DimMul<N>>::Output;
pub trait Exponentiable: ToTypenum + DimName + DimMin<Self, Output = Self> {}
impl<T: ToTypenum + DimName + DimMin<Self, Output = Self>> Exponentiable for T {}

pub trait Doubleable
where
    Self: ToTypenum + DimAdd<Self>,
{
}
impl<T> Doubleable for T where T: ToTypenum + DimAdd<T> {}

trait Squareable
where
    Self: na::DimMul<Self>,
    <Self as na::DimMul<Self>>::Output: DimName,
{
}
impl<T> Squareable for T
where
    T: na::DimMul<T>,
    <T as na::DimMul<T>>::Output: DimName,
{
}

pub trait ViableAllocator<T, const N: usize>
where
    Self: Allocator<T, TwoTimesConst<N>, TwoTimesConst<N>>
        + Allocator<(usize, usize), TwoTimesConst<N>>
        + Allocator<T, Const<N>, Const<N>, Buffer = na::ArrayStorage<Float, N, N>>
        + Allocator<T, TwoTimesConst<N>>,
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

trait SquareableAllocator<N, F>
where
    N: DimName + Squareable,
    Squared<N>: DimName,
    F: FloatTrait,
    Self: Allocator<F, N> + Allocator<F, Squared<N>>,
{
}
impl<A, N, F> SquareableAllocator<N, F> for A
where
    N: DimName + Squareable,
    Squared<N>: DimName,
    F: FloatTrait,
    A: Allocator<F, N> + Allocator<F, Squared<N>>,
{
}
