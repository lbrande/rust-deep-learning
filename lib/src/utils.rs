use mkl::cblas::*;
use libc::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::StandardNormal;

use std::isize;
use std::iter::*;
use std::ops::*;

pub fn dcopy(n: i32, x: &[f64], y: &mut [f64]) {
    unsafe {
        cblas_dcopy(n, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

#[derive(Debug, PartialEq)]
pub struct Vector<T> {
    nrows: isize,
    data: Vec<T>,
}

impl<T> Vector<T> {
    pub fn from_fn(nrows: isize, f: &Fn(isize) -> T) -> Self {
        if nrows < 0 {
            panic!("nrows can't be negative");
        }
        Self {
            nrows,
            data: (0..nrows).map(f).collect(),
        }
    }

    pub fn from_data(data: Vec<T>) -> Self {
        Self {
            nrows: data.len() as isize,
            data,
        }
    }

    pub fn nrows(&self) -> isize {
        self.nrows
    }

    pub fn shape(&self) -> (isize) {
        (self.nrows)
    }

    fn fix_start(&self, start: isize) -> usize {
        if start < 0 {
            (start + self.nrows) as usize
        } else {
            start as usize
        }
    }

    fn fix_end(&self, end: isize) -> usize {
        if end <= 0 {
            (end + self.nrows) as usize
        } else {
            end as usize
        }
    }
}

impl<T: Copy> Vector<T> {
    pub fn from_val(nrows: isize, val: T) -> Self {
        if nrows < 0 {
            panic!("nrows can't be negative");
        }
        Self {
            nrows,
            data: vec![val; nrows as usize],
        }
    }

    pub fn shuffle(&mut self) {
        SliceRandom::shuffle(&mut self.data[..], &mut thread_rng());
    }
}

impl<T> Index<isize> for Vector<T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        &self.data[self.fix_start(index)]
    }
}
impl<T> IndexMut<isize> for Vector<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = self.fix_start(index);
        &mut self.data[index]
    }
}

impl<T> Index<Range<isize>> for Vector<T> {
    type Output = [T];

    fn index(&self, range: Range<isize>) -> &Self::Output {
        &self.data[self.fix_start(range.start)..self.fix_end(range.end)]
    }
}
impl<T> IndexMut<Range<isize>> for Vector<T> {
    fn index_mut(&mut self, range: Range<isize>) -> &mut Self::Output {
        let start = self.fix_start(range.start);
        let end = self.fix_end(range.end);
        &mut self.data[start..end]
    }
}

impl<T> Index<RangeFrom<isize>> for Vector<T> {
    type Output = [T];

    fn index(&self, range: RangeFrom<isize>) -> &Self::Output {
        &self.data[self.fix_start(range.start)..]
    }
}
impl<T> IndexMut<RangeFrom<isize>> for Vector<T> {
    fn index_mut(&mut self, range: RangeFrom<isize>) -> &mut Self::Output {
        let start = self.fix_start(range.start);
        &mut self.data[start..]
    }
}

impl<T> Index<RangeTo<isize>> for Vector<T> {
    type Output = [T];

    fn index(&self, range: RangeTo<isize>) -> &Self::Output {
        &self.data[..self.fix_end(range.end)]
    }
}
impl<T> IndexMut<RangeTo<isize>> for Vector<T> {
    fn index_mut(&mut self, range: RangeTo<isize>) -> &mut Self::Output {
        let end = self.fix_end(range.end);
        &mut self.data[..end]
    }
}

impl<T> Index<RangeFull> for Vector<T> {
    type Output = [T];

    fn index(&self, _: RangeFull) -> &Self::Output {
        &self.data[..]
    }
}
impl<T> IndexMut<RangeFull> for Vector<T> {
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        &mut self.data[..]
    }
}

#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    nrows: isize,
    ncols: isize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn from_fn(nrows: isize, ncols: isize, f: &Fn(isize) -> T) -> Self {
        if nrows < 0 || ncols < 0 {
            panic!("nrows and ncols can't be negative");
        }
        Self {
            nrows,
            ncols,
            data: (0..nrows * ncols).map(f).collect(),
        }
    }

    pub fn from_data(data: Vec<T>, nrows: isize) -> Self {
        if nrows < 0 {
            panic!("nrows can't be negative");
        }
        if data.len() % nrows as usize != 0 {
            panic!("len not divisible by nrows")
        }
        Self {
            nrows,
            ncols: (data.len() / nrows as usize) as isize,
            data,
        }
    }

    pub fn nrows(&self) -> isize {
        self.nrows
    }

    pub fn ncols(&self) -> isize {
        self.ncols
    }

    pub fn shape(&self) -> (isize, isize) {
        (self.nrows, self.ncols)
    }

    fn fix_index(&self, mut start: (isize, isize)) -> usize {
        if start.0 >= self.nrows {
            panic!("row out of bounds");
        }
        if start.0 < 0 {
            start.0 += self.nrows;
        }
        if start.1 < 0 {
            start.1 += self.ncols;
        }
        start.1 as usize * self.nrows as usize + start.0 as usize
    }
}

impl<T: Copy> Matrix<T> {
    pub fn from_val(nrows: isize, ncols: isize, val: T) -> Self {
        if nrows < 0 || ncols < 0 {
            panic!("nrows and ncols can't be negative");
        }
        Self {
            nrows,
            ncols,
            data: vec![val; (nrows * ncols) as usize],
        }
    }
}

impl<T> Index<(isize, isize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (isize, isize)) -> &Self::Output {
        &self.data[self.fix_index(index)]
    }
}
impl<T> IndexMut<(isize, isize)> for Matrix<T> {
    fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
        let index = self.fix_index(index);
        &mut self.data[index]
    }
}

/*impl Add<Self> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn add(self, other: Self) -> Self::Output {
        if self.shape() != other.shape() {
            panic!("shapes don't match");
        }
        let mut result = Matrix::from_val(self.nrows, self.ncols, 0.0);
        unsafe {
            daxpy(self.data.len() as i32, 1.0, &self.data[..], 1, &mut result.data[..], 1);
        }
        result
    }
}*/

pub fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
