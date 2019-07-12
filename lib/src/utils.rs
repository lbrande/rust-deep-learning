use mkl::blas::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::StandardNormal;

use std::isize;
use std::iter::*;
use std::ops::*;

static ILLEGAL_SHAPE: &'static str = "illegal shape";
static MISSMATCHED_SHAPE: &'static str = "missmatched shape";
static MISSMATCHED_SHAPES: &'static str = "missmatched shapes";
static INDEX_OUT_OF_BOUNDS: &'static str = "index out of bounds";

#[derive(Debug, PartialEq)]
pub struct Vector<T> {
    shape: isize,
    data: Vec<T>,
}

impl<T> Vector<T> {
    pub fn from_fn(shape: isize, f: &Fn(isize) -> T) -> Self {
        Self::illegal_shape_panic(shape);
        Self {
            shape,
            data: (0..shape).map(f).collect(),
        }
    }

    pub fn from_data(shape: isize, data: Vec<T>) -> Self {
        Self::missmatched_shape_panic(shape, &data);
        Self {
            shape: data.len() as isize,
            data,
        }
    }

    pub fn shape(&self) -> isize {
        self.shape
    }

    fn fix_start(&self, start: isize) -> usize {
        if start < 0 {
            (start + self.shape) as usize
        } else {
            start as usize
        }
    }

    fn fix_end(&self, end: isize) -> usize {
        if end <= 0 {
            (end + self.shape) as usize
        } else {
            end as usize
        }
    }

    fn illegal_shape_panic(shape: isize) {
        if shape < 0 {
            panic!(ILLEGAL_SHAPE);
        }
    }

    fn missmatched_shape_panic(shape: isize, data: &[T]) {
        if shape as usize != data.len() {
            panic!(MISSMATCHED_SHAPE);
        }
    }

    fn missmatched_shapes_panic(&self, other: &Self) {
        if self.shape() != other.shape() {
            panic!(MISSMATCHED_SHAPES);
        }
    }
}

impl<T: Copy> Vector<T> {
    pub fn from_val(shape: isize, val: T) -> Self {
        Self::illegal_shape_panic(shape);
        Self {
            shape,
            data: vec![val; shape as usize],
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
    shape: (isize, isize),
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn from_fn(shape: (isize, isize), f: &Fn(isize) -> T) -> Self {
        Self::illegal_shape_panic(shape);
        Self {
            shape,
            data: (0..shape.0 * shape.1).map(f).collect(),
        }
    }

    pub fn from_data(shape: (isize, isize), data: Vec<T>) -> Self {
        Self::illegal_shape_panic(shape);
        Self::missmatched_shape_panic(shape, &data);
        Self {
            shape,
            data,
        }
    }

    pub fn shape(&self) -> (isize, isize) {
        self.shape
    }

    fn fix_index(&self, mut index: (isize, isize)) -> usize {
        self.index_out_of_bound_panic(index);
        if index.0 < 0 {
            index.0 += self.shape.0;
        }
        if index.1 < 0 {
            index.1 += self.shape.1;
        }
        index.1 as usize * self.shape.0 as usize + index.0 as usize
    }

    fn illegal_shape_panic(shape: (isize, isize)) {
        if shape.0 < 0 || shape.1 < 0 {
            panic!(ILLEGAL_SHAPE);
        }
    }

    fn missmatched_shape_panic(shape: (isize, isize), data: &[T]) {
        if (shape.0 * shape.1) as usize != data.len() {
            panic!(MISSMATCHED_SHAPE);
        }
    }

    fn missmatched_shapes_panic(&self, other: &Self) {
        if self.shape() != other.shape() {
            panic!(MISSMATCHED_SHAPES);
        }
    }

    fn index_out_of_bound_panic(&self, index: (isize, isize)) {
        if index.0 >= self.shape.0 {
            panic!(INDEX_OUT_OF_BOUNDS);
        }
    }
}

impl<T: Copy> Matrix<T> {
    pub fn from_val(shape: (isize, isize), val: T) -> Self {
        Self::illegal_shape_panic(shape);
        Self {
            shape,
            data: vec![val; (shape.0 * shape.1) as usize],
        }
    }

    pub fn shuffle(&mut self) {
        SliceRandom::shuffle(&mut self.data[..], &mut thread_rng());
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

impl Add<Self> for &Vector<f64> {
    type Output = Vector<f64>;

    fn add(self, other: Self) -> Self::Output {
        self.missmatched_shapes_panic(other);
        let mut result = Vector::from_val(self.shape, 0.0);
        unsafe {
            DAXPY(
                &(self.data.len() as i32),
                &1.0,
                self.data.as_ptr(),
                &1,
                result.data.as_mut_ptr(),
                &1,
            );
            DAXPY(
                &(self.data.len() as i32),
                &1.0,
                other.data.as_ptr(),
                &1,
                result.data.as_mut_ptr(),
                &1,
            );
        }
        result
    }
}

impl Add<Self> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn add(self, other: Self) -> Self::Output {
        self.missmatched_shapes_panic(other);
        let mut result = Matrix::from_val(self.shape, 0.0);
        unsafe {
            DAXPY(
                &(self.data.len() as i32),
                &1.0,
                self.data.as_ptr(),
                &1,
                result.data.as_mut_ptr(),
                &1,
            );
            DAXPY(
                &(self.data.len() as i32),
                &1.0,
                other.data.as_ptr(),
                &1,
                result.data.as_mut_ptr(),
                &1,
            );
        }
        result
    }
}

pub fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
