use nalgebra::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::StandardNormal;
use std::iter::*;
use std::slice::Iter;

pub type Vector = DVector<f64>;
pub type Matrix = DMatrix<f64>;
pub type Data = (Vector, Vector);

pub fn zip<'a, T, U>(slice_0: &'a [T], slice_1: &'a [U]) -> Zip<Iter<'a, T>, Iter<'a, U>> {
    slice_0.iter().zip(slice_1)
}

pub fn map<T, U>(f: &Fn(&T) -> U, slice: &[T]) -> Vec<U> {
    slice.iter().map(|t| f(t)).collect()
}

pub fn sum_by<T, U: Sum>(f: &Fn(&T) -> U, slice: &[T]) -> U {
    slice.iter().map(|t| f(t)).sum()
}

pub fn zip_with<T, U, V>(f: &Fn(&T, &U) -> V, slice_0: &[T], slice_1: &[U]) -> Vec<V> {
    slice_0
        .iter()
        .zip(slice_1)
        .map(|(t_0, t_1)| f(t_0, t_1))
        .collect()
}

pub fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

pub fn shuffle<T>(slice: &mut [T]) {
    slice.shuffle(&mut thread_rng())
}

pub fn randn1(len: usize) -> Vector {
    DVector::from_fn(len, &|_, _| randn())
}

pub fn randn2(nrows: usize, ncols: usize) -> Matrix {
    DMatrix::from_fn(nrows, ncols, &|_, _| randn())
}

pub fn zeros1(len: usize) -> Vector {
    DVector::from_element(len, 0.0)
}

pub fn zeros2(nrows: usize, ncols: usize) -> Matrix {
    DMatrix::from_element(nrows, ncols, 0.0)
}