use nalgebra::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::StandardNormal;
use std::iter::*;
use std::slice::Iter;

pub type Vector = DVector<f64>;
pub type Matrix = DMatrix<f64>;
pub type Data = (Vector, Vector);

pub trait SliceUp<T> {
    fn zip<'a, U>(&'a self, other: &'a [U]) -> Zip<Iter<'a, T>, Iter<'a, U>>;
    fn map<U>(&self, f: &Fn(&T) -> U) -> Vec<U>;
    fn sum_by<U: Sum>(&self, f: &Fn(&T) -> U) -> U;
    fn zip_map<U, V>(&self, other: &[U], f: &Fn(&T, &U) -> V) -> Vec<V>;
    fn elem(&self, index: i32) -> &T;
    fn set_elem(&mut self, index: i32, value: T);
    fn slice(&self, start: i32, end: i32) -> &[T];
    fn shuffle(&mut self);
}

impl<T> SliceUp<T> for [T] {
    fn zip<'a, U>(&'a self, other: &'a [U]) -> Zip<Iter<'a, T>, Iter<'a, U>> {
        self.iter().zip(other)
    }

    fn map<U>(&self, f: &Fn(&T) -> U) -> Vec<U> {
        self.iter().map(|t| f(t)).collect()
    }

    fn sum_by<U: Sum>(&self, f: &Fn(&T) -> U) -> U {
        self.iter().map(|t| f(t)).sum()
    }

    fn zip_map<U, V>(&self, other: &[U], f: &Fn(&T, &U) -> V) -> Vec<V> {
        self.iter()
            .zip(other)
            .map(|(t_0, t_1)| f(t_0, t_1))
            .collect()
    }

    fn elem(&self, mut index: i32) -> &T {
        if index < 0 {
            index += self.len() as i32;
        }
        &self[index as usize]
    }

    fn set_elem(&mut self, mut index: i32, value: T) {
        if index < 0 {
            index += self.len() as i32;
        }
        self[index as usize] = value;
    }

    fn slice(&self, mut start: i32, mut end: i32) -> &[T] {
        if start < 0 {
            start += self.len() as i32;
        }
        if end <= 0 {
            end += self.len() as i32;
        }
        &self[start as usize..end as usize]
    }

    fn shuffle(&mut self) {
        SliceRandom::shuffle(self, &mut thread_rng())
    }
}

pub trait VectorUp {
    fn sigmoid(&self) -> Vector;

    fn sigmoid_prime(&self) -> Vector;

    fn hadamard(&self, v: &Vector) -> Vector;
}

impl VectorUp for Vector {
    fn sigmoid(&self) -> Vector {
        self.map(&sigmoid)
    }

    fn sigmoid_prime(&self) -> Vector {
        self.map(&sigmoid_prime)
    }

    fn hadamard(&self, v: &Vector) -> Vector {
        self.zip_map(v, |x, y| x * y)
    }
}

fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
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