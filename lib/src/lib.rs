use nalgebra as na;
use rand::prelude::*;
use rand_distr::StandardNormal;

type Vector = na::DVector<f64>;
type Matrix = na::DMatrix<f64>;

trait FunctionalList<T> {
    fn map<U>(&self, function: &Fn(&T) -> U) -> Vec<U>;
    fn zip_map<U>(&self, other: &Self, function: &Fn(&T, &T) -> U) -> Vec<U>;
}

impl<T> FunctionalList<T> for [T] {
    fn map<U>(&self, function: &Fn(&T) -> U) -> Vec<U> {
        let mut result = Vec::with_capacity(self.len());
        for t in self {
            result.push(function(t));
        }
        result
    }

    fn zip_map<U>(&self, other: &Self, function: &Fn(&T, &T) -> U) -> Vec<U> {
        let mut result = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            result.push(function(&self[i], &other[i]));
        }
        result
    }
}

fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

fn vector(len: usize, generator: &Fn() -> f64) -> Vector {
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        result.push(generator());
    }
    na::DVector::from_vec(result)
}

fn matrix(rows: usize, columns: usize, generator: &Fn() -> f64) -> Matrix {
    let mut result = Vec::with_capacity(rows * columns);
    for _ in 0..rows * columns {
        result.push(generator());
    }
    na::DMatrix::from_vec(rows, columns, result)
}

pub struct Network {
    biases: Vec<Vector>,
    weights: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let init = &layers[1..];
        let tail = &layers[..layers.len() - 1];
        Self {
            biases: init.map(&|&x| vector(x, &randn)),
            weights: tail.zip_map(init, &|&x, &y| matrix(x, y, &randn)),
        }
    }
}

#[cfg(test)]
mod tests {}
