use nalgebra as na;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::StandardNormal;
use std::iter::Sum;

type Vector = na::DVector<f64>;
type Matrix = na::DMatrix<f64>;
type Data = (Vector, Vector);

fn map<T: Copy, U>(f: &Fn(T) -> U, slice: &[T]) -> Vec<U> {
    slice.iter().map(|&t| f(t)).collect()
}

fn map_and_sum<T, U: Sum>(f: &Fn(&T) -> U, slice: &[T]) -> U {
    slice.iter().map(|t| f(t)).sum()
}

fn zip_with<T: Copy, U>(f: &Fn(T, T) -> U, slice_0: &[T], slice_1: &[T]) -> Vec<U> {
    slice_0
        .iter()
        .zip(slice_1)
        .map(|(&t_0, &t_1)| f(t_0, t_1))
        .collect()
}

fn randn() -> f64 {
    thread_rng().sample(StandardNormal)
}

fn shuffle<T>(slice: &mut [T]) {
    slice.shuffle(&mut thread_rng())
}

fn randn_vector(rows: usize) -> Vector {
    Vector::from_vec((0..rows).map(|_| randn()).collect())
}

fn randn_matrix(rows: usize, columns: usize) -> Matrix {
    Matrix::from_vec(
        rows,
        columns,
        (0..rows * columns).map(|_| randn()).collect(),
    )
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
            biases: map(&randn_vector, init),
            weights: zip_with(&randn_matrix, tail, init),
        }
    }

    pub fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        training_data: &mut Vec<Data>,
        test_data: Option<&[Data]>,
    ) {
        for i in 0..epochs {
            shuffle(training_data);
            for j in (0..training_data.len()).step_by(batch_size) {
                let batch = &training_data[j..j + batch_size];
                self.train_batch(batch, learning_rate);
            }
            if let Some(test_data) = test_data {
                println!("Epoch {}: {} / {}", i, self.evaluate(test_data), test_data.len());
            } else {
                println!("Epoch {} complete", i);
            }
        }
    }

    fn train_batch(&mut self, batch: &[Data], learning_rate: f64) {}

    fn evaluate(&self, test_data: &[Data]) -> usize {
        map_and_sum(&|(x, y)| (x == y) as usize, test_data)
    }
}

#[cfg(test)]
mod tests {}
