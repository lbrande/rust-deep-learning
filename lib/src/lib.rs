use ndarray::linalg::*;
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;
use utils::*;
pub use utils::{Matrix, Vector};

mod utils;

pub type Data = (Vector, Vector);

pub struct Network {
    nlayers: usize,
    bs: Vec<Vector>,
    ws: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        Self {
            nlayers: layers.len(),
            bs: layers[1..]
                .iter()
                .map(|&x| Vector::random(x, StandardNormal))
                .collect(),
            ws: layers[1..]
                .iter()
                .zip(&layers[..layers.len() - 1])
                .map(|(&x, &y)| Matrix::random((x, y), StandardNormal))
                .collect(),
        }
    }

    pub fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        training_data: &mut [Data],
        test_data: Option<&[Data]>,
    ) {
        for i in 0..epochs {
            shuffle(training_data);
            for j in (0..training_data.len()).step_by(batch_size) {
                let batch = &training_data[j..j + batch_size];
                self.train_batch(learning_rate, batch);
            }
            if let Some(test_data) = test_data {
                println!(
                    "Epoch {}: {} / {}",
                    i,
                    self.evaluate(test_data),
                    test_data.len()
                );
            } else {
                println!("Epoch {} complete", i);
            }
        }
    }

    fn train_batch(&mut self, learning_rate: f64, batch: &[Data]) {
        let mut nabla_bs: Vec<Vector> = self.bs.iter().map(|b| Vector::zeros(b.dim())).collect();
        let mut nabla_ws: Vec<Matrix> = self.ws.iter().map(|w| Matrix::zeros(w.dim())).collect();
        for data in batch {
            self.backpropagate(&mut nabla_bs, &mut nabla_ws, data);
        }
        self.bs
            .iter_mut()
            .zip(&nabla_bs)
            .for_each(|(b, nb)| *b -= &((learning_rate / batch.len() as f64) * nb));
        self.ws
            .iter_mut()
            .zip(&nabla_ws)
            .for_each(|(w, nw)| *w -= &((learning_rate / batch.len() as f64) * nw));
    }

    fn backpropagate(&mut self, nabla_b: &mut [Vector], nabla_w: &mut [Matrix], (x, y): &Data) {
        let mut acts = vec![x.clone()];
        let mut zs = Vec::new();
        for (b, w) in self.bs.iter().zip(&self.ws) {
            let mut new_z = b.clone();
            general_mat_vec_mul(1.0, w, &acts.last().unwrap(), 1.0, &mut new_z);
            zs.push(new_z);
            acts.push(zs.last().unwrap().iter().map(|x| sigmoid(*x)).collect());
        }

        let mut delta = (acts.last().unwrap() - y) * zs.last().unwrap().map(|x| sigmoid_prime(*x));
        *nabla_b.last_mut().unwrap() += &delta;
        let mut new_nabla_w = Matrix::zeros((delta.dim(), acts[acts.len() - 2].dim()));
        general_mat_mul(
            1.0,
            &as_matrix(&delta),
            &as_matrix(&acts[acts.len() - 2]).t(),
            1.0,
            &mut new_nabla_w,
        );
        *nabla_w.last_mut().unwrap() += &new_nabla_w;
        for i in 2..self.nlayers {
            let z = &zs[zs.len() - i];
            let mut new_delta = Vector::zeros(self.ws[self.ws.len() - i + 1].dim().1);
            general_mat_vec_mul(
                1.0,
                &(&self.ws[self.ws.len() - i + 1]).t(),
                &delta,
                1.0,
                &mut new_delta,
            );
            new_delta *= &z.map(|x| sigmoid_prime(*x));
            delta = new_delta;
            nabla_b[nabla_b.len() - i] += &delta;
            new_nabla_w = Matrix::zeros((delta.dim(), acts[acts.len() - i - 1].dim()));
            general_mat_mul(
                1.0,
                &as_matrix(&delta),
                &as_matrix(&acts[acts.len() - i - 1]).t(),
                1.0,
                &mut new_nabla_w,
            );
            nabla_w[nabla_w.len() - i] += &new_nabla_w;
        }
    }

    fn evaluate(&self, test_data: &[Data]) -> usize {
        test_data
            .iter()
            .map(|(x, y)| (imax(&self.feedforward(x)) == imax(y)) as usize)
            .sum()
    }

    fn feedforward(&self, input: &Vector) -> Vector {
        let mut a = input.clone();
        for (b, w) in self.bs.iter().zip(&self.ws) {
            let mut new_a = b.clone();
            general_mat_vec_mul(1.0, w, &a, 1.0, &mut new_a);
            new_a.iter_mut().for_each(|x| *x = sigmoid(*x));
            a = new_a;
        }
        a
    }
}
