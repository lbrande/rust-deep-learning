pub mod utils;
use utils::*;

pub type Data = (Matrix, Matrix);

pub struct Network {
    nlayers: usize,
    biases: Vector<Matrix>,
    weights: Vector<Matrix>,
}

impl Network {
    pub fn new(layers: &Vector<isize>) -> Self {
        Self {
            nlayers: layers.len() as usize,
            biases: layers[1..].map(&|&x| Matrix::from_fn((x, 1), &|_| randn())),
            weights: (&layers[1..]).zip_map(&layers[..-1], &|(&x, &y)| {
                Matrix::from_fn((x, y), &|_| randn())
            }),
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
            training_data.shuffle();
            for j in (0..training_data.len()).step_by(batch_size) {
                let batch = &training_data[j..j + batch_size];
                self.train_batch(batch, learning_rate);
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

    fn train_batch(&mut self, batch: &[Data], learning_rate: f64) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        for data in batch {
            let (delta_nabla_b, delta_nabla_w) = self.backpropagate(data);
            nabla_b.zip_apply(&delta_nabla_b, &|(nb, dnb)| *nb = &*nb + &*dnb);
            nabla_w.zip_apply(&delta_nabla_w, &|(nw, dnw)| *nw = &*nw + &*dnw);
        }
        self.biases.zip_apply(&nabla_b, &|(b, nb)| {
            *b = &*b - &(&(learning_rate * &*nb) / batch.len() as f64)
        });
        self.weights.zip_apply(&nabla_w, &|(w, nw)| {
            *w = &*w - &(&(learning_rate * &*nw) / batch.len() as f64)
        });
    }

    fn backpropagate(&self, (x, y): &Data) -> (Vector<Matrix>, Vector<Matrix>) {
        let (mut nabla_b, mut nabla_w) = self.nabla_zeros();
        let mut activations = Vector::from_vec(vec![x.clone()]);
        let mut zs = Vector::new();
        for (b, w) in self.biases.zip(&self.weights) {
            zs.push(&(w * &activations[-1]) + b);
            activations.push(zs[-1].map(&sigmoid));
        }
        let mut delta = (&activations[-1] - y).hadamard(&(zs[-1].map(&sigmoid_prime)));
        nabla_b[-1] = delta.clone();
        nabla_w[-1] = &delta.clone() * &*activations[-2].transpose();
        for i in 2..self.nlayers {
            let i = i as isize;
            let sp = zs[-i].map(&sigmoid_prime);
            let mut ab = self.weights[-i + 1].clone();
            delta = (&*ab.transpose() * &delta).hadamard(&sp);
            nabla_b[-i] = delta.clone();
            nabla_w[-i] = &delta.clone() * &*activations[-i - 1].transpose();
        }
        (nabla_b, nabla_w)
    }

    fn nabla_zeros(&self) -> (Vector<Matrix>, Vector<Matrix>) {
        (
            self.biases.map(&|b| Matrix::from_val(b.shape(), 0.0)),
            self.weights.map(&|w| Matrix::from_val(w.shape(), 0.0)),
        )
    }

    fn evaluate(&self, test_data: &[Data]) -> usize {
        test_data.map_sum(&|(x, y): &Data| (self.feedforward(x).imax() == y.imax()) as usize)
    }

    fn feedforward(&self, x: &Matrix) -> Matrix {
        let mut a = x.clone();
        for (b, w) in self.biases.zip(&self.weights) {
            a = (&(w * &a) + b).map(&sigmoid);
        }
        a
    }

    pub fn vecs_into_data(x: Vec<f64>, y: Vec<f64>) -> Data {
        (
            Matrix::from_vec((x.len() as isize, 1), x),
            Matrix::from_vec((y.len() as isize, 1), y),
        )
    }
}
