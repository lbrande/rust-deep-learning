pub mod utils;
use utils::*;

pub type Data = (Matrix, Matrix);

pub struct Network {
    nlayers: isize,
    biases: Vector<Matrix>,
    weights: Vector<Matrix>,
}

impl Network {
    pub fn new(layers: &Vector<isize>) -> Self {
        Self {
            nlayers: layers.len(),
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
        let mut nabla_b = self.biases.map(&|b| Matrix::from_val(b.shape(), 0.0));
        let mut nabla_w = self.weights.map(&|w| Matrix::from_val(w.shape(), 0.0));
        for data in batch {
            self.backpropagate(&mut nabla_b, &mut nabla_w, data);
        }
        self.biases.zip_apply(&mut nabla_b, &|(b, nb): (&mut Matrix, &mut Matrix)| {
            *nb *= learning_rate / batch.len() as f64;
            *b -= nb;
        });
        self.weights.zip_apply(&mut nabla_w, &|(w, nw): (&mut Matrix, &mut Matrix)| {
            *nw *= learning_rate / batch.len() as f64;
            *w -= nw;
        });
    }

    fn backpropagate(&self, nabla_b: &mut Vector<Matrix>, nabla_w: &mut Vector<Matrix>, (x, y): &Data) {
        let mut activations = Vector::from_vec(vec![self.biases[0].clone()]);
        Matrix::dgemv('N', &self.weights[0], x, &mut activations[0]);
        activations[0].apply(&sigmoid);
        let mut zs = Vector::new();
        for (b, w) in self.biases[1..].zip(&self.weights[1..]) {
            let mut b = b.clone();
            Matrix::dgemv('N', &w, &activations[-1], &mut b);
            zs.push(b.clone());
            b.apply(&sigmoid);
            activations.push(b);
        }
        let mut delta = Matrix::from_shape(activations[-1].shape());
        delta.set_ref_to(&activations[-1]);
        delta -= y;
        zs[-1].apply(&sigmoid_prime);
        delta.hadamard(&zs[-1]);
        nabla_b[-1] += &delta;
        let mut delta_nabla_w = Matrix::from_val((delta.shape().1, activations[-2].shape().1), 0.0);
        Matrix::dgemm('N', 'T', &delta, &activations[-2], &mut delta_nabla_w);
        nabla_w[-1] += &delta_nabla_w;
        for i in 2..self.nlayers {
            let i = i as isize;
            zs[-i].apply(&sigmoid_prime);
            let w = self.weights[-i + 1].clone();
            let mut delta_new = Matrix::from_val((w.shape().0, delta.shape().0), 0.0);
            Matrix::dgemm('T', 'N', &w, &delta, &mut delta_new);
            delta = delta_new;
            delta.hadamard(&zs[-i]);
            nabla_b[-i] += &delta;
            let mut delta_nabla_w = Matrix::from_val((delta.shape().1, activations[-i - 1].shape().1), 0.0);
            Matrix::dgemm('N', 'T', &delta, &activations[-i - 1], &mut delta_nabla_w);
            nabla_w[-i] += &delta_nabla_w;
        }
    }

    fn evaluate(&self, test_data: &[Data]) -> usize {
        test_data.map_sum(&|(x, y): &Data| (self.feedforward(x).imax() == y.imax()) as usize)
    }

    fn feedforward(&self, x: &Matrix) -> Matrix {
        let mut a = self.biases[0].clone();
        Matrix::dgemv('N', &self.weights[0], x, &mut a);
        a.apply(&sigmoid);
        for (b, w) in self.biases[1..].zip(&self.weights[1..]) {
            let mut b = b.clone();
            Matrix::dgemv('N', &w, &a, &mut b);
            b.apply(&sigmoid);
            a = b;
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
