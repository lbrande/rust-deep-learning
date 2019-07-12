fn main() {
    
}
/*use lib::*;
use lib::utils::SliceUp;

use std::fs::File;
use std::io::*;

fn main() {
    let mut network = Network::new(&[784, 100, 10]);
    network.train(5, 10, 3.0, &mut load("train"), Some(&load("t10k")));
}

fn load(name: &str) -> Vec<Data> {
    let mut images = File::open(format!("{}-images.idx3-ubyte", name)).unwrap();
    let mut labels = File::open(format!("{}-labels.idx1-ubyte", name)).unwrap();
    let (nimages, image_size) = read_info(&mut images, &mut labels);
    let mut result = Vec::new();
    for _ in 0..nimages {
        let x = read_bytes(&mut images, image_size).map(&|&b| f64::from(b) / 255.0);
        let mut y = vec![0.0; 10];
        y[read_bytes(&mut labels, 1)[0] as usize] = 1.0;
        result.push(Network::data_from_vecs(x, y));
    }
    result
}

fn read_info(images: &mut File, labels: &mut File) -> (usize, usize) {
    read_bytes(images, 4);
    read_bytes(labels, 8);
    (read_usize(images), read_usize(images) * read_usize(images))
}

fn read_usize(file: &mut File) -> usize {
    let mut result = 0;
    for byte in read_bytes(file, 4) {
        result = (result << 8) + usize::from(byte);
    }
    result
}

fn read_bytes(file: &mut File, nbytes: usize) -> Vec<u8> {
    let mut buf = vec![0; nbytes];
    file.read_exact(&mut buf).unwrap();
    buf
}*/