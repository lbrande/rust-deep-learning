use std::fs::File;
use std::io::Read;
use lib::*;

fn main() {
    let x = Network::data_from_vec(Vec::new(), Vec::new());
    println!("Hello, world!");
}

fn load(name: &str) -> Vec<Data> {
    let mut image_file = File::open(format!("{}-images.idx3-ubyte", name)).unwrap();
    let mut label_file = File::open(format!("{}-labels.idx1-ubyte", name)).unwrap();
    let mut image_buffer = Vec::new();
    let mut label_buffer = Vec::new();
    image_file.read_to_end(&mut image_buffer);
    label_file.read_to_end(&mut label_buffer);
    let mut result = Vec::new();
    result
}