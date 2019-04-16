extern crate nalgebra as na;
use f64;
use std::str;
use std::str::FromStr;
use std::vec::Vec;
use na::{DMatrix, DVector, Matrix4};
use rand::{Rng, SeedableRng, StdRng};


fn main() {
    println!("Hello, world!");

    // irisデータセットを使用
    let path = "./data/iris.csv";
    let mut reader = csv::Reader::from_path(path).unwrap();

    let mut x: Vec<f64> = vec![];
    let mut nrows: usize = 0;

    for record in reader.byte_records().map(|r| r.unwrap()) {
        // f64 に変換できる列のみ読み込み
        for item in record.iter().map(|i| str::from_utf8(i).unwrap()) {
            match f64::from_str(item) {
                Ok(v) => x.push(v),
                Err(e) => {}
            };
        }
        nrows += 1;
    }

    let ncols = x.len() / nrows;

    let dx = DMatrix::from_row_slice(nrows, ncols, &x);

    let perceptron = Perceptron::new();
//    perceptron.fit();
}

struct Perceptron {
    eta: f64,
    n_iter: u32,
    random_state: i32,
    w_: DMatrix<f64>,
    error_: Vec<Vec<i32>>,
}

impl Perceptron {
    fn new() -> Perceptron {
        Perceptron {
            eta: 0.01,
            n_iter: 50,
            random_state: 1,
        }
    }

    fn fit(&self, X: DMatrix<f64>, y: Vec<f64>) {
        let mut rng: StdRng = SeedableRng::from_seed(self.random_state);
        self.w_ = (0..X)
    }
}
