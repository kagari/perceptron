extern crate rand;
extern crate csv;
extern crate nalgebra as na;
use f64;
use std::str;
use std::str::FromStr;
use std::vec::Vec;
use na::{DMatrix, DVector, Matrix4};
use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{Normal, Distribution};

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

    // パーセプトロンコンストラクタを生成
    let mut perceptron = Perceptron::new();
    // 学習
    println!("a shape of dx: {:?}", dx.shape());
    perceptron.fit(dx, vec![1., 2., 3., 4.]);
    println!("perceptron w value is {:?}", perceptron.w_);
}

struct Perceptron {
    eta: f64,
    n_iter: u32,
    random_state: [u8; 32],
    w_: Option<Vec<f64>>,
    error_: Option<Vec<Vec<i32>>>,
}

impl Perceptron {
    fn new() -> Perceptron {
        Perceptron {
            eta: 0.01,
            n_iter: 50,
            random_state: [1; 32],
            w_: None,
            error_: None,
        }
    }

    fn fit(&mut self, X: DMatrix<f64>, y: Vec<f64>) {
        let mut rng: StdRng = SeedableRng::from_seed(self.random_state);
        let normal = Normal::new(2.0, 3.0);
        self.w_ = Option::Some((0..X.shape().1).map(|_| normal.sample(&mut rng)).collect());
        self.error_ = Option::Some(Vec::new());

        for i in 0..self.n_iter { // n_iter回トレーニングを行う
            // println!("epoch: {}", i);
            let error = 0;
            let mut i = 0;
            for (xi, target) in X.row_iter().zip(y.iter()) { // yの長さに合わせられる
                //println!("xi is {:?}", xi);
                //println!("target is {}", target);
                i += 1;
            }
            println!("i: {}", i);
        }
    }
}
