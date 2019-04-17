// extern crate nalgebra as na;
extern crate ndarray;
extern crate rand;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::{Normal, Distribution};
// use na::core::DMatrix;
use ndarray::prelude::*;
use std::fmt::Debug;

fn main() {
    println!("Hello, world");
    // シードからランダムな値を生成
    let mut rng: StdRng = SeedableRng::from_seed([1; 32]);
    // 正規分布に従う
    let normal = Normal::new(1.0, 0.0);


    // X: 入力データ(nxm行列)
    // y: 目的データ(nx1行列)
    // x: 入力信号(1xm行列)
    // w: 重み(mx1行列)
    // b: バイアス(1x1要素)
    // th: 閾値(要素)
    let X = Array::from_shape_vec((1, 4), vec![0., 0., 0., 0.]).unwrap();
    // let x = X;
    let y = 1;
    let w = Array1::from_shape_vec(4, vec![1., 2., 3., 4.,]).unwrap();
    let b = Option::Some(normal.sample(&mut rng));
    let th = Option::Some(1.);

    let perceptron = Perceptron::new(&w, &b, &th);
    println!("perceptron is {:?}", &perceptron);

    let perceptron = perceptron.fit(X, Array1::from_shape_vec(1, vec![1.]).unwrap());

    // println!("x is {:?}", x);
    // println!("w is {:?}", w);
    // let input_sum = (x.dot(&w) + b).into_raw_vec()[0];
    // println!("input sum: {:?}", input_sum);
    
    // let pred = if input_sum > th { 1 } else { 0 };
    // println!("pred is {:?}", pred);
}

#[derive(Debug)]
struct Perceptron<'a> {
    w: &'a Array1<f64>, // 重み (4x1行列)
    b: &'a Option<f64>, // バイアス
    th: &'a Option<f64>, // 閾値
}

impl<'a> Perceptron<'a> {
    fn new(w: &'a Array1<f64>, b: &'a Option<f64>, th: &'a Option<f64>) -> Self {
        Perceptron {
            w,
            b,
            th,
        }
    }

    fn fit(&self, X: Array<f64, Ix2>, y: Array1<f64>) -> &Self {
        let x = X;
        let input_sum = (x.dot(self.w) + self.b.unwrap()).into_raw_vec()[0];
        println!("input sum: {:?}", input_sum);
        let pred = if input_sum > self.th.unwrap() { 1 } else { 0 };
        println!("pred is {:?}", pred);
        self
    }
}
