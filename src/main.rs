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
    let X = Array::from_shape_vec((3, 4), vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.,]).unwrap();
    let y = Array1::from_shape_vec(3, vec![1., 2., 3.,]).unwrap();
    let w = Array1::from_shape_vec(4, vec![1., 2., 3., 4.,]).unwrap();
    let b = normal.sample(&mut rng);
    let th = 1.;

    // パーセプトロンのインスタンスを作成
    let mut perceptron = Perceptron::new(w, b, th);
    println!("perceptron is {:?}", &perceptron);

    // 学習を行う関数
    let perceptron = perceptron.fit(&X, &y);
    println!("perceptron is {:?}", &perceptron);
    
    // 予測を行う
    let pred = perceptron.pred(Array::from_shape_vec((1, 4), vec![4., 4., 4., 4.,]).unwrap());
    println!("prediction is {:?}", pred.to_vec());
}

#[derive(Debug)]
struct Perceptron {
    w: Array1<f64>, // 重み (4x1行列)
    b: f64, // バイアス
    th: f64, // 閾値
    eta: f64, // 学習率
    error: Vec<i32>,
}

impl Perceptron {
    fn new(w: Array1<f64>, b: f64, th: f64) -> Self {
        Perceptron {
            w,
            b,
            th,
            eta: 0.01,
            error: Vec::new(),
        }
    }

    // 学習を行う関数
    fn fit(&mut self, X: &Array<f64, Ix2>, y: &Array1<f64>) -> &Self {
        let mut error = 0;
        for (x, target) in X.outer_iter().zip(y.outer_iter()) {
            let update = self.eta * (target.to_owned() - self._pred(x.to_owned()));
            // println!("update value is {:?}", update);
            let s = update.into_scalar(); // into_scalar()が所有権を奪うため、結果をsに格納する
            self.w = ((s * x.to_owned()) - &self.w) * -1.;
            self.b += s;
            error += if s != 0. { 1 } else { 0 };
        }
        self.error.push(error);
        self
    }

    // 実データに対して予測を行う関数
    fn pred(&self, X: Array<f64, Ix2>) -> Array1<f64> {
        let mut pred = Vec::new();
        for row in X.outer_iter() {
            let y = self._pred(row.to_owned());
            pred.push(y);
        }
        Array1::from_shape_vec(X.rows(), pred).unwrap()
    }

    // 1xn行列に対して予測を行う関数
    fn _pred(&self, X: Array<f64, Ix1>) -> f64 {
        let x = X;
        let input_sum = x.dot(&self.w) + self.b; // 入力と重み(バイアスも含む)の積和を取る
        let pred = if input_sum > self.th { 1. } else { 0. };
        pred
    }
}
