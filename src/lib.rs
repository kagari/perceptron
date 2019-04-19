extern crate ndarray;
use ndarray::prelude::*;
use std::fmt::Debug;

mod activation_function;

#[derive(Debug)]
pub struct Perceptron {
    w: Array1<f64>, // 重み (4x1行列)
    b: f64, // バイアス
    eta: f64, // 学習率
    error: Vec<i32>,
}

impl Perceptron {
    pub fn new(w: Array1<f64>, b: f64) -> Self {
        Perceptron {
            w,
            b,
            eta: 0.01,
            error: Vec::new(),
        }
    }

    // 学習を行う関数
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>, n_iter: i32) -> &Self {
        for _ in 0..n_iter {
            let mut error = 0;
            for (x, target) in X.outer_iter().zip(y.outer_iter()) {
                let update = self.eta * (target.to_owned() - self._pred(x.to_owned()));
                let update = update.into_scalar(); // into_scalar()が所有権を奪うため、結果をsに格納する
                self.w = update * x.to_owned() + &self.w;
                self.b += update;
                error += if update != 0. { 1 } else { 0 };
            }
            self.error.push(error);
        }
        self
    }

    // 実データに対して予測を行う関数
    pub fn pred(&self, X: &Array2<f64>) -> Array1<f64> {
        let mut pred = Vec::new();
        for row in X.outer_iter() {
            let y = self._pred(row.to_owned());
            pred.push(y);
        }
        Array1::from_shape_vec(X.rows(), pred).unwrap()
    }

    // 1xn行列に対して予測を行う関数
    fn _pred(&self, X: Array1<f64>) -> f64 {
        let x = X;
        let sop = x.dot(&self.w) + self.b; // 入力と重み(バイアスも含む)の積和(sum of product -> sop)を取る
        let pred = activation_function::step_function(sop);
        pred
    }
}
