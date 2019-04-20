use ndarray::prelude::*;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Adaline {
    w: Array1<f64>, // 重み (4x1行列)
    b: f64, // バイアス
    eta: f64, // 学習率
    cost: Vec<f64>,
}

impl Adaline {
    pub fn new(w: Array1<f64>, b: f64) -> Self {
        Adaline {
            w,
            b,
            eta: 0.01,
            cost: Vec::new(),
        }
    }

    // 学習を行う関数
    pub fn fit(mut self, X: &Array2<f64>, y: &Array1<f64>, n_iter: i32) -> Self {
        for _ in 0..n_iter {
            let net_input = self.net_input(&X);
            let output = self.activation(net_input); // outputはベクトル
            let errors = y.to_owned() - output; // 誤差
            self.w = self.eta * X.t().dot(&errors) + &self.w;
            self.b += self.eta * errors.sum();
            let cost = errors.map(|x| x.powi(2)).sum() / 2.;
            self.cost.push(cost);
        }
        self
    }

    pub fn pred(&self, X: &Array2<f64>) -> Array1<f64> {
        self.activation(self.net_input(X)).map(|x| if x > &0. { 1. } else { -1. })
    }

    // 重み付き入力を計算する
    fn net_input(&self, X: &Array2<f64>) -> Array1<f64> {
        let sop = X.dot(&self.w) + self.b;
        sop
    }

    fn activation(&self, X: Array1<f64>) -> Array1<f64> {
        // 恒等関数
        X
    }
}
