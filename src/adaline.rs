use ndarray::prelude::*;
use std::fmt::Debug;

#[derive(Debug)]
pub struct AdalineDG {
    w: Array1<f64>, // 重み (4x1行列)
    b: f64, // バイアス
    eta: f64, // 学習率
    cost: Vec<f64>,
}

impl AdalineDG {
    pub fn new(w: Array1<f64>, b: f64) -> Self {
        AdalineDG {
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
    
    fn net_input(&self, X: &Array2<f64>) -> Array1<f64> {
        let sop = X.dot(&self.w) + self.b;
        sop
    }

    fn activation(&self, X: Array1<f64>) -> Array1<f64> {
        // 恒等関数
        X
    }
}

#[derive(Debug)]
pub struct AdalineSDG {
    w: Array1<f64>, // 重み (4x1行列)
    b: f64, // バイアス
    eta: f64, // 学習率p
    cost: Vec<f64>,
    is_shuffle: bool, // エポックごとにシャッフルするかのフラグ
}

impl AdalineSDG {
    pub fn new(w: Array1<f64>, b: f64, shuffle: bool) -> Self {
        AdalineSDG {
            w,
            b,
            eta: 0.01,
            cost: Vec::new(),
            is_shuffle: shuffle,
        }
    }

    // 学習を行う関数
    // 確率的勾配降下法
    pub fn fit(mut self, X: &Array2<f64>, y: &Array1<f64>, n_iter: i32) -> Self {
        for _ in 0..n_iter {
            let (X, y) = if self.is_shuffle {
                (X, y)
            } else {
                (X, y)
            };
            let mut cost = Vec::new();
            for (xi, target) in X.outer_iter().zip(y.outer_iter()) {
                cost.push(self.update_weights(xi.to_owned(), target.to_owned()));
            }
            let avg_cost: f64 = cost.iter().sum::<f64>() / cost.len() as f64;
            self.cost.push(avg_cost);
        }
        self
    }
    
    pub fn pred(&self, X: &Array2<f64>) -> Array1<f64> {
        self.activation(self.net_input(X)).map(|x| if x > &0. { 1. } else { -1. })
    }
    
    fn net_input(&self, X: &Array2<f64>) -> Array1<f64> {
        let sop = X.dot(&self.w) + self.b;
        sop
    }
    
    fn activation(&self, X: Array1<f64>) -> Array1<f64> {
        // 恒等関数
        X
    }

    fn update_weights(&mut self, xi: Array1<f64>, target: Array0<f64>) -> f64 {
        let xi = xi.into_shape((1,2)).unwrap();
        let output = self.activation(self.net_input(&xi)); // 重み付き積和→活性化関数の出力
        let error = target.into_scalar() - output.into_shape(()).unwrap().into_scalar(); // 誤差
        self.w = (self.eta * xi * error + &self.w).into_shape((2)).unwrap();
        self.b += self.eta * error;
        let cost = error.powi(2) / 2.;//error.map(|x| x.powi(2)).sum() / 2.;
        cost
    }
}
