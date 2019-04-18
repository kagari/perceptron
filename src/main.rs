extern crate rand;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::{Normal, Distribution};
use ndarray::prelude::*;

extern crate perceptron;
use perceptron::Perceptron;

fn main() {
    // シードからランダムな値を生成
    let mut rng: StdRng = SeedableRng::from_seed([1; 32]);
    // 正規分布に従う
    let normal = Normal::new(0.0, 0.01); // new(平均、分散)

    // X: 入力データ(nxm行列)
    // y: 目的データ(nx1行列)
    // w: 重み(mx1行列)
    // b: バイアス(1x1要素)
    // th: 閾値(要素)
    let X = Array::from_shape_vec((4, 4), vec![1., 1., 1., 1., 2., 2., 2., 2., 1., 1., 1., 1., 2., 2., 2., 2.,]).unwrap();
    println!("X: {:?}", X);
    let y = Array1::from_shape_vec(4, vec![0., 1., 0., 1.,]).unwrap();
    let w = (0..X.cols()).map(|_| normal.sample(&mut rng)).collect::<Vec<f64>>();
    let w = Array1::from_shape_vec(4, w).unwrap();
    let b = normal.sample(&mut rng);
    let th = 0.01;

    // パーセプトロンのインスタンスを作成
    let mut perceptron = Perceptron::new(w, b, th);
    println!("perceptron is {:?}", &perceptron);

    // 学習を行う関数
    let perceptron = perceptron.fit(&X, &y, 1000);
    println!("perceptron is {:?}", &perceptron);
    
    // 予測を行う
    let test_X = Array::from_shape_vec((1, 4), vec![2., 2., 2., 2.,]).unwrap();
    let pred = perceptron.pred(test_X);
    println!("prediction is {:?}", pred.to_vec());
}
