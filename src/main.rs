extern crate rand;
extern crate csv;
extern crate ndarray;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::{Normal, Distribution};
use ndarray::prelude::*;
use ndarray::Slice;
use std::fs::File;

use std::f64;
use std::str::FromStr;
extern crate perceptron;
use perceptron::Perceptron;

fn main() {
    // シードからランダムな値を生成
    let mut rng: StdRng = SeedableRng::from_seed([1; 32]);
    // 正規分布に従う
    let normal = Normal::new(0.0, 0.01); // new(平均、分散)

    // データの読み込み
    let file = File::open("./data/iris.csv").expect("opening file failed");
    let mut rdr = csv::ReaderBuilder::new().delimiter(b',').has_headers(true).from_reader(file);
    let mut X: Vec<f64> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut nrows = 0;
    for result in rdr.records() {
        let recode = result.unwrap();
        for item in recode.iter() {
            match f64::from_str(item) {
                Ok(v) => X.push(v),
                Err(e) => y.push(if item == "setosa" { 1. } else { -1. }),
            };
        };
        nrows += 1;
    };

    // X: 入力データ(nxm行列)
    // y: 目的データ(nx1行列)
    // w: 重み(mx1行列)
    // b: バイアス(1x1要素)
    let X = Array2::from_shape_vec((X.len()/4, 4), X).unwrap().slice_axis(Axis(0), Slice::from(..100)).to_owned();
    let y = Array1::from_shape_vec(y.len(), y).unwrap();
    let w = (0..X.cols()).map(|_| normal.sample(&mut rng)).collect::<Vec<f64>>();
    let w = Array1::from_shape_vec(4, w).unwrap();
    let b = normal.sample(&mut rng);
    println!("w: {:?}", w);
    
    // パーセプトロンのインスタンスを作成
    let mut perceptron = Perceptron::new(w, b);
    println!("perceptron is {:?}", &perceptron);

    // 学習を行う関数
    let perceptron = perceptron.fit(&X, &y, 1000);
    println!("perceptron is {:?}", &perceptron);
    
    // 予測を行う
    // 1.0と予測するはず
    let test_X = X.slice_axis(Axis(0), Slice::from(0..5)).to_owned(); // s! macroが使えなかったので
    let test_y = y.slice_axis(Axis(0), Slice::from(0..5)).to_owned(); // s! macroが使えなかったので
    let pred = perceptron.pred(&test_X);
    println!("test_y     : {:?}", test_y);
    println!("prediction : {:?}", pred.to_vec());
    println!("prediction : {}", pred == test_y);

    let test_X = X.slice_axis(Axis(0), Slice::from(-5..)).to_owned(); // s! macroが使えなかったので
    let test_y = y.slice_axis(Axis(0), Slice::from(-5..)).to_owned(); // s! macroが使えなかったので
    let pred = perceptron.pred(&test_X);
    println!("test_y     : {:?}", test_y);
    println!("prediction : {:?}", pred.to_vec());
    println!("prediction : {}", pred == test_y);

}
