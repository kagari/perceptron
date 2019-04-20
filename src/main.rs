extern crate rand;
extern crate csv;
#[macro_use]
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
use perceptron::adaline::Adaline;

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
    // let X = Array2::from_shape_vec((X.len()/4, 4), X).unwrap().slice_axis(Axis(1), Slice::from(..100)).to_owned();
    let X = Array2::from_shape_vec((X.len()/4, 4), X).unwrap().select(Axis(1), &[0, 2]).slice(s![..100, ..]).to_owned();
    let y = Array1::from_shape_vec(y.len(), y).unwrap().slice(s![..100]).to_owned();
    let w = (0..X.cols()).map(|_| normal.sample(&mut rng)).collect::<Vec<f64>>();
    let w = Array1::from_shape_vec(2, w).unwrap();
    let b = normal.sample(&mut rng);
    
    // // パーセプトロンのインスタンスを作成
    // let mut perceptron = Perceptron::new(w, b);
    // println!("perceptron : {:?}", &perceptron);

    // // 学習を行う関数
    // let perceptron = perceptron.fit(&X, &y, 10);
    // println!("perceptron : {:?}", &perceptron);
    
    // // 予測を行う
    // // 1.0と予測するはず
    // let test_X = X.slice_axis(Axis(0), Slice::from(0..5)).to_owned(); // s! macroが使えなかったので
    // let test_y = y.slice_axis(Axis(0), Slice::from(0..5)).to_owned(); // s! macroが使えなかったので
    // let pred = perceptron.pred(&test_X);
    // println!("test_y     : {:?}", test_y);
    // println!("prediction : {:?}", pred.to_vec());
    // println!("prediction : {}", pred == test_y);

    // let test_X = X.slice_axis(Axis(0), Slice::from(-5..)).to_owned(); // s! macroが使えなかったので
    // let test_y = y.slice_axis(Axis(0), Slice::from(-5..)).to_owned(); // s! macroが使えなかったので
    // let pred = perceptron.pred(&test_X);
    // println!("test_y     : {:?}", test_y);
    // println!("prediction : {:?}", pred.to_vec());
    // println!("prediction : {}", pred == test_y);

    // let mut adaline = Adaline::new(w, b);
    // println!("adaline: {:?}", adaline);

    // let test_X = X.slice_axis(Axis(0), Slice::from(0..5)).to_owned(); // s! macroが使えなかったので
    // let test_y = y.slice_axis(Axis(0), Slice::from(0..5)).to_owned(); // s! macroが使えなかったので
    // let adaline = adaline.fit(&X, &y, 100);
    // println!("adaline: {:?}", adaline);
    // let pred = adaline.pred(&test_X);
    // println!("test_y     : {:?}", test_y);
    // println!("pred: {:?}", pred);
    // println!("test: {}", pred == test_y);

    // let test_X = X.slice_axis(Axis(0), Slice::from(-5..)).to_owned(); // s! macroが使えなかったので
    // let test_y = y.slice_axis(Axis(0), Slice::from(-5..)).to_owned(); // s! macroが使えなかったので
    // let adaline = adaline.fit(&X, &y, 100);
    // let pred = adaline.pred(&test_X);
    // println!("test_y     : {:?}", test_y);
    // println!("pred : {:?}", pred.to_vec());
    // println!("test : {}", pred == test_y);

    // 標準化 standardizationを行い、学習がうまくいくことを確認する
    let mut adaline = Adaline::new(w, b);
    println!("adaline: {:?}", adaline);
    
    // 訓練データの標準化
    let X_std = (&X - &X.mean_axis(Axis(0))) / &X.std_axis(Axis(0), 0.);
    let adaline = adaline.fit(&X_std, &y, 100); // 学習
    println!("adaline: {:?}", adaline);

    // 訓練データを用いて学習が上手くいっているかの確認(1と予測)
    let test_X = X_std.slice(s![..5, ..]).to_owned();
    let test_y = y.slice(s![..5]).to_owned();
    let pred = adaline.pred(&test_X);
    println!("test_y     : {:?}", test_y);
    println!("pred: {:?}", pred);
    println!("test: {}", pred == test_y);

    // 訓練データを用いて学習が上手くいっているかの確認(-1と予測)
    let test_X = X_std.slice(s![-5.., ..]).to_owned();
    let test_y = y.slice(s![-5..]).to_owned();
    let pred = adaline.pred(&test_X);
    println!("test_y     : {:?}", test_y);
    println!("pred : {:?}", pred.to_vec());
    println!("test : {}", pred == test_y);
}
