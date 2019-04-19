pub fn step_function(x: f64) -> f64 {
    if x > 0. { 1. } else { -1. }
}

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn ReLU(x: f64) -> f64 {
    if x > 0. { x } else { 0. }
}
