use std::time::{SystemTime, UNIX_EPOCH};

pub fn rand_gen() -> f32 {
    // Get current time in nanos as seed
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // Use a simple LCG formula: X_{n+1} = (a*X_n + c) mod m
    // Here we just compute one step from the seed
    let a: u128 = 6364136223846793005;
    let c: u128 = 1;
    let m: u128 = 1 << 31;

    let rand = (a.wrapping_mul(nanos) + c) % m;

    // Convert to f32 in [-1, 1]
    let uniform = (rand as f32) / (m as f32 / 2.0) - 1.0;

    // Scale to Xavier range [-0.085, 0.085]
    uniform * 0.085
}

pub fn dot_hidden(a: &[f32; 784], b: &[f32; 784]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_output(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn softmax(outputs: &[f32; 10]) -> [f32; 10] {
    let max = outputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut exps = [0.0f32; 10];
    for i in 0..10 {
        exps[i] = (outputs[i] - max).exp();
    }

    let sum: f32 = exps.iter().sum();

    let mut result = [0.0f32; 10];
    for i in 0..10 {
        result[i] = exps[i] / sum;
    }

    result
}

// pub fn relu_derivative(x: f32) -> f32 {
//     if x > 0.0 { 1.0 } else { 0.0 }
// }

// pub fn cross_entropy_loss(predicted: &[f32; 10], target: &[f32; 10]) -> f32 {
//     -predicted
//         .iter()
//         .zip(target.iter())
//         .map(|(p, t)| t * p.ln())
//         .sum::<f32>()
// }

pub fn one_hot(digit: usize) -> [f32; 10] {
    let mut arr = [0.0; 10];
    if digit < 10 {
        arr[digit] = 1.0;
    }
    arr
}

pub fn from_one_hot(arr: [f32; 10]) -> usize {
    let mut idx = 0;
    let mut max_val = arr[0];

    for (i, &val) in arr.iter().enumerate() {
        if val > max_val {
            max_val = val;
            idx = i;
        }
    }

    idx
}

pub fn argmax(arr: [f32; 10]) -> u8 {
    let mut max_index = 0;
    let mut max_value = arr[0];

    for i in 1..arr.len() {
        if arr[i] > max_value {
            max_value = arr[i];
            max_index = i;
        }
    }

    max_index as u8
}
