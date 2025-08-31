use crate::utils::{self, softmax};
const HIDDEN: usize = 64;

#[derive(Debug)]
pub struct Network {
    input: usize,  // 784
    hidden: usize, // 64
    output: usize, // 10

    hidden_weights: [[f32; 784]; HIDDEN],
    output_weights: [[f32; HIDDEN]; 10],

    hidden_biase: [f32; HIDDEN],
    output_biase: [f32; 10],
}

impl Network {
    pub fn new(input: usize, hidden: usize, output: usize) -> Self {
        let mut hidden_weights = [[0.0; 784]; HIDDEN];
        let mut output_weights = [[0.0; HIDDEN]; 10];
        let mut hidden_biase = [0.0; HIDDEN];
        let mut output_biase = [0.0; 10];

        for h in 0..hidden {
            for i in 0..input {
                hidden_weights[h][i] = utils::rand_gen();
            }
        }

        for o in 0..output {
            for h in 0..hidden {
                output_weights[o][h] = utils::rand_gen();
            }
        }

        for h in 0..hidden {
            hidden_biase[h] = utils::rand_gen();
        }

        for o in 0..output {
            output_biase[o] = utils::rand_gen();
        }

        Self {
            input,
            hidden,
            output,
            hidden_weights,
            output_weights,
            hidden_biase,
            output_biase,
        }
    }

    pub fn forward(&self, input_pixels: [f32; 784]) -> [f32; 10] {
        // hiden layers
        let mut hidden_out = [0.0; HIDDEN];
        for i in 0..HIDDEN {
            hidden_out[i] = utils::relu(
                utils::dot_hidden(&self.hidden_weights[i], &input_pixels) + self.hidden_biase[i],
            );
        }

        // output layer
        let mut output_out = [0.0; 10];
        for i in 0..10 {
            output_out[i] =
                utils::dot_output(&hidden_out, &self.output_weights[i]) + self.output_biase[i];
        }

        output_out = softmax(&output_out);
        output_out
    }
}

impl Network {
    pub fn backprop(&mut self, input_pixels: &[f32; 784], target: [f32; 10], lr: f32) {
        // -------- 1. Forward pass (store activations) --------
        let mut hidden_z = [0.0; HIDDEN]; // pre-activation
        let mut hidden_a = [0.0; HIDDEN]; // post-activation
        for h in 0..HIDDEN {
            hidden_z[h] =
                utils::dot_hidden(&self.hidden_weights[h], &input_pixels) + self.hidden_biase[h];
            hidden_a[h] = utils::relu(hidden_z[h]);
        }

        let mut output_z = [0.0; 10];
        for o in 0..self.output {
            output_z[o] =
                utils::dot_output(&hidden_a, &self.output_weights[o]) + self.output_biase[o];
        }

        let output_a = softmax(&output_z); // prediction (ŷ)

        // -------- 2. Compute output error delta^(2) --------
        let mut delta_output = [0.0; 10];
        for o in 0..self.output {
            delta_output[o] = output_a[o] - target[o];
        }

        // -------- 3. Gradients for output weights & biases --------
        for o in 0..self.output {
            for h in 0..self.hidden {
                let grad = delta_output[o] * hidden_a[h];
                self.output_weights[o][h] -= lr * grad;
            }
            self.output_biase[o] -= lr * delta_output[o];
        }

        // -------- 4. Backpropagate error to hidden layer --------
        let mut delta_hidden = [0.0; HIDDEN];
        for h in 0..self.hidden {
            let mut sum = 0.0;
            for o in 0..self.output {
                sum += self.output_weights[o][h] * delta_output[o];
            }
            // Apply ReLU derivative
            delta_hidden[h] = if hidden_z[h] > 0.0 { sum } else { 0.0 };
        }

        // -------- 5. Gradients for hidden weights & biases --------
        for h in 0..self.hidden {
            for i in 0..self.input {
                let grad = delta_hidden[h] * input_pixels[i];
                self.hidden_weights[h][i] -= lr * grad;
            }
            self.hidden_biase[h] -= lr * delta_hidden[h];
        }
    }
}
