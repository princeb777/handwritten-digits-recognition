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

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;

impl Network {
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Save metadata (input, hidden, output)
        file.write_all(&(self.input as u64).to_le_bytes())?;
        file.write_all(&(self.hidden as u64).to_le_bytes())?;
        file.write_all(&(self.output as u64).to_le_bytes())?;

        // Save hidden_weights
        for row in &self.hidden_weights {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Save output_weights
        for row in &self.output_weights {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Save hidden_biase
        for &val in &self.hidden_biase {
            file.write_all(&val.to_le_bytes())?;
        }

        // Save output_biase
        for &val in &self.output_biase {
            file.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }

    pub fn load(path: PathBuf) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buf = [0u8; 8];

        // Read metadata
        file.read_exact(&mut buf)?;
        let input = u64::from_le_bytes(buf) as usize;
        file.read_exact(&mut buf)?;
        let hidden = u64::from_le_bytes(buf) as usize;
        file.read_exact(&mut buf)?;
        let output = u64::from_le_bytes(buf) as usize;

        // Read helper
        fn read_f32(file: &mut File) -> io::Result<f32> {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            Ok(f32::from_le_bytes(buf))
        }

        // hidden_weights
        let mut hidden_weights = [[0.0f32; 784]; HIDDEN];
        for row in &mut hidden_weights {
            for val in row.iter_mut() {
                *val = read_f32(&mut file)?;
            }
        }

        // output_weights
        let mut output_weights = [[0.0f32; HIDDEN]; 10];
        for row in &mut output_weights {
            for val in row.iter_mut() {
                *val = read_f32(&mut file)?;
            }
        }

        // hidden_biase
        let mut hidden_biase = [0.0f32; HIDDEN];
        for val in hidden_biase.iter_mut() {
            *val = read_f32(&mut file)?;
        }

        // output_biase
        let mut output_biase = [0.0f32; 10];
        for val in output_biase.iter_mut() {
            *val = read_f32(&mut file)?;
        }

        Ok(Self {
            input,
            hidden,
            output,
            hidden_weights,
            output_weights,
            hidden_biase,
            output_biase,
        })
    }
}
use std::io::Cursor;
impl Network {
    pub fn load_default(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor::new(bytes);
        let mut buf = [0u8; 8];

        // Read metadata
        cursor.read_exact(&mut buf)?;
        let input = u64::from_le_bytes(buf) as usize;
        cursor.read_exact(&mut buf)?;
        let hidden = u64::from_le_bytes(buf) as usize;
        cursor.read_exact(&mut buf)?;
        let output = u64::from_le_bytes(buf) as usize;

        // Helper
        fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(f32::from_le_bytes(buf))
        }

        // hidden_weights
        let mut hidden_weights = [[0.0f32; 784]; HIDDEN];
        for row in &mut hidden_weights {
            for val in row.iter_mut() {
                *val = read_f32(&mut cursor)?;
            }
        }

        // output_weights
        let mut output_weights = [[0.0f32; HIDDEN]; 10];
        for row in &mut output_weights {
            for val in row.iter_mut() {
                *val = read_f32(&mut cursor)?;
            }
        }

        // hidden_biase
        let mut hidden_biase = [0.0f32; HIDDEN];
        for val in hidden_biase.iter_mut() {
            *val = read_f32(&mut cursor)?;
        }

        // output_biase
        let mut output_biase = [0.0f32; 10];
        for val in output_biase.iter_mut() {
            *val = read_f32(&mut cursor)?;
        }

        Ok(Self {
            input,
            hidden,
            output,
            hidden_weights,
            output_weights,
            hidden_biase,
            output_biase,
        })
    }
}
