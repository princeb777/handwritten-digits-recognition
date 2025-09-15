# Handwritten Digits Recognition

This project is a neural network built with Rust to recognize handwritten digits. It is written completely from scratch and does not use any external AI-related crates.

## Data

The training and testing data for this project is sourced from the MNIST dataset.

To run the program, you need to obtain the data from the official MNIST website or other sources that provide the dataset in CSV format.

1.  Download the `train.csv` and `test.csv` files.
2.  Create a `data` folder in the root of the project directory.
3.  Place `train.csv` and `test.csv` inside the `data` folder.

## Building and Running

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)

### Building

To build the project, run the following command:

```bash
cargo build --release
```

### Running

To run the program, use the following command:

```bash
cargo run
```