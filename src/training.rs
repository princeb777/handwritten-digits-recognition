const LEARNING_RATE: f32 = 0.001;
// [0.1, 0.03, 0.01, 0.003, 0.001]

use crate::dataset;
use crate::network;
use crate::utils;

pub fn do_things() -> network::Network {
    let train = dataset::data_loader("data/train.csv", 60000);
    let test = dataset::data_loader("data/test.csv", 10000);

    let mut new_net = network::Network::new(784, 64, 10);
    let epoch = 5;
    for _i in 0..epoch {
        for (lable, pixels) in &train {
            new_net.backprop(pixels, utils::one_hot(*lable as usize), LEARNING_RATE);
        }

        let mut corrects: f32 = 0.0;
        for (lable, pixels) in &test {
            let d = new_net.forward(*pixels);
            let d = utils::argmax(d);
            if *lable == d {
                corrects += 1.0;
            } else {
                // println!("{} :: {}", lable, d);
            }
        }

        println!("acc {}", corrects / 10000.0)
    }
    new_net
}
