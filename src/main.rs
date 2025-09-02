use crate::draw::draw_handles;
const DEFAULT_MODEL: &[u8] = include_bytes!(".././first.mnistai");

mod dataset;
mod draw;
mod network;
mod training;
mod utils;

fn main() {
    let ai = menu();
    let (mut rl, thread, mut canvas) = draw_handles();
    loop {
        let data = draw::do_drawing(&mut rl, &thread, &mut canvas);
        let g = ai.forward(data);
        println!(" pridiction = {:?}", utils::from_one_hot(g));
    }
}

fn menu() -> network::Network {
    println!("0 : Use Default MNIST model");
    println!("1 : Load the MNIST model");
    println!("2 : Train the MNIST model");

    let mut buff = String::new();

    // read menu choice
    loop {
        buff.clear();
        match std::io::stdin().read_line(&mut buff) {
            Ok(_) => break,
            Err(e) => eprintln!("Error reading input: {}. Try again.", e),
        }
    }
    if buff.trim() == "0" {
        println!("Loaded default model from embedded bytes");
        network::Network::load_default(DEFAULT_MODEL).unwrap()
    } else if buff.trim() == "1" {
        // collect all .mnistai files
        let current_dir = ".";
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(current_dir).unwrap() {
            let e = entry.unwrap();
            let path = e.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mnistai") {
                paths.push(path);
            }
        }

        if paths.is_empty() {
            println!("No saved models found. Falling back to training...");
            return training::do_things();
        }

        // show list
        for (i, path) in paths.iter().enumerate() {
            println!("{} : {:?}", i, path);
        }

        // select index
        let mut path_id = String::new();
        let index: usize;
        loop {
            path_id.clear();
            match std::io::stdin().read_line(&mut path_id) {
                Ok(_) => match path_id.trim().parse::<usize>() {
                    Ok(num) if num < paths.len() => {
                        index = num;
                        break; // valid input
                    }
                    _ => {
                        eprintln!(
                            "Invalid selection. Please enter a number between 0 and {}.",
                            paths.len() - 1
                        );
                    }
                },
                Err(e) => eprintln!("Error reading input: {}. Try again.", e),
            }
        }

        println!("Loading the model... this shouldn't take too long.");
        network::Network::load(paths[index].clone()).unwrap()
    } else if buff.trim() == "2" {
        println!("Training the model... you might want to grab a cup of coffee while it runs.");
        training::do_things()
    } else {
        eprintln!("Invalid option, quitting.");
        std::process::exit(1); // quit program
    }
}
