// pub fn data_loader(path: &'static str, capacity: usize) -> Vec<(u8, [f32; 784])> {
//     println!("{}", path);
//     println!("Loading Samples");
//     let file = File::open(path).expect("file does not exist");
//     let reader = BufReader::new(file);

//     // Pre-allocate space for capacity MNIST samples
//     let mut dataset: Vec<(u8, [f32; 784])> = Vec::with_capacity(capacity);

//     for (i, line) in reader.lines().enumerate() {
//         let line = line.unwrap();

//         // Skip header
//         if i == 0 {
//             continue;
//         }

//         // Split row into label + 784 pixel values
//         let mut iter = line.split(',');
//         let label: u8 = iter.next().unwrap().parse().unwrap();

//         let mut pixels = [0.0f32; 784];
//         for (j, val) in iter.enumerate() {
//             let temp: f32 = val.parse().unwrap();
//             pixels[j] = temp / 255.0;
//         }

//         dataset.push((label, pixels));
//     }
//     println!("Loaded {} samples", dataset.len());
//     dataset
// }

pub fn load_train_data() -> Vec<(u8, [f32; 784])> {
    println!("Loading Training Data (embedded)");
    let data = include_str!(".././data/train.csv");
    let mut dataset: Vec<(u8, [f32; 784])> = Vec::with_capacity(60000);

    for (i, line) in data.lines().enumerate() {
        // Skip header
        if i == 0 {
            continue;
        }

        let mut iter = line.split(',');
        let label: u8 = iter.next().unwrap().parse().unwrap();

        let mut pixels = [0.0f32; 784];
        for (j, val) in iter.enumerate() {
            let temp: f32 = val.parse().unwrap();
            pixels[j] = temp / 255.0;
        }

        dataset.push((label, pixels));
    }

    println!("Loaded {} samples", dataset.len());
    dataset
}

pub fn load_test_data() -> Vec<(u8, [f32; 784])> {
    println!("Loading Testing Data (embedded)");

    // Embed the CSV file at compile time
    let data = include_str!(".././data/test.csv");

    let mut dataset: Vec<(u8, [f32; 784])> = Vec::with_capacity(10000);

    for (i, line) in data.lines().enumerate() {
        // Skip header
        if i == 0 {
            continue;
        }
        let mut iter = line.split(',');
        let label: u8 = iter.next().unwrap().parse().unwrap();

        let mut pixels = [0.0f32; 784];
        for (j, val) in iter.enumerate() {
            let temp: f32 = val.parse().unwrap();
            pixels[j] = temp / 255.0;
        }

        dataset.push((label, pixels));
    }
    println!("Loaded {} samples", dataset.len());
    dataset
}
