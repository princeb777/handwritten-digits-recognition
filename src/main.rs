use crate::draw::draw_handles;

mod dataset;
mod draw;
mod network;
mod training;
mod utils;

fn main() {
    let ai = training::do_things();
    let (mut rl, thread, mut canvas) = draw_handles();
    loop {
        let data = draw::do_drawing(&mut rl, &thread, &mut canvas);
        let g = ai.forward(data);
        println!("{:?}", utils::from_one_hot(g));
    }
}
