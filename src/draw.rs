use raylib::prelude::*;

const WIDTH: i32 = 280;
const HEIGHT: i32 = 280;
const SOURCE: Rectangle = Rectangle::new(0.0, 0.0, WIDTH as f32, -HEIGHT as f32);

pub fn draw_handles() -> (RaylibHandle, RaylibThread, RenderTexture2D) {
    let (mut rl, thread) = raylib::init()
        .size(WIDTH, HEIGHT)
        .title("Draw Canvas")
        .build();
    rl.set_target_fps(144);
    let mut canvas = rl
        .load_render_texture(&thread, WIDTH as u32, HEIGHT as u32)
        .unwrap();
    {
        let mut d = rl.begin_texture_mode(&thread, &mut canvas);
        d.clear_background(Color::BLACK);
    }
    (rl, thread, canvas)
}

pub fn do_drawing(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    canvas: &mut RenderTexture2D,
) -> [f32; 784] {
    let mut data: Vec<Color> = Vec::new();
    while !rl.window_should_close() {
        let mouse_down = rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT);
        let mouse_up = rl.is_mouse_button_released(MouseButton::MOUSE_BUTTON_LEFT);
        let clear_btn = rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_RIGHT);

        let mouse_pos = rl.get_mouse_position();
        if mouse_down {
            let mut d = rl.begin_texture_mode(&thread, canvas);
            d.draw_circle(mouse_pos.x as i32, mouse_pos.y as i32, 8.0, Color::WHITE);
        }
        if clear_btn {
            let mut d = rl.begin_texture_mode(&thread, canvas);
            d.clear_background(Color::BLACK);
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);
        d.draw_texture_rec(&canvas, SOURCE, Vector2::new(0.0, 0.0), Color::WHITE);

        // Example: Save screen image when releasing mouse
        if mouse_up {
            let mut img = canvas.load_image().unwrap();
            img.resize(28, 28);
            img.flip_vertical();
            println!("Captured image: {}x{}", img.width(), img.height());
            let g = img.get_image_data();
            data = g.to_vec();
            break;
        }
    }
    // let datas = data.into_vec();
    let mut new_data = [0f32; 784];
    for i in 0..28 {
        for j in 0..28 {
            let idx = (j * 28 + i) as usize;
            let c = data[idx];
            let r = c.r as f32 / 255.0;
            let g = c.g as f32 / 255.0;
            let b = c.b as f32 / 255.0;
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            new_data[idx] = gray;
        }
    }

    for (i, d) in new_data.iter().enumerate() {
        if i % 28 == 0 {
            println!("");
        }
        if *d > 0.0 {
            print!("0 ");
        } else {
            print!("_ ");
        }
    }

    return new_data;
}
