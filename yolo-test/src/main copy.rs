use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, IxDyn, s, Axis};
use ort::{Environment, SessionBuilder, Value};
use std::sync::Arc;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;

fn main() {
    // Example usage
    // Simulating image data from the frontend
    let buf = simulate_frontend_upload();
    
    // Run the detection on the image
    let boxes = detect_objects_on_image(buf);
    
    // Output the detected objects and their bounding boxes
    for (x1, y1, x2, y2, label, prob) in boxes {
        println!("Detected: {} (prob: {:.2}) at [{}, {}, {}, {}]", label, prob, x1, y1, x2, y2);
    }
}

// Simulate receiving image data from the frontend
fn simulate_frontend_upload() -> Vec<u8> {
    let base64_data = "data:image/webp;base64,UklGRoa5AABXRUJQVlA4WAoAAAAgAAAAfwIA3wEASUNDUMgBAAAAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADZWUDggmLcAAFA3A50BKoAC4AE+KRSIQqGhIRJJHVgYAoSytYfdLpAWGaZs1+msy/sl1I6R9X//BzI++f9vyhvbPW0+Xv/K9Wv9E/4fsP+Wb09f2z0vebR62f716PPp/+vD//+lb9bP7wPh5z9D5T44/mvYP+X/fP8r81vaZ/5vItvm9Ff59+lP5X+f9YvIv8//yP/N6l/5h/bP+R6iE19Rr/s/c779P43/g/wft/+sf5z/pe4P/PP6x/xfXP/r+WT+H/4n7ffAN/SP8N+znu6f7H/4/3Hqf/Xv95/9P9r8Dv7Gf+QdzaU9L+b1+d5u/H/WZD6WPL3yagcdhM7ctlFa/UipbO3LTD75tVW+h+LYpxxsEqXptEQnH/fuQ5Ffmw1//pJ5P/axtx5LaX+FigrT8c7+RhlnfRE39pPY3GpS6EeNkwTeHG8t8l9+/ofAs2eDxVNEu/17o4izqKvaMcRFvmqDXsWoVcAHXRLHfwSrxPebmir1/h8uZVtfX3vSpqni9YG5XGYEvyZzcM+CBIzeTR/8voFlW9dJ4g5IRw7jKVWyd3svdEs0iN0B0buls4P/DuQLbD518eofqjiVxKahcDXE7tlYFPOXjaGMuKU5rVRlpi8q00Q+yhQJkEgE/tcEanDIgT6U5zn24rki0FaSV3Vse/3yeHHrj3FssKJJa00hdnw9a+dDP2AjydjP6hmgK2ommlw0c+2y+8eoPICJxcWwlAUpFrr+h+CFCwvgIHjL/b1I9d0y3iQJL4y6D1TJzwL/INeIKW/5wilM/WbNAUf/EasW4JWMl9DBrD4oN3p2u9YdKdwNDqyGhCq/sa0gRN8bKOVpsgxGtZL6eNpoQAuNCvZ9QMf+qHlwG/fpvVWaI24uFRTX9mgO36pdssgFCIVLFfkIX/jd3qqd0nUILzxyCxItNH8NlXRCd+mRSJAy1pYj4o3ySk4q/Y2yGkTwSwp50slnqLln67lDUJ/2xyxRsknWFPEysFOHwu+ZpVcVkD+TLVRjA+wXakwmBNG58fH7n/zT5eVHBu2r7sXjzxsy7F9DPHFuEakVz/lTHZwHKXDBgRern4yrdbZhPZdqUVLrXWXAVPDWorDo4Em0qw/8bYV72Q9xQmHejOV9PHoOrmIjrXG6D+ULze7+Nm9Fbd4Xd5LB32HpDImvtkKCoNSszckxGhEzUWNzkwFgjNmuVuQw/gQC4j2r0wuXTu863Sq2Tfei55niJVE9VEsWQLAHT6YTONUD5kNQ+MIAbtctJTsqnpFzs5/UlrBXNnZypS+1z8mdYfObhbX1LNpfaGQVa6MqcQ3etgjj+glcUaqQz6TZi15+ystFk+cqf13615W/veTNhMzyXjx8hFCyNNUVmsezmoG9vg68VhINerd+r3As72dcoeap2cPybGds//AsCbmGtEZAAHgAlZPHrUp7nQR1mOZS10wCCuqyj2CyFkpixVf9YrE9vr1drSZeLbZYCmwTRLIyA2tAM+5oY72WHV9tNXeMsqs1rlvghsPaH2LXHm9c3BpqNHte5JercUrhKbYrIXZ0pf2a58XB6kwe0zjrBI9qa8AmKqv57GasIucV6Ll+L+J+/nbr/hY6tDF99wrafAIfZvXu0r2VVvmYzMaPmrrrvnuXW2l95/XTUgsLREkjNDwRYIJhGVwdaOnSniIjmN9DEbN7ft7qAMoNJQiKNbfwkEyOy2Wj7qeQ2ro7weeRDpilrAbXAYhgrX04c1tnPdP2P2FO0cqD3GTfy6TVI15liP+M8ajGWLxQyoM43Q1eMR3opTTpm49WiUXMfIWe8ilaWR1LUmBqtR3I7EXgL3d5lUi080TT65WIePXEIso23tDLD/hqXONrgYgfc38t2yVBbAfTCDywHG3R1ykF9HaSalOTvSOirdLiLD86eQEwJZiPdJMv8rQq3eSQCKnyRGVwdw4nGjo8ytbJ1REOumyR2QtJRHpHr+Vnp5u8XxbFlEjpBvjo/1WeAZiA91l0Tb/sS8DSkEOfKfY45sZ9WO8+RxjB1mAq7+jnC8j+QCSZv2N8mh31zx3BDJ7+jvGuOUDavWtC2vI/1wqR6/Xb7/pnGllA1Gy2PX81Bvz6YgFHq8TaCJV5qyp12qlNAeAWMwKZto8vveuS/W6bfs0GaOjJgIfnxsykRmw64H2dcgJYYQwXbu7emRAQ2hfSy8c99KOqOiKfNHrFYDh/oIaZvpfsFsjE+qaRFiL/hZTSTkmBYdoXpeJNm11Z8kqvz42e4PadxSencV9eCxpU7Q4A7TpaLVIAGx+7wkIczLHJjEeuPvJOPwYDWlOrB2/2YqDClI+DJGW8Xz2n+VyZASWfeFh6Sf4UhcNBdur+Wuoo5Eob26Mh2F3BEFaXgG3KTChQJQrOja3/75eR142heR997GZdPnx9nyrALUpkx9MpK6ELO/kPehqBaijgvFvKFAvDRMVt4Oh6BDiA7JZ8SDysmiLZbKhaTw3r3rF2Cj2h+C2X7qXCoNJzwkGaxtv7kUgTXkX8kDPgmrh2Pe8on7SpibyHdNdRw6BuaY/ej4Nh5HE1GOArXJIuX2h4TOkVZhfGzM7834MOyHRgEAo93c89IKiAOt4NBdJseVerCXpVuNOc5cGc83knI/AghNwO89/gCh911QZM5yhY4Z3227gMQFCbg9xfW71ooeMpHHWe8ksdPMpwrFcPqM8SJh+puqR1isdCUrY6Czj9QIP6iEUGGR2D9NcJbnx2R8zHv4DedF8pkqcXYZFmg2kWfzJ2jTQSXQblySvFEQlLu0ArBNtRX1IUMrynfVkhSFayOzphFyUyoYYdCX8ZG5YPMRSykpTpfa3hEED9l0Gw59AhAf8EeVwL1VSCitcZ993L2RLqTiwchOc8zvRWrlfMwHWuAXJgEm3EePInUQON4vA9qD0Szl8osofVZtVXnD21wRIIt3rT3bjlVLlocoRbs0dXq0rTQ6FVglVQ7THMLqoGAl7CtaScaa0NwQf0zWiSY6Dhpx9xd9rj3OAFeApsVNiTX/o2+bsWt6K4fW8VyCzJsfY0O6cDCdqhktZeStd0n8bA5afnDXa8GW7RJ8D+kyWI+wjDYcTYoFQbtw97J8Q6dInJlI7mkVTfUGrWDo0YK8nH4vmz8OnkgYYiH8lewwIbRy5sRWoEp9D5TeycyK8LPnpcVedfSDbwmVZZvOz6+VQdjJCtfKhASQ4k3J5GYwViErkNXPx6vnpIBsOnrPvnhTaiK1UKcwqs8mq5D+3AlYEKA1Us4lWiVkYFbRNWnLEY7ZCIcNq7ImG+e/tcaUJ2GZHv49VyIlrxNydxnbg8+MNzQcIVeYeNjFHQYqi9w5nKnlYzDdrqJAFLd0zyBOAR2J6Gf2wr/ElHk/hKAMItcIqkUPzB4BCs+yp4x9dgf0boTjU1Nfm9hvahmjC6DK25YPOLi0C4gAcAtHW3Z0XYBPDBk6/diSJV2Jt4DrIeD56GPO3KxUgvnWrHtVr+M+1PgjcejA9+nCB0415DitgvlOdMjWa153WA68wPyqn1fkjJgNHRIOlV/bXvak93ANcejLSWU5igCQiK1XHD42Iu5o28axyAkL9RRdE5yhNQbQwvpe6lX4gILDjSUbSJzw3Kyw0AuH61f1eXY8TLdlv4vwE7sU/fGqZZ27w0T4MHMGKLhERNwlSqloH18XDGzoxP+hRMiLeFaT71Rv9ho1hvek2LTZ3OXFYFam68zHUgWYVV2tfm5p3bpDNf0aqc4k3CdlFc7kmra7z9+iLFaddUoLCoiRJ+J6ZRtvxUhCfls8KcM/+AMw1kic5vYAf8lVgqp/UAF64YSAbbTMVe6mpqgKfLB7CG/3mAyKz0+pv3M+fBuNRnPY6T1FK9+5BZF9bpzbpNFv36RjS0s3+WnfF8ElRoq0zW1JNukiS2gTN5XCiGoL9YtCmb7q9Iq1arPJ5eYxwrgQu31cjDBF2ONgaTE5Eu7dyF1vgA+Ufb2v1+Qeyngqx142y64/Ak717Wtu/mlAHiD2giTkKDIzRrOWxXc9yxckEjkc6mrIspBi0jjRLx4rAEw/I0eJJx9YsSRRUoFsFaKTIfLga/DeSSXVQyXQjpuwMeh5JzhriEynh6mgx65U5mqmx5a3g3hL8X7uEAVqM6/O67j3UcHYKUUrkCZzbtKL5zC4IxptfsLPvTeG4XXZAmxKP4GncRJU7uTYtI5MfnUMnEfApUHVtat7oTtRj72Z5SlK5BrQDYvt8olV5WPVbevwkTZi7wiJBFBS4YE9etlNVsOqIaBqb1tDffffah+lZ6gjcAE+VRI2aYUy7i9A6wG+qrGHReRGrJyMbFK2BX5";
    let encoded = base64_data.split(',').nth(1).expect("Invalid base64 data URL format");
    
    // Add padding if missing
    let padded_encoded = match encoded.len() % 4 {
        2 => format!("{}==", encoded),
        3 => format!("{}=", encoded),
        _ => encoded.to_string(),
    };

    STANDARD.decode(&padded_encoded).expect("Failed to decode base64 data")
}

fn detect_objects_on_image(buf: Vec<u8>) -> Vec<(f32, f32, f32, f32, &'static str, f32)> {
    let (input, img_width, img_height) = prepare_input(buf);
    let output = run_model(input);
    process_output(output, img_width, img_height)
}

fn prepare_input(buf: Vec<u8>) -> (Array<f32, IxDyn>, u32, u32) {
    let img = image::load_from_memory(&buf).unwrap();
    let (img_width, img_height) = (img.width(), img.height());
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.0;
        input[[0, 1, y, x]] = (g as f32) / 255.0;
        input[[0, 2, y, x]] = (b as f32) / 255.0;
    }
    (input, img_width, img_height)
}

fn run_model(input: Array<f32, IxDyn>) -> Array<f32, IxDyn> {
    let env = Arc::new(Environment::builder().with_name("YOLOv8").build().unwrap());
    let model = SessionBuilder::new(&env).unwrap().with_model_from_file("src/pretrained/yolov8n.onnx").unwrap();
    let input_as_values = &input.as_standard_layout();
    let model_inputs = vec![Value::from_array(model.allocator(), input_as_values).unwrap()];
    let outputs = model.run(model_inputs).unwrap();
    let output = outputs.get(0).unwrap().try_extract::<f32>().unwrap().view().t().into_owned();
    output
}

fn process_output(output: Array<f32, IxDyn>, img_width: u32, img_height: u32) -> Vec<(f32, f32, f32, f32, &'static str, f32)> {
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row.iter().skip(4).enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum }).unwrap();
        
        if prob < 0.5 {
            continue;
        }
        
        let label = YOLO_CLASSES[class_id];
        let xc = row[0] / 640.0 * (img_width as f32);
        let yc = row[1] / 640.0 * (img_height as f32);
        let w = row[2] / 640.0 * (img_width as f32);
        let h = row[3] / 640.0 * (img_height as f32);
        let x1 = xc - w / 2.0;
        let x2 = xc + w / 2.0;
        let y1 = yc - h / 2.0;
        let y2 = yc + h / 2.0;
        boxes.push((x1, y1, x2, y2, label, prob));
    }

    boxes.sort_by(|box1, box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len() > 0 {
        result.push(boxes[0]);
        boxes = boxes.iter().filter(|box1| iou(&boxes[0], box1) < 0.7).map(|x| *x).collect();
    }
    result
}

fn iou(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    intersection(box1, box2) / union(box1, box2)
}

fn union(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _) = *box2;
    let box1_area = (box1_x2 - box1_x1) * (box1_y1 - box1_y2);
    let box2_area = (box2_x2 - box2_x1) * (box2_y1 - box2_y2);
    box1_area + box2_area - intersection(box1, box2)
}

fn intersection(box1: &(f32, f32, f32, f32, &'static str, f32), box2: &(f32, f32, f32, f32, &'static str, f32)) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    (x2 - x1).max(0.0) * (y2 - y1).max(0.0)
}

const YOLO_CLASSES: [&str; 2] = [
    "cheque", "illegal drugs",
];
