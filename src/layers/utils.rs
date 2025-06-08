pub fn create_random_tensor(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..size)
        .map(|i| (i as f32 * PI / size as f32).sin() * 0.5 + 0.5)
        .collect()
}