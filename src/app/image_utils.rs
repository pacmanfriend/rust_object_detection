use anyhow::{Result, anyhow};
use image::{ImageBuffer, Rgb, RgbImage};
use tch::{Device, Kind, Tensor};

pub fn preprocess_image(image_data: &[u8], device: Device) -> Result<Tensor> {
    // Загрузка изображения из байтов
    let img = image::load_from_memory(image_data)
        .map_err(|e| anyhow!("Не удалось загрузить изображение: {}", e))?;

    // Конвертация в RGB если необходимо
    let rgb_img = img.to_rgb8();

    // Изменение размера до 224x224 (стандартный размер для VGG)
    let resized =
        image::imageops::resize(&rgb_img, 224, 224, image::imageops::FilterType::Lanczos3);

    // Конвертация в тензор
    let tensor = image_to_tensor(&resized, device)?;

    // Нормализация ImageNet
    let normalized = normalize_imagenet(tensor)?;

    // Добавление batch dimension
    let batched = normalized.unsqueeze(0);

    Ok(batched)
}

fn image_to_tensor(img: &RgbImage, device: Device) -> Result<Tensor> {
    let (width, height) = img.dimensions();
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    // Конвертация из HWC в CHW формат и в float
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let value = pixel[channel] as f32 / 255.0;
                data.push(value);
            }
        }
    }

    let tensor = Tensor::from_slice(&data)
        .reshape(&[3, height as i64, width as i64])
        .to_device(device);

    Ok(tensor)
}

fn normalize_imagenet(tensor: Tensor) -> Result<Tensor> {
    // ImageNet нормализация
    // mean = [0.485, 0.456, 0.406]
    // std = [0.229, 0.224, 0.225]

    let mean = Tensor::from_slice(&[0.485, 0.456, 0.406])
        .reshape(&[3, 1, 1])
        .to_device(tensor.device());

    let std = Tensor::from_slice(&[0.229, 0.224, 0.225])
        .reshape(&[3, 1, 1])
        .to_device(tensor.device());

    let normalized = (tensor - mean) / std;

    Ok(normalized)
}
