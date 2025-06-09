use anyhow::Result;
use axum::{
    Router,
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use nvml_wrapper::Nvml;
use serde::Serialize;
use std::sync::Arc;
use tch::Device;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

mod gpu_info;
mod image_utils;
mod vgg;

use crate::app::vgg::ImagenetPredict;
use crate::app::{gpu_info::GpuInfo, image_utils::save_image, vgg::VGG19Model};

#[derive(Debug, Serialize)]
struct PredictionResponse {
    predicted_class: String,
    confidence: f64,
    predictions: Vec<ImagenetPredict>,
    gpu_info: GpuInfo,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    gpu_info: GpuInfo,
}

#[derive(Clone)]
struct AppState {
    model: Arc<VGG19Model>,
    class_names: Arc<Vec<String>>,
    nvml: Arc<Nvml>,
}

pub async fn start_app() -> Result<()> {
    // Инициализация логирования
    tracing_subscriber::fmt::init();

    info!("Инициализация VGG19 Image Classification API...");

    // Инициализация NVML для мониторинга GPU
    let nvml = Arc::new(Nvml::init().map_err(|e| {
        warn!("Не удалось инициализировать NVML: {}", e);
        e
    })?);

    // Определение устройства (GPU или CPU)
    let device = if tch::Cuda::is_available() {
        info!("CUDA доступна, используем GPU");
        Device::Cuda(0)
    } else {
        info!("CUDA недоступна, используем CPU");
        Device::Cpu
    };

    // Загрузка модели VGG19
    info!("Загрузка модели VGG19...");
    let model = Arc::new(VGG19Model::new(device)?);

    // Загрузка имен классов ImageNet
    let class_names = Arc::new(load_imagenet_classes());

    let state = Arc::new(AppState {
        model,
        class_names,
        nvml,
    });

    // Создание маршрутов
    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .route("/predict", post(predict_image))
        .route("/gpu-info", get(gpu_info_endpoint))
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!("Сервер запускается на http://127.0.0.1:8000");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let gpu_info = GpuInfo::new(&state.nvml);
    Json(HealthResponse {
        status: "OK".to_string(),
        gpu_info,
    })
}

async fn gpu_info_endpoint(State(state): State<Arc<AppState>>) -> Json<GpuInfo> {
    Json(GpuInfo::new(&state.nvml))
}

async fn predict_image(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut image_data: Option<Vec<u8>> = None;

    // Извлечение изображения из multipart form data
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Ошибка при чтении multipart данных: {}", e),
            }),
        )
    })? {
        if field.name() == Some("image") {
            let data = field.bytes().await.map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("Ошибка при чтении файла: {}", e),
                    }),
                )
            })?;

            image_data = Some(data.to_vec());
            break;
        }
    }

    let image_data = image_data.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Файл изображения не найден. Используйте поле 'image'".to_string(),
            }),
        )
    })?;

    let image_path = save_image(&image_data).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Ошибка при обработке изображения: {}", e),
            }),
        )
    })?;

    let predictions = state.model.predict_imagenet(image_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Ошибка предсказания: {}", e),
            }),
        )
    })?;

    let predicted_class = predictions[0].class.clone();
    let confidence = predictions[0].probability;

    // Получение информации о GPU
    let gpu_info = GpuInfo::new(&state.nvml);

    Ok(Json(PredictionResponse {
        predicted_class,
        confidence,
        predictions,
        gpu_info,
    }))
}

fn load_imagenet_classes() -> Vec<String> {
    // Базовые классы ImageNet (упрощенная версия)
    // В реальном проекте загрузите полный список из файла
    vec![
        "tench".to_string(),
        "goldfish".to_string(),
        "great white shark".to_string(),
        "tiger shark".to_string(),
        "hammerhead".to_string(),
        "electric ray".to_string(),
        "stingray".to_string(),
        "cock".to_string(),
        "hen".to_string(),
        "ostrich".to_string(),
        // ... добавьте остальные 1000 классов
        // Для демонстрации добавляем только несколько
    ]
    .into_iter()
    .chain((10..1000).map(|i| format!("class_{}", i)))
    .collect()
}
