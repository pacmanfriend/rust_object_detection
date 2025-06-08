use anyhow::Result;
use axum::{
    Router,
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tch::{Device, Kind, Tensor, nn, vision};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

mod gpu_info;
mod image_utils;
mod vgg;

use crate::app::{gpu_info::GpuInfo, image_utils::preprocess_image, vgg::VGG19Model};

#[derive(Debug, Serialize)]
struct PredictionResponse {
    predicted_class: String,
    confidence: f64,
    top_5_predictions: Vec<ClassPrediction>,
    gpu_info: GpuInfo,
}

#[derive(Debug, Serialize)]
struct ClassPrediction {
    class_name: String,
    confidence: f64,
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

// #[tokio::main]
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

    let state = AppState {
        model,
        class_names,
        nvml,
    };

    // Создание маршрутов
    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .route("/predict", post(predict_image))
        .route("/gpu-info", get(gpu_info_endpoint))
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!("Сервер запускается на http://0.0.0.0:3000");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let gpu_info = GpuInfo::new(&state.nvml);
    Json(HealthResponse {
        status: "OK".to_string(),
        gpu_info,
    })
}

async fn gpu_info_endpoint(State(state): State<AppState>) -> Json<GpuInfo> {
    Json(GpuInfo::new(&state.nvml))
}

async fn predict_image(
    State(state): State<AppState>,
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

    // Предобработка изображения
    let tensor = preprocess_image(&image_data, state.model.device()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Ошибка при обработке изображения: {}", e),
            }),
        )
    })?;

    // Предсказание
    let predictions = state.model.predict(&tensor).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Ошибка предсказания: {}", e),
            }),
        )
    })?;

    // Получение топ-5 предсказаний
    let top_5_indices = predictions.argsort(-1, true).select(1, 0).slice(1, 0, 5, 1);

    let top_5_probs = predictions
        .softmax(-1, Kind::Float)
        .gather(1, &top_5_indices, false)
        .squeeze_dim(0);

    let top_5_indices: Vec<i64> = top_5_indices.squeeze_dim(0).into();
    let top_5_probs: Vec<f64> = top_5_probs.into();

    let mut top_5_predictions = Vec::new();
    for (i, &prob) in top_5_probs.iter().enumerate() {
        let class_idx = top_5_indices[i] as usize;
        let class_name = state
            .class_names
            .get(class_idx)
            .unwrap_or(&format!("Class {}", class_idx))
            .clone();

        top_5_predictions.push(ClassPrediction {
            class_name,
            confidence: prob,
        });
    }

    let predicted_class = top_5_predictions[0].class_name.clone();
    let confidence = top_5_predictions[0].confidence;

    // Получение информации о GPU
    let gpu_info = GpuInfo::new(&state.nvml);

    Ok(Json(PredictionResponse {
        predicted_class,
        confidence,
        top_5_predictions,
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
