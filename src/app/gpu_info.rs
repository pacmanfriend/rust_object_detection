use nvml_wrapper::{Device, Nvml};
use serde::Serialize;
use tracing::warn;

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub cuda_available: bool,
    pub gpu_count: u32,
    pub gpus: Vec<GpuDevice>,
}

#[derive(Debug, Serialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub memory_total: u64,
    pub memory_used: u64,
    pub memory_free: u64,
    pub memory_usage_percent: f32,
    pub temperature: Option<u32>,
    pub power_usage: Option<u32>,
    pub power_limit: Option<u32>,
    pub utilization_gpu: Option<u32>,
    pub utilization_memory: Option<u32>,
}

impl GpuInfo {
    pub fn new(nvml: &Nvml) -> Self {
        let cuda_available = tch::Cuda::is_available();

        // if !cuda_available {
        //     return GpuInfo {
        //         cuda_available: false,
        //         gpu_count: 0,
        //         gpus: Vec::new(),
        //     };
        // }

        let device_count = match nvml.device_count() {
            Ok(count) => count,
            Err(e) => {
                warn!("Не удалось получить количество GPU устройств: {}", e);
                return GpuInfo {
                    cuda_available: true,
                    gpu_count: 0,
                    gpus: Vec::new(),
                };
            }
        };

        let mut gpus = Vec::new();

        for i in 0..device_count {
            if let Ok(device) = nvml.device_by_index(i) {
                if let Some(gpu_device) = Self::get_device_info(&device, i) {
                    gpus.push(gpu_device);
                }
            }
        }

        GpuInfo {
            cuda_available: true,
            gpu_count: device_count,
            gpus,
        }
    }

    fn get_device_info(device: &Device, index: u32) -> Option<GpuDevice> {
        let name = device.name().unwrap_or_else(|_| format!("GPU {}", index));

        let memory_info = device.memory_info().ok()?;
        let memory_total = memory_info.total / 1024 / 1024;
        let memory_used = memory_info.used / 1024 / 1024;
        let memory_free = memory_info.free / 1024 / 1024;
        let memory_usage_percent = (memory_used as f32 / memory_total as f32) * 100.0;

        let temperature = device
            .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
            .ok();

        let power_usage = device.power_usage().ok();
        let power_limit = device.power_management_limit().ok();

        let utilization = device.utilization_rates().ok();
        let (utilization_gpu, utilization_memory) = if let Some(util) = utilization {
            (Some(util.gpu), Some(util.memory))
        } else {
            (None, None)
        };

        Some(GpuDevice {
            index,
            name,
            memory_total,
            memory_used,
            memory_free,
            memory_usage_percent,
            temperature,
            power_usage,
            power_limit,
            utilization_gpu,
            utilization_memory,
        })
    }
}
