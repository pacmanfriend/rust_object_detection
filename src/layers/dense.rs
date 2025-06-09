use cudarc::driver::{CudaContext, CudaSlice, DriverError, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::layers::utils::create_random_tensor;

const DENSE_LAYER_KERNEL: &str = "
extern \"C\" __global__ void dense_forward(
    const float* input,     // [batch_size, input_size]
    const float* weights,   // [input_size, output_size]
    const float* bias,      // [output_size]
    float* output,          // [batch_size, output_size]
    int batch_size,
    int input_size,
    int output_size,
    int activation_type     // 0=linear, 1=relu, 2=sigmoid, 3=tanh
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron index

    if (row < batch_size && col < output_size) {
        float sum = 0.0f;

        // Матричное умножение: output[row][col] = sum(input[row][k] * weights[k][col])
        for (int k = 0; k < input_size; k++) {
            sum += input[row * input_size + k] * weights[k * output_size + col];
        }

        // Добавляем bias
        sum += bias[col];

        // Применяем функцию активации
        float activated;
        switch (activation_type) {
            case 0: // Linear
                activated = sum;
                break;
            case 1: // ReLU
                activated = fmaxf(0.0f, sum);
                break;
            case 2: // Sigmoid
                activated = 1.0f / (1.0f + expf(-sum));
                break;
            case 3: // Tanh
                activated = tanhf(sum);
                break;
            default:
                activated = sum;
        }

        output[row * output_size + col] = activated;
    }
}

extern \"C\" __global__ void dense_backward_weights(
    const float* input,        // [batch_size, input_size]
    const float* grad_output,  // [batch_size, output_size]
    float* grad_weights,       // [input_size, output_size]
    int batch_size,
    int input_size,
    int output_size
) {
    int input_idx = blockIdx.y * blockDim.y + threadIdx.y;   // input dimension
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;  // output dimension

    if (input_idx < input_size && output_idx < output_size) {
        float grad_sum = 0.0f;

        // Градиент по весам: grad_W[i][j] = sum_batch(input[batch][i] * grad_output[batch][j])
        for (int batch = 0; batch < batch_size; batch++) {
            grad_sum += input[batch * input_size + input_idx] *
                       grad_output[batch * output_size + output_idx];
        }

        grad_weights[input_idx * output_size + output_idx] = grad_sum;
    }
}

extern \"C\" __global__ void dense_backward_bias(
    const float* grad_output,  // [batch_size, output_size]
    float* grad_bias,         // [output_size]
    int batch_size,
    int output_size
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_idx < output_size) {
        float grad_sum = 0.0f;

        // Градиент по bias: grad_b[j] = sum_batch(grad_output[batch][j])
        for (int batch = 0; batch < batch_size; batch++) {
            grad_sum += grad_output[batch * output_size + output_idx];
        }

        grad_bias[output_idx] = grad_sum;
    }
}

extern \"C\" __global__ void dense_backward_input(
    const float* weights,      // [input_size, output_size]
    const float* grad_output,  // [batch_size, output_size]
    float* grad_input,        // [batch_size, input_size]
    int batch_size,
    int input_size,
    int output_size
) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && input_idx < input_size) {
        float grad_sum = 0.0f;

        // Градиент по входу: grad_input[batch][i] = sum_j(weights[i][j] * grad_output[batch][j])
        for (int j = 0; j < output_size; j++) {
            grad_sum += weights[input_idx * output_size + j] *
                       grad_output[batch_idx * output_size + j];
        }

        grad_input[batch_idx * input_size + input_idx] = grad_sum;
    }
}
";

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Linear = 0,
    ReLU = 1,
    Sigmoid = 2,
    Tanh = 3,
}

pub struct DenseLayer {
    weights: CudaSlice<f32>, // [input_size, output_size]
    bias: CudaSlice<f32>,    // [output_size]
    input_size: usize,
    output_size: usize,
    activation: Activation,
}

impl DenseLayer {
    pub fn new(
        ctx: &Arc<CudaContext>,
        input_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Result<Self, DriverError> {
        let stream = ctx.default_stream();

        // Инициализация весов методом Xavier/Glorot
        let xavier_std = (2.0 / (input_size + output_size) as f32).sqrt();
        let mut weights_host = vec![0.0f32; input_size * output_size];
        let bias_host = vec![0.0f32; output_size];

        // Простая инициализация (в реальности используйте генератор случайных чисел)
        for i in 0..weights_host.len() {
            weights_host[i] = (((i * 7 + 13) % 1000) as f32 / 1000.0 - 0.5) * xavier_std * 2.0;
        }

        let weights = stream.memcpy_stod(&weights_host)?;
        let bias = stream.memcpy_stod(&bias_host)?;

        Ok(DenseLayer {
            weights,
            bias,
            input_size,
            output_size,
            activation,
        })
    }

    pub fn forward(
        &self,
        ctx: &Arc<CudaContext>,
        input: &CudaSlice<f32>, // [batch_size, input_size]
        batch_size: usize,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let stream = ctx.default_stream();
        let output_size_total = batch_size * self.output_size;
        let mut output = unsafe { stream.alloc::<f32>(output_size_total) }?;

        // Компилируем и загружаем kernel
        let ptx = compile_ptx(DENSE_LAYER_KERNEL).unwrap();
        let module = ctx.load_module(ptx)?;
        let kernel = module.load_function("dense_forward")?;

        let block_size = 16;
        let grid_x = (self.output_size + block_size - 1) / block_size;
        let grid_y = (batch_size + block_size - 1) / block_size;

        let mut i = 1;

        while i <= 32 {
            let block_size = 32 * i;

            let block_dim = (block_size as u32, 1, 1);
            let grid_dim = (output_size_total as u32 + block_dim.0 - 1 / 1, 1, 1);

            let config = LaunchConfig {
                grid_dim,
                block_dim,
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&kernel);
            builder.arg(input); // input
            builder.arg(&self.weights); // weights
            builder.arg(&self.bias); // bias
            builder.arg(&mut output); // output
            builder.arg(&batch_size); // batch_size
            builder.arg(&self.input_size); // input_size
            builder.arg(&self.output_size); // output_size
            builder.arg(&1); // activation_type

            let start_event =
                &stream.record_event(Option::from(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;
            let end_event =
                &stream.record_event(Option::from(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

            start_event.record(&stream)?;
            unsafe { builder.launch(config) }?;
            end_event.record(&stream)?;

            stream.synchronize()?;

            let time = start_event.elapsed_ms(&end_event)?;

            println!(
                "computed elements: {:}, block size: {:}, kernel time {:} ms",
                output_size_total, block_size, time
            );

            i = i * 2;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense() -> Result<(), DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Параметры слоя
        let batch_size = 1000;
        let input_size = 1000;
        let hidden_size = 1000;

        // let total_input_size = batch_size * input_size * hidden_size;
        // println!("Total input size: {}", total_input_size);

        let layer1 = DenseLayer::new(&ctx, input_size, hidden_size, Activation::ReLU)?;

        let input_data = create_random_tensor(batch_size * input_size);
        let input = stream.memcpy_stod(&input_data)?;

        layer1.forward(&ctx, &input, batch_size)?;

        Ok(())
    }
}

// Пример использования
