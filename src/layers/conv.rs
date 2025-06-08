use cudarc::driver::{
    self, CudaContext, CudaEvent, CudaSlice, DriverError, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::layers::utils::create_random_tensor;

// CUDA kernel for 2D convolution
const CONV_KERNEL: &str = r#"
extern "C" __global__ void conv2d_forward(
    const float* input,     // Input tensor [batch, in_channels, height, width]
    const float* weight,    // Weights [out_channels, in_channels, kernel_h, kernel_w]
    const float* bias,      // Bias [out_channels]
    float* output,          // Output tensor [batch, out_channels, out_h, out_w]
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Calculate thread indices
    int batch = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_y = blockIdx.z * blockDim.x + threadIdx.x;
    int out_x = threadIdx.y;

    if (batch >= batch_size || out_ch >= out_channels ||
        out_y >= output_h || out_x >= output_w) {
        return;
    }

    float sum = 0.0f;

    // Perform convolution
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = out_y * stride_h - pad_h + ky;
                int in_x = out_x * stride_w - pad_w + kx;

                // Check bounds (zero padding)
                if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                    int input_idx = batch * (in_channels * input_h * input_w) +
                                   in_ch * (input_h * input_w) +
                                   in_y * input_w + in_x;

                    int weight_idx = out_ch * (in_channels * kernel_h * kernel_w) +
                                    in_ch * (kernel_h * kernel_w) +
                                    ky * kernel_w + kx;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias and store result
    sum += bias[out_ch];

    int output_idx = batch * (out_channels * output_h * output_w) +
                    out_ch * (output_h * output_w) +
                    out_y * output_w + out_x;

    output[output_idx] = sum;
}

extern "C" __global__ void conv2d_backward_input(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int output_h,
    int output_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int batch = blockIdx.x;
    int in_ch = blockIdx.y;
    int in_y = blockIdx.z * blockDim.x + threadIdx.x;
    int in_x = threadIdx.y;

    if (batch >= batch_size || in_ch >= in_channels ||
        in_y >= input_h || in_x >= input_w) {
        return;
    }

    float sum = 0.0f;

    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int out_y = (in_y + pad_h - ky) / stride_h;
                int out_x = (in_x + pad_w - kx) / stride_w;

                if (out_y >= 0 && out_y < output_h && out_x >= 0 && out_x < output_w &&
                    (in_y + pad_h - ky) % stride_h == 0 && (in_x + pad_w - kx) % stride_w == 0) {

                    int grad_output_idx = batch * (out_channels * output_h * output_w) +
                                         out_ch * (output_h * output_w) +
                                         out_y * output_w + out_x;

                    int weight_idx = out_ch * (in_channels * kernel_h * kernel_w) +
                                    in_ch * (kernel_h * kernel_w) +
                                    ky * kernel_w + kx;

                    sum += grad_output[grad_output_idx] * weight[weight_idx];
                }
            }
        }
    }

    int input_idx = batch * (in_channels * input_h * input_w) +
                   in_ch * (input_h * input_w) +
                   in_y * input_w + in_x;

    grad_input[input_idx] = sum;
}
"#;

pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub bias: bool,
}

pub struct Conv2dLayer {
    config: Conv2dConfig,
    weight: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
}

impl Conv2dLayer {
    pub fn new(
        ctx: &Arc<CudaContext>,
        config: Conv2dConfig,
        weight_data: Vec<f32>,
        bias_data: Option<Vec<f32>>,
    ) -> Result<Self, DriverError> {
        let stream = ctx.default_stream();

        // Validate weight dimensions
        let expected_weight_size =
            config.out_channels * config.in_channels * config.kernel_size.0 * config.kernel_size.1;
        assert_eq!(
            weight_data.len(),
            expected_weight_size,
            "Weight data size mismatch"
        );

        // Copy weights to device
        let weight = stream.memcpy_stod(&weight_data)?;

        // Copy bias to device if provided
        let bias = if let Some(bias_data) = bias_data {
            assert_eq!(
                bias_data.len(),
                config.out_channels,
                "Bias data size mismatch"
            );
            Some(stream.memcpy_stod(&bias_data)?)
        } else {
            None
        };

        Ok(Self {
            config,
            weight,
            bias,
        })
    }

    pub fn forward(
        &self,
        ctx: &Arc<CudaContext>,
        input: &CudaSlice<f32>,
        batch_size: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let stream = ctx.default_stream();

        // Calculate output dimensions
        let output_height = (input_height + 2 * self.config.padding.0 - self.config.kernel_size.0)
            / self.config.stride.0
            + 1;
        let output_width = (input_width + 2 * self.config.padding.1 - self.config.kernel_size.1)
            / self.config.stride.1
            + 1;

        let output_size = batch_size * self.config.out_channels * output_height * output_width;
        let mut output = stream.alloc_zeros::<f32>(output_size)?;

        // Compile and load kernel
        let ptx = compile_ptx(CONV_KERNEL).unwrap();
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("conv2d_forward")?;

        // Prepare bias (use zero bias if none provided)
        let zero_bias;
        let bias_ptr = if let Some(ref bias) = self.bias {
            bias
        } else {
            zero_bias = stream.alloc_zeros::<f32>(self.config.out_channels)?;
            &zero_bias
        };

        let mut i = 1;

        while i <= 32 {
            let block_size = 32 * i;
            let block_dim = (block_size as u32, 1, 1);
            let grid_dim = (((output_size as u32) + block_dim.0 - 1) / block_dim.0, 1, 1);

            let cfg = LaunchConfig {
                grid_dim,
                block_dim,
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&f);
            builder.arg(input);
            builder.arg(&self.weight);
            builder.arg(bias_ptr);
            builder.arg(&mut output);
            builder.arg(&batch_size);
            builder.arg(&self.config.in_channels);
            builder.arg(&self.config.out_channels);
            builder.arg(&input_height);
            builder.arg(&input_width);
            builder.arg(&self.config.kernel_size.0);
            builder.arg(&self.config.kernel_size.1);
            builder.arg(&output_height);
            builder.arg(&output_width);
            builder.arg(&self.config.stride.0);
            builder.arg(&self.config.stride.1);
            builder.arg(&self.config.padding.0);
            builder.arg(&self.config.padding.1);

            let start_event =
                stream.record_event(Option::from(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;
            let end_event =
                stream.record_event(Option::from(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

            start_event.record(&stream)?;
            unsafe { builder.launch(cfg) }?;
            end_event.record(&stream)?;

            stream.synchronize()?;

            let time = start_event.elapsed_ms(&end_event)?;

            println!(
                "computed elements: {:}, block size: {:}, kernel time {:} ms",
                output_size, block_size, time
            );

            i = i * 2;
        }

        Ok(output)
    }
}

pub fn create_identity_kernel(
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let mut weights = vec![0.0; out_channels * in_channels * kernel_size * kernel_size];
    let center = kernel_size / 2;

    for out_ch in 0..out_channels.min(in_channels) {
        let idx = out_ch * in_channels * kernel_size * kernel_size
            + out_ch * kernel_size * kernel_size
            + center * kernel_size
            + center;
        weights[idx] = 1.0;
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_layer() -> Result<(), DriverError> {
        // Initialize CUDA context
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Define convolution parameters
        let batch_size = 32;
        let in_channels = 64;
        let out_channels = 64;
        let input_height = 224;
        let input_width = 224;
        let kernel_size = 3;

        // println!("Initializing Conv2D layer...");
        // println!(
        //     "Input: {}x{}x{}x{}",
        //     batch_size, in_channels, input_height, input_width
        // );
        // println!(
        //     "Kernel: {}x{}x{}x{}",
        //     out_channels, in_channels, kernel_size, kernel_size
        // );

        // Create layer configuration
        let config = Conv2dConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (1, 1),
            padding: (1, 1),
            bias: true,
        };

        // Create random weights and bias
        let weight_data =
            create_random_tensor(out_channels * in_channels * kernel_size * kernel_size);
        let bias_data = create_random_tensor(out_channels);

        // Create convolution layer
        let conv_layer = Conv2dLayer::new(&ctx, config, weight_data, Some(bias_data))?;

        // Create input tensor
        let input_data =
            create_random_tensor(batch_size * in_channels * input_height * input_width);
        let input_tensor = stream.memcpy_stod(&input_data)?;

        // println!("Running forward pass...");

        // Forward pass
        // let start = std::time::Instant::now();
        let output =
            conv_layer.forward(&ctx, &input_tensor, batch_size, input_height, input_width)?;
        // let forward_time = start.elapsed();

        // Calculate output dimensions

        // println!("Forward pass completed in {:?}", forward_time);

        Ok(())
    }
}
