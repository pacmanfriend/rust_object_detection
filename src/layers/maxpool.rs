use cudarc::driver::{CudaContext, CudaSlice, DriverError, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::layers::utils::create_random_tensor;

const MAXPOOL_KERNEL: &str = "
extern \"C\" __global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total output elements
    int total_output_elements = batch_size * channels * output_height * output_width;
    
    if (idx >= total_output_elements) return;
    
    // Decode output position
    int batch = idx / (channels * output_height * output_width);
    int remaining = idx % (channels * output_height * output_width);
    int channel = remaining / (output_height * output_width);
    remaining = remaining % (output_height * output_width);
    int out_y = remaining / output_width;
    int out_x = remaining % output_width;
    
    // Calculate input region bounds
    int start_y = out_y * stride - padding;
    int start_x = out_x * stride - padding;
    int end_y = start_y + kernel_size;
    int end_x = start_x + kernel_size;
    
    // Clamp to input bounds
    start_y = max(0, start_y);
    start_x = max(0, start_x);
    end_y = min(input_height, end_y);
    end_x = min(input_width, end_x);
    
    // Find maximum value in the pooling window
    float max_val = 0;
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            int input_idx = batch * (channels * input_height * input_width) +
                           channel * (input_height * input_width) +
                           y * input_width + x;
            max_val = fmaxf(max_val, input[input_idx]);
        }
    }
    
    output[idx] = max_val;
}
";

#[derive(Debug, Clone)]
pub struct MaxPool2DConfig {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Default for MaxPool2DConfig {
    fn default() -> Self {
        Self {
            kernel_size: 2,
            stride: 2,
            padding: 0,
        }
    }
}

pub struct MaxPool2D {
    config: MaxPool2DConfig,
}

impl MaxPool2D {
    pub fn new(ctx: &Arc<CudaContext>, config: MaxPool2DConfig) -> Result<Self, DriverError> {
        // Compile and load the kernel
        Ok(Self { config })
    }

    pub fn forward(
        &self,
        ctx: &Arc<CudaContext>,
        input: &CudaSlice<f32>,
        batch_size: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let stream = ctx.default_stream();

        // Calculate output dimensions
        let output_height = (input_height + 2 * self.config.padding - self.config.kernel_size)
            / self.config.stride
            + 1;
        let output_width = (input_width + 2 * self.config.padding - self.config.kernel_size)
            / self.config.stride
            + 1;
        let output_size = batch_size * channels * output_height * output_width;

        // Allocate output buffer
        let mut output = stream.alloc_zeros::<f32>(output_size)?;

        // Get the kernel function
        let ptx = compile_ptx(MAXPOOL_KERNEL).unwrap();
        let module = ctx.load_module(ptx)?;
        let kernel = module.load_function("maxpool2d_kernel")?;

        let mut i = 1;

        while i <= 32 {
            // Launch configuration
            let block_size = 32 * i;
            let grid_size = (output_size + block_size - 1) / block_size;

            let cfg = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // Launch kernel
            let mut builder = stream.launch_builder(&kernel);
            builder.arg(input);
            builder.arg(&mut output);
            builder.arg(&batch_size);
            builder.arg(&channels);
            builder.arg(&input_height);
            builder.arg(&input_width);
            builder.arg(&output_height);
            builder.arg(&output_width);
            builder.arg(&self.config.kernel_size);
            builder.arg(&self.config.stride);
            builder.arg(&self.config.padding);

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

    pub fn output_shape(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let output_height = (input_height + 2 * self.config.padding - self.config.kernel_size)
            / self.config.stride
            + 1;
        let output_width = (input_width + 2 * self.config.padding - self.config.kernel_size)
            / self.config.stride
            + 1;
        (output_height, output_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() -> Result<(), DriverError> {
        // Test configuration
        let config = MaxPool2DConfig {
            kernel_size: 2,
            stride: 2,
            padding: 0,
        };

        let ctx = CudaContext::new(0)?;

        let maxpool = MaxPool2D::new(&ctx, config)?;
        let stream = ctx.default_stream();

        let batch_size = 100;
        let channels = 10;
        let height = 100;
        let width = 100;
        let input_size = batch_size * channels * height * width;

        let input_data = create_random_tensor(input_size);

        let input_gpu = stream.memcpy_stod(&input_data)?;

        maxpool.forward(&ctx, &input_gpu, batch_size, channels, height, width)?;

        println!("MaxPool2D test passed! ");

        Ok(())
    }
}
