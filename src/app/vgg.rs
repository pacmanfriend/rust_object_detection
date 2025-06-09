use anyhow::Result;
use serde::Serialize;
use tch::nn::SequentialT;
use tch::vision::{imagenet, vgg::vgg19};
use tch::{Device, Tensor, nn, nn::ModuleT};

pub struct VGG19Model {
    model: SequentialT,
    device: Device,
}

#[derive(Debug, Serialize)]
pub struct ImagenetPredict {
    pub class: String,
    pub probability: f64,
}

unsafe impl Send for VGG19Model {}
unsafe impl Sync for VGG19Model {}

impl VGG19Model {
    pub fn new(device: Device) -> Result<Self> {
        let mut var_store = nn::VarStore::new(device);
        let model = vgg19(&var_store.root(), 1000);
        var_store.load("./src/vgg19.safetensors")?;
        var_store.freeze();

        Ok(Self { model, device })
    }

    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.model.forward_t(input, false);
        Ok(output)
    }

    pub fn predict_imagenet(&self, image_path: &str) -> Result<Vec<ImagenetPredict>> {
        let image = imagenet::load_image_and_resize224(image_path)?.to_device(self.device);

        let output = image
            .unsqueeze(0)
            .apply_t(&self.model, false)
            .to_device(self.device);

        let mut top_five = Vec::new();

        for (probability, class) in imagenet::top(&output, 5).iter() {
            let current_class = class.clone();
            let current_probability = probability.clone();

            let predict = ImagenetPredict {
                class: current_class,
                probability: current_probability,
            };

            top_five.push(predict);

            // println!("{:50} {:5.2}%", class, 100.0 * probability)
        }

        Ok(top_five)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

// fn vgg19_custom(vs: &nn::Path, num_classes: i32) -> impl ModuleT {
//     let conv1_1 = nn::conv2d(vs / "conv1_1", 3, 64, 3, Default::default());
//     let conv1_2 = nn::conv2d(vs / "conv1_2", 64, 64, 3, Default::default());
//
//     let conv2_1 = nn::conv2d(vs / "conv2_1", 64, 128, 3, Default::default());
//     let conv2_2 = nn::conv2d(vs / "conv2_2", 128, 128, 3, Default::default());
//
//     let conv3_1 = nn::conv2d(vs / "conv3_1", 128, 256, 3, Default::default());
//     let conv3_2 = nn::conv2d(vs / "conv3_2", 256, 256, 3, Default::default());
//     let conv3_3 = nn::conv2d(vs / "conv3_3", 256, 256, 3, Default::default());
//     let conv3_4 = nn::conv2d(vs / "conv3_4", 256, 256, 3, Default::default());
//
//     let conv4_1 = nn::conv2d(vs / "conv4_1", 256, 512, 3, Default::default());
//     let conv4_2 = nn::conv2d(vs / "conv4_2", 512, 512, 3, Default::default());
//     let conv4_3 = nn::conv2d(vs / "conv4_3", 512, 512, 3, Default::default());
//     let conv4_4 = nn::conv2d(vs / "conv4_4", 512, 512, 3, Default::default());
//
//     let conv5_1 = nn::conv2d(vs / "conv5_1", 512, 512, 3, Default::default());
//     let conv5_2 = nn::conv2d(vs / "conv5_2", 512, 512, 3, Default::default());
//     let conv5_3 = nn::conv2d(vs / "conv5_3", 512, 512, 3, Default::default());
//     let conv5_4 = nn::conv2d(vs / "conv5_4", 512, 512, 3, Default::default());
//
//     let fc1 = nn::linear(vs / "fc1", 512 * 7 * 7, 4096, Default::default());
//     let fc2 = nn::linear(vs / "fc2", 4096, 4096, Default::default());
//     let fc3 = nn::linear(vs / "fc3", 4096, 1000, Default::default());
//
//     nn::func_t(move |xs, train| {
//         let xs = xs
//             .apply(&conv1_1)
//             .relu()
//             .apply(&conv1_2)
//             .relu()
//             .max_pool2d_default(2);
//
//         let xs = xs
//             .apply(&conv2_1)
//             .relu()
//             .apply(&conv2_2)
//             .relu()
//             .max_pool2d_default(2);
//
//         let xs = xs
//             .apply(&conv3_1)
//             .relu()
//             .apply(&conv3_2)
//             .relu()
//             .apply(&conv3_3)
//             .relu()
//             .apply(&conv3_4)
//             .relu()
//             .max_pool2d_default(2);
//
//         let xs = xs
//             .apply(&conv4_1)
//             .relu()
//             .apply(&conv4_2)
//             .relu()
//             .apply(&conv4_3)
//             .relu()
//             .apply(&conv4_4)
//             .relu()
//             .max_pool2d_default(2);
//
//         let xs = xs
//             .apply(&conv5_1)
//             .relu()
//             .apply(&conv5_2)
//             .relu()
//             .apply(&conv5_3)
//             .relu()
//             .apply(&conv5_4)
//             .relu()
//             .max_pool2d_default(2);
//
//         let xs = xs.flatten(1, -1);
//
//         let xs = xs
//             .apply(&fc1)
//             .relu()
//             .dropout(0.5, train)
//             .apply(&fc2)
//             .relu()
//             .dropout(0.5, train)
//             .apply(&fc3);
//
//         xs
//     })
// }
