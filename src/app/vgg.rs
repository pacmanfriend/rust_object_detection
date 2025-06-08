use anyhow::Result;
use tch::{nn, nn::ModuleT, Device, Tensor};

pub struct VGG19Model {
    model: Box<dyn ModuleT + Send + Sync>,
    device: Device,
}

impl VGG19Model {
    pub fn new(device: Device) -> Result<Self> {
        let vs = nn::VarStore::new(device);

        // Создание архитектуры VGG19
        let model = Box::new(vgg19(&vs.root()));

        // В продакшене здесь нужно загрузить предобученные веса
        // vs.load("vgg19_weights.pt")?;

        // Для демонстрации используем случайную инициализацию
        // В реальном проекте загрузите предобученную модель

        Ok(Self { model, device })
    }

    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.model.forward_t(input, false);
        Ok(output)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

fn vgg19(vs: &nn::Path) -> impl ModuleT {
    let conv1_1 = nn::conv2d(vs / "conv1_1", 3, 64, 3, Default::default());
    let conv1_2 = nn::conv2d(vs / "conv1_2", 64, 64, 3, Default::default());

    let conv2_1 = nn::conv2d(vs / "conv2_1", 64, 128, 3, Default::default());
    let conv2_2 = nn::conv2d(vs / "conv2_2", 128, 128, 3, Default::default());

    let conv3_1 = nn::conv2d(vs / "conv3_1", 128, 256, 3, Default::default());
    let conv3_2 = nn::conv2d(vs / "conv3_2", 256, 256, 3, Default::default());
    let conv3_3 = nn::conv2d(vs / "conv3_3", 256, 256, 3, Default::default());
    let conv3_4 = nn::conv2d(vs / "conv3_4", 256, 256, 3, Default::default());

    let conv4_1 = nn::conv2d(vs / "conv4_1", 256, 512, 3, Default::default());
    let conv4_2 = nn::conv2d(vs / "conv4_2", 512, 512, 3, Default::default());
    let conv4_3 = nn::conv2d(vs / "conv4_3", 512, 512, 3, Default::default());
    let conv4_4 = nn::conv2d(vs / "conv4_4", 512, 512, 3, Default::default());

    let conv5_1 = nn::conv2d(vs / "conv5_1", 512, 512, 3, Default::default());
    let conv5_2 = nn::conv2d(vs / "conv5_2", 512, 512, 3, Default::default());
    let conv5_3 = nn::conv2d(vs / "conv5_3", 512, 512, 3, Default::default());
    let conv5_4 = nn::conv2d(vs / "conv5_4", 512, 512, 3, Default::default());

    let fc1 = nn::linear(vs / "fc1", 512 * 7 * 7, 4096, Default::default());
    let fc2 = nn::linear(vs / "fc2", 4096, 4096, Default::default());
    let fc3 = nn::linear(vs / "fc3", 4096, 1000, Default::default());

    nn::func_t(move |xs, train| {
        let xs = xs
            .apply(&conv1_1)
            .relu()
            .apply(&conv1_2)
            .relu()
            .max_pool2d_default(2);

        let xs = xs
            .apply(&conv2_1)
            .relu()
            .apply(&conv2_2)
            .relu()
            .max_pool2d_default(2);

        let xs = xs
            .apply(&conv3_1)
            .relu()
            .apply(&conv3_2)
            .relu()
            .apply(&conv3_3)
            .relu()
            .apply(&conv3_4)
            .relu()
            .max_pool2d_default(2);

        let xs = xs
            .apply(&conv4_1)
            .relu()
            .apply(&conv4_2)
            .relu()
            .apply(&conv4_3)
            .relu()
            .apply(&conv4_4)
            .relu()
            .max_pool2d_default(2);

        let xs = xs
            .apply(&conv5_1)
            .relu()
            .apply(&conv5_2)
            .relu()
            .apply(&conv5_3)
            .relu()
            .apply(&conv5_4)
            .relu()
            .max_pool2d_default(2);

        let xs = xs.flatten(1, -1);

        let xs = xs
            .apply(&fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&fc2)
            .relu()
            .dropout(0.5, train)
            .apply(&fc3);

        xs
    })
}