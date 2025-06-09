mod app;
mod layers;

use app::start_app;

#[tokio::main]
async fn main() {
    start_app().await.unwrap();
}
