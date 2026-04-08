use image::{Rgba, RgbaImage};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

mod app {
    #![allow(dead_code, unused_imports)]

    include!("../src/main.rs");

    use criterion::{BatchSize, Criterion};
    use std::hint::black_box;

    fn sample_source_image() -> RgbaImage {
        RgbaImage::from_fn(2048, 2048, |x, y| {
            let r = (x & 0xff) as u8;
            let g = (y & 0xff) as u8;
            let b = ((x ^ y) & 0xff) as u8;
            let a = 255;
            Rgba([r, g, b, a])
        })
    }

    pub fn bench_tile_generation(c: &mut Criterion) {
        let source = sample_source_image();
        let zoom = 4;
        let num_tiles: u32 = 1 << zoom;
        let target_size = TILE_SIZE * num_tiles;
        let batch: Vec<(u32, u32, u32)> = (0..4)
            .flat_map(|x| (0..16).map(move |y| (x, y, num_tiles)))
            .collect();

        if let Some(renderer) = GpuTileRenderer::try_new() {
            let uploaded = renderer.upload_source(&source);
            let mut group = c.benchmark_group("tile_generation");
            group.bench_function("gpu_render_batch", |bench| {
                bench.iter(|| {
                    let tiles = renderer.render_batch(&uploaded, black_box(&batch));
                    black_box(tiles);
                });
            });
            group.finish();
            return;
        }

        let mut group = c.benchmark_group("tile_generation");
        group.bench_function("cpu_scale_and_crop", |bench| {
            bench.iter_batched(
                || source.clone(),
                |input| {
                    let scaled = DynamicImage::ImageRgba8(input).resize_exact(
                        target_size,
                        target_size,
                        image::imageops::FilterType::Lanczos3,
                    );
                    let tiles: Vec<RgbaImage> = batch
                        .iter()
                        .map(|&(x, y, _)| {
                            imageops::crop_imm(
                                &scaled,
                                x * TILE_SIZE,
                                y * TILE_SIZE,
                                TILE_SIZE,
                                TILE_SIZE,
                            )
                            .to_image()
                        })
                        .collect();
                    black_box(tiles);
                },
                BatchSize::SmallInput,
            );
        });
        group.finish();
    }
}

fn sample_tile() -> RgbaImage {
    RgbaImage::from_fn(256, 256, |x, y| {
        let r = (x & 0xff) as u8;
        let g = (y & 0xff) as u8;
        let b = ((x ^ y) & 0xff) as u8;
        let a = 255;
        Rgba([r, g, b, a])
    })
}

fn bench_encode_tile(c: &mut Criterion) {
    let input = sample_tile();
    let mut group = c.benchmark_group("tile_encoding");

    for (name, format) in [
        ("webp", app::OutputFormat::Webp),
        ("jpg", app::OutputFormat::Jpg),
    ] {
        group.bench_with_input(
            BenchmarkId::new("encode", name),
            &format,
            |bench, &format| {
                bench.iter_batched(
                    || input.clone(),
                    |tile| {
                        let encoded = app::encode_tile(black_box(tile), format, 75).unwrap();
                        black_box(encoded);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    #[cfg(feature = "avif")]
    {
        group.bench_with_input(
            BenchmarkId::new("encode", "avif"),
            &app::OutputFormat::Avif,
            |bench, &format| {
                bench.iter_batched(
                    || input.clone(),
                    |tile| {
                        let encoded = app::encode_tile(black_box(tile), format, 75).unwrap();
                        black_box(encoded);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, app::bench_tile_generation, bench_encode_tile);
criterion_main!(benches);
