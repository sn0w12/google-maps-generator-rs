use anyhow::{Context, Result};
use clap::Parser;
use image::{DynamicImage, Rgba, RgbaImage, imageops};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const TILE_SIZE: u32 = 256;
/// Floor for the dynamically-computed GPU batch size.
const MIN_BATCH_SIZE: usize = 64;
/// Ceiling for the dynamically-computed GPU batch size; prevents excessive
/// command-buffer recording overhead at very high tile counts.
const MAX_BATCH_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// WGSL shader: renders a tile-sized quad by sampling a UV sub-region of the
// source texture. Mipmaps on the source texture give trilinear filtering,
// which approximates Lanczos quality without any CPU resize work.
// ---------------------------------------------------------------------------
const TILE_SHADER: &str = r#"
struct Uniforms {
    uv_offset : vec2<f32>,
    uv_scale  : vec2<f32>,
}
@group(0) @binding(0) var<uniform> u   : Uniforms;
@group(0) @binding(1) var          tex : texture_2d<f32>;
@group(0) @binding(2) var          smp : sampler;

struct VOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       uv  : vec2<f32>,
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    // TriangleStrip: 0=BL 1=BR 2=TL 3=TR
    let x = select(-1.0, 1.0, (vi & 1u) != 0u);
    let y = select(-1.0, 1.0, (vi & 2u) != 0u);
    var o : VOut;
    o.pos = vec4(x, y, 0.0, 1.0);
    // map NDC to UV; flip Y because texture (0,0) is top-left
    let base_uv = vec2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    o.uv = u.uv_offset + base_uv * u.uv_scale;
    return o;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // textureSample lets the GPU automatically select the correct mip level
    // based on the UV derivatives, giving high-quality filtered downscaling.
    return textureSample(tex, smp, in.uv);
}
"#;

// Simple blit shader used to generate mip levels on the GPU.
const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;

struct VOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       uv  : vec2<f32>,
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    let x = select(-1.0, 1.0, (vi & 1u) != 0u);
    let y = select(-1.0, 1.0, (vi & 2u) != 0u);
    var o : VOut;
    o.pos = vec4(x, y, 0.0, 1.0);
    o.uv  = vec2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return o;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return textureSample(tex, smp, in.uv);
}
"#;

// ---------------------------------------------------------------------------
// GPU renderer
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniforms {
    uv_offset: [f32; 2],
    uv_scale:  [f32; 2],
}

struct GpuSource {
    _texture: wgpu::Texture,
    view:     wgpu::TextureView,
}

struct GpuTileRenderer {
    device:          wgpu::Device,
    queue:           wgpu::Queue,
    tile_pipeline:   wgpu::RenderPipeline,
    blit_pipeline:   wgpu::RenderPipeline,
    tile_bgl:        wgpu::BindGroupLayout,
    blit_bgl:        wgpu::BindGroupLayout,
    sampler:         wgpu::Sampler,          // trilinear, clamp
    blit_sampler:    wgpu::Sampler,          // bilinear, clamp
    /// One render-target texture per batch slot.
    output_textures: Vec<wgpu::Texture>,
    /// Single staging buffer large enough for all tiles in one batch.
    output_buffer:   wgpu::Buffer,
    /// Single uniform buffer; per-tile data is written at `i * uniform_align` offsets.
    uniform_buffer:  wgpu::Buffer,
    /// Minimum dynamic-offset alignment (device limit, typically 256).
    uniform_align:   u32,
    /// Number of tiles per GPU submission, computed at initialisation from
    /// the adapter's maximum buffer size.
    batch_size:      usize,
}

impl GpuTileRenderer {
    fn try_new() -> Option<Self> {
        pollster::block_on(Self::try_new_async())
    }

    async fn try_new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface:     None,
            })
            .await.ok()?;

        // Compute optimal batch size from the adapter's reported maximum buffer
        // size.  We target at most 25 % of that limit for the staging readback
        // buffer, clamped between MIN_BATCH_SIZE and MAX_BATCH_SIZE.
        let tile_bytes      = (TILE_SIZE * TILE_SIZE * 4) as u64;
        let adapter_max_buf = adapter.limits().max_buffer_size;
        let batch_size      = ((adapter_max_buf / 4) / tile_bytes)
            .clamp(MIN_BATCH_SIZE as u64, MAX_BATCH_SIZE as u64) as usize;
        let staging_buf_size = batch_size as u64 * tile_bytes;

        // Request the device with max_buffer_size large enough to hold the
        // staging buffer (stays within the adapter's own reported limit).
        let required_max_buf = staging_buf_size
            .max(wgpu::Limits::default().max_buffer_size)
            .min(adapter_max_buf);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_buffer_size: required_max_buf,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .ok()?;

        let fmt = wgpu::TextureFormat::Rgba8Unorm;

        // ---- tile bind group layout: uniform + texture + sampler ----
        let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size:   std::num::NonZeroU64::new(
                            std::mem::size_of::<TileUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ---- blit bind group layout: texture + sampler ----
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let tile_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  None,
            source: wgpu::ShaderSource::Wgsl(TILE_SHADER.into()),
        });
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  None,
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let tile_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[&tile_bgl],
            push_constant_ranges: &[],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[&blit_bgl],
            push_constant_ranges: &[],
        });

        let mk_pipeline = |layout: &wgpu::PipelineLayout,
                           module: &wgpu::ShaderModule,
                           fmt:    wgpu::TextureFormat|
         -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:   None,
                layout:  Some(layout),
                vertex:  wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers:             &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets:             &[Some(wgpu::ColorTargetState {
                        format:     fmt,
                        blend:      None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive:     wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
                cache:         None,
            })
        };

        let tile_pipeline = mk_pipeline(&tile_layout, &tile_module, fmt);
        let blit_pipeline = mk_pipeline(&blit_layout, &blit_module, fmt);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Per-device alignment for dynamic uniform buffer offsets (typically 256 bytes).
        let uniform_align = device.limits().min_uniform_buffer_offset_alignment;

        // One render-target texture per batch slot.
        let mk_output_tex = || device.create_texture(&wgpu::TextureDescriptor {
            label:           None,
            size:            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          fmt,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats:    &[],
        });

        // Single staging buffer large enough for all tiles in one batch.
        // 256 * 256 * 4 = 262_144 bytes per tile; offset is already 256-byte aligned.
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               staging_buf_size,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Single uniform buffer with dynamic-offset slots.  Each slot is
        // `uniform_align` bytes wide; the first 16 bytes hold TileUniforms.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               batch_size as u64 * uniform_align as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_textures: Vec<_> = (0..batch_size).map(|_| mk_output_tex()).collect();

        Some(Self {
            device,
            queue,
            tile_pipeline,
            blit_pipeline,
            tile_bgl,
            blit_bgl,
            sampler,
            blit_sampler,
            output_textures,
            output_buffer,
            uniform_buffer,
            uniform_align,
            batch_size,
        })
    }

    /// Upload the source image to the GPU as a mipmapped texture, then
    /// generate all mip levels via GPU blit passes.
    fn upload_source(&self, img: &RgbaImage) -> GpuSource {
        let (w, h) = img.dimensions();
        let mip_count = (w.max(h) as f32).log2().floor() as u32 + 1;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label:           None,
            size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: mip_count,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            img.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(w * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        self.generate_mipmaps(&texture, mip_count);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        GpuSource { _texture: texture, view }
    }

    /// Downsample each mip level from the previous one via a GPU blit pass.
    fn generate_mipmaps(&self, texture: &wgpu::Texture, mip_count: u32) {
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for mip in 1..mip_count {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level:  mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level:  mip,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
                ],
            });

            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                    view:           &dst_view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops:            wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            rp.set_pipeline(&self.blit_pipeline);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..4, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    /// Render a batch of up to `BATCH_SIZE` tiles in a single GPU submission.
    /// Returns pixel data for each tile in order. The caller encodes WebP in
    /// parallel on CPU while the GPU is busy with the next batch.
    fn render_batch(
        &self,
        source: &GpuSource,
        tiles:  &[(u32, u32, u32)], // (x, y, num_tiles)
    ) -> Vec<RgbaImage> {
        let n = tiles.len();
        assert!(n <= self.batch_size);

        let align      = self.uniform_align as usize;
        let tile_bytes = (TILE_SIZE * TILE_SIZE * 4) as usize;

        // Pack all per-tile uniforms into one upload (one write_buffer call).
        let mut uniform_data = vec![0u8; n * align];
        for (i, &(x, y, num_tiles)) in tiles.iter().enumerate() {
            let inv = 1.0 / num_tiles as f32;
            let u   = TileUniforms {
                uv_offset: [x as f32 * inv, y as f32 * inv],
                uv_scale:  [inv, inv],
            };
            let slot = &mut uniform_data[i * align..i * align + std::mem::size_of::<TileUniforms>()];
            slot.copy_from_slice(bytemuck::bytes_of(&u));
        }
        self.queue.write_buffer(&self.uniform_buffer, 0, &uniform_data);

        // One bind group for the whole batch; dynamic offset selects per-tile uniform.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.tile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size:   std::num::NonZeroU64::new(
                            std::mem::size_of::<TileUniforms>() as u64,
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Record all draw + copy commands in a single encoder.
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for i in 0..n {
            let out_view = self.output_textures[i]
                .create_view(&wgpu::TextureViewDescriptor::default());
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:                    None,
                    color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                        view:           &out_view,
                        resolve_target: None,
                        depth_slice:    None,
                        ops:            wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes:         None,
                    occlusion_query_set:      None,
                });
                rp.set_pipeline(&self.tile_pipeline);
                // Dynamic offset selects slot i in the shared uniform buffer.
                rp.set_bind_group(0, &bind_group, &[(i * align) as u32]);
                rp.draw(0..4, 0..1);
            }
            // Copy tile into its region of the single staging buffer.
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture:   &self.output_textures[i],
                    mip_level: 0,
                    origin:    wgpu::Origin3d::ZERO,
                    aspect:    wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.output_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset:         (i * tile_bytes) as u64,
                        bytes_per_row:  Some(TILE_SIZE * 4),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width:                 TILE_SIZE,
                    height:                TILE_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }

        // One submit for the whole batch.
        self.queue.submit([encoder.finish()]);

        // Single map request covering the entire staging buffer.
        let (tx, rx) = std::sync::mpsc::channel();
        self.output_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());

        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();

        // Extract per-tile images from the mapped buffer, then unmap once.
        let images: Vec<RgbaImage> = {
            let mapped = self.output_buffer.slice(..).get_mapped_range();
            (0..n)
                .map(|i| {
                    let start = i * tile_bytes;
                    let end   = start + tile_bytes;
                    RgbaImage::from_raw(TILE_SIZE, TILE_SIZE, mapped[start..end].to_vec()).unwrap()
                })
                .collect()
        };
        self.output_buffer.unmap();

        images
    }
}

// ---------------------------------------------------------------------------
// Output format
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// WebP (lossy)
    Webp,
    /// JPEG (no alpha channel; alpha is stripped)
    Jpg,
    /// AVIF
    Avif,
}

impl OutputFormat {
    fn ext(self) -> &'static str {
        match self {
            OutputFormat::Webp => "webp",
            OutputFormat::Jpg  => "jpg",
            OutputFormat::Avif => "avif",
        }
    }
}

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

/// Generate Google Maps tiles from an image.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image file path (supports PNG, JPEG, BMP, etc.)
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory where tiles will be written
    #[arg(short, long)]
    output: PathBuf,

    /// Minimum zoom level
    #[arg(short = 'm', long, default_value_t = 0)]
    min_zoom: u32,

    /// Maximum zoom level
    #[arg(short = 'M', long)]
    max_zoom: u32,

    /// Background color: "transparent" or a hex code without '#' (e.g., "ff00ff" or "#ff00ff")
    #[arg(short, long, default_value = "transparent")]
    background: String,

    /// Encoding quality (0-100; applies to WebP, JPEG, and AVIF)
    #[arg(short, long, default_value_t = 75)]
    quality: u32,

    /// Output format
    #[arg(short, long, default_value = "webp", value_enum)]
    format: OutputFormat,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse color string into RGBA tuple.
/// Accepts "transparent", empty string, or a 6-digit hex code (with or without '#').
fn parse_color(s: &str) -> (u8, u8, u8, u8) {
    let s = s.trim();
    if s.eq_ignore_ascii_case("transparent") || s.is_empty() {
        return (0, 0, 0, 0);
    }
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&s[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&s[4..6], 16).unwrap_or(0);
        (r, g, b, 255)
    } else {
        eprintln!("Warning: Invalid color format, using transparent");
        (0, 0, 0, 0)
    }
}

/// Make the image square by padding with the background color.
fn make_square(img: &DynamicImage, bg: (u8, u8, u8, u8)) -> RgbaImage {
    let (w, h)  = (img.width(), img.height());
    let size    = w.max(h);
    let bg_rgba = Rgba([bg.0, bg.1, bg.2, bg.3]);
    let canvas  = RgbaImage::from_pixel(size, size, bg_rgba);
    let mut new_img = DynamicImage::ImageRgba8(canvas);
    let x_offset = (size - w) / 2;
    let y_offset = (size - h) / 2;
    imageops::overlay(&mut new_img, img, x_offset as i64, y_offset as i64);
    new_img.into_rgba8()
}

// ---------------------------------------------------------------------------
// Tile encoding
// ---------------------------------------------------------------------------

/// Encode a single RGBA tile to the requested format and return the raw bytes.
pub fn encode_tile(img: RgbaImage, format: OutputFormat, quality: u32) -> Result<Vec<u8>> {
    match format {
        OutputFormat::Webp => {
            let dynamic = DynamicImage::ImageRgba8(img);
            let encoder = webp::Encoder::from_image(&dynamic)
                .map_err(|e| anyhow::anyhow!("WebP encoder error: {}", e))?;
            Ok(encoder.encode(quality as f32).as_ref().to_vec())
        }
        OutputFormat::Jpg => {
            // JPEG has no alpha channel; discard it by converting to RGB.
            let rgb = DynamicImage::ImageRgba8(img).to_rgb8();
            // Pre-allocate a reasonable output buffer; encode directly from raw
            // bytes to avoid re-wrapping in DynamicImage.
            let mut buf = Vec::with_capacity(TILE_SIZE as usize * TILE_SIZE as usize / 2);
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality as u8)
                .encode(rgb.as_raw(), TILE_SIZE, TILE_SIZE, image::ExtendedColorType::Rgb8)?;
            Ok(buf)
        }
        #[cfg(feature = "avif")]
        OutputFormat::Avif => {
            let w = img.width() as usize;
            let h = img.height() as usize;
            let pixels: Vec<rgb::RGBA8> = img
                .pixels()
                .map(|p| rgb::RGBA8 { r: p[0], g: p[1], b: p[2], a: p[3] })
                .collect();
            let result = ravif::Encoder::new()
                .with_quality(quality as f32)
                .with_alpha_quality(quality as f32)
                .encode_rgba(ravif::Img::new(&pixels, w, h))?;
            Ok(result.avif_file)
        }
        #[cfg(not(feature = "avif"))]
        OutputFormat::Avif => {
            anyhow::bail!("AVIF support was not compiled in. \
                Rebuild with `cargo build --features avif` (requires NASM).")
        }
    }
}

// ---------------------------------------------------------------------------
// Per-zoom processing
// ---------------------------------------------------------------------------

/// GPU path: each tile is rendered directly from the source texture — no
/// intermediate scaled image is ever allocated.
fn process_zoom_gpu(
    renderer:   &GpuTileRenderer,
    source:     &GpuSource,
    zoom:       u32,
    output_dir: &Path,
    quality:    u32,
    format:     OutputFormat,
) -> Result<()> {
    let num_tiles:   u32 = 1 << zoom;
    let total_tiles: u64 = (num_tiles as u64) * (num_tiles as u64);

    let zoom_dir = output_dir.join(zoom.to_string());
    fs::create_dir_all(&zoom_dir)?;
    for x in 0..num_tiles {
        fs::create_dir_all(zoom_dir.join(x.to_string()))?;
    }

    let pb = ProgressBar::new(total_tiles);
    pb.set_style(
        ProgressStyle::with_template(
            "Zoom {msg:>2} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} tiles ({eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(zoom.to_string());

    // Collect all (x, y) pairs and process them in batches.
    let all_tiles: Vec<(u32, u32)> = (0..num_tiles)
        .flat_map(|x| (0..num_tiles).map(move |y| (x, y)))
        .collect();

    // Pipeline GPU rendering with CPU encoding via a bounded channel.
    // Capacity 1 means the GPU can render one batch ahead of the encoder,
    // so GPU and CPU work overlaps almost completely when encoding is the
    // bottleneck.  If the GPU is slower, backpressure naturally throttles it.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<RgbaImage>, Vec<(u32, u32)>)>(1);

    let zoom_dir2 = zoom_dir.clone();
    let pb2       = pb.clone();

    // Encoder thread: drains the channel and encodes batches in parallel.
    let enc_thread = std::thread::spawn(move || -> Result<()> {
        while let Ok((images, chunk)) = rx.recv() {
            chunk
                .par_iter()
                .zip(images.into_par_iter())
                .try_for_each(|(&(x, y), img)| -> Result<()> {
                    let encoded = encode_tile(img, format, quality)?;
                    fs::write(
                        zoom_dir2
                            .join(x.to_string())
                            .join(format!("{}.{}", y, format.ext())),
                        &encoded,
                    )?;
                    pb2.inc(1);
                    Ok(())
                })?;
        }
        Ok(())
    });

    // Main thread: renders batches and hands them off; GPU is free to start
    // the next batch as soon as render_batch returns, while the encoder
    // thread is still processing the previous one.
    for chunk in all_tiles.chunks(renderer.batch_size) {
        let specs: Vec<(u32, u32, u32)> =
            chunk.iter().map(|&(x, y)| (x, y, num_tiles)).collect();
        let images = renderer.render_batch(source, &specs);
        tx.send((images, chunk.to_vec()))
            .map_err(|_| anyhow::anyhow!("encoder thread panicked"))?;
    }
    drop(tx); // Signal the encoder thread that no more batches are coming.

    enc_thread
        .join()
        .map_err(|_| anyhow::anyhow!("encoder thread panicked"))??;

    pb.finish_with_message(format!("{} done", zoom));
    Ok(())
}

/// CPU fallback used when no GPU is available.
fn process_zoom_cpu(
    input_img:  &DynamicImage,
    zoom:       u32,
    output_dir: &Path,
    bg:         (u8, u8, u8, u8),
    quality:    u32,
    format:     OutputFormat,
) -> Result<()> {
    let num_tiles:   u32 = 1 << zoom;
    let total_tiles: u64 = (num_tiles as u64) * (num_tiles as u64);

    let zoom_dir = output_dir.join(zoom.to_string());
    fs::create_dir_all(&zoom_dir)?;

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("Zoom {msg:>2} [CPU — scaling image...] {spinner}")
            .unwrap()
            .tick_chars("|/-\\"),
    );
    spinner.set_message(zoom.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));

    let target_size = TILE_SIZE * num_tiles;
    let square      = make_square(input_img, bg);
    let scaled      = DynamicImage::ImageRgba8(square)
        .resize_exact(target_size, target_size, image::imageops::FilterType::Lanczos3);
    let scaled_arc  = Arc::new(scaled);

    spinner.finish_and_clear();

    for x in 0..num_tiles {
        fs::create_dir_all(zoom_dir.join(x.to_string()))?;
    }

    let pb = ProgressBar::new(total_tiles);
    pb.set_style(
        ProgressStyle::with_template(
            "Zoom {msg:>2} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} tiles ({eta}) [CPU]",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(zoom.to_string());

    (0..num_tiles)
        .flat_map(|x| (0..num_tiles).map(move |y| (x, y)))
        .collect::<Vec<_>>()
        .par_iter()
        .try_for_each(|&(x, y)| -> Result<()> {
            let x_dir   = zoom_dir.join(x.to_string());
            let cropped = imageops::crop_imm(
                &*scaled_arc, x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE,
            ).to_image();
            let encoded = encode_tile(cropped, format, quality)?;
            fs::write(x_dir.join(format!("{}.{}", y, format.ext())), &encoded)?;
            pb.inc(1);
            Ok(())
        })?;

    pb.finish_with_message(format!("{} done (CPU)", zoom));
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    if args.min_zoom > args.max_zoom {
        eprintln!("Error: min_zoom must be <= max_zoom");
        std::process::exit(1);
    }

    let bg       = parse_color(&args.background);
    let img      = image::open(&args.input)
        .with_context(|| format!("Failed to open image: {}", args.input.display()))?;
    let rgba_img = make_square(&img, bg);

    match GpuTileRenderer::try_new() {
        Some(renderer) => {
            println!("GPU acceleration active (batch size: {} tiles).", renderer.batch_size);
            let source = renderer.upload_source(&rgba_img);
            for zoom in args.min_zoom..=args.max_zoom {
                process_zoom_gpu(&renderer, &source, zoom, &args.output, args.quality, args.format)?;
            }
        }
        None => {
            eprintln!("Warning: no GPU available, falling back to CPU.");
            let img_dynamic = DynamicImage::ImageRgba8(rgba_img);
            for zoom in args.min_zoom..=args.max_zoom {
                process_zoom_cpu(&img_dynamic, zoom, &args.output, bg, args.quality, args.format)?;
            }
        }
    }

    Ok(())
}