//! Benchmark Qwen2.5-0.5B GGUF with a Rust (Candle) decode loop.

use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;
use serde::Serialize;

const DEFAULT_MODEL: &str =
    r"C:\Users\007sh\.finetuner\bench\models\qwen2.5-0.5b-instruct-q4_k_m.gguf";
const REPEATS: usize = 3;
const PREFILL_LENS: [usize; 2] = [128, 512];
const DECODE_LENS: [usize; 2] = [64, 128];

#[derive(Serialize)]
struct Run {
    name: String,
    tokens: usize,
    mean_ms: f64,
    stdev_ms: f64,
    mean_tps: f64,
    stdev_tps: f64,
    ms_per_token: f64,
}

#[derive(Serialize)]
struct Report {
    runtime: String,
    model: String,
    threads: usize,
    neon: bool,
    avx: bool,
    load_ms: f64,
    tensors: usize,
    runs: Vec<Run>,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn stdev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

fn dummy_tokens(len: usize, device: &Device) -> Result<Tensor> {
    let ids: Vec<u32> = (0..len).map(|i| (i as u32 % 100) + 1).collect();
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}

fn argmax_u32(logits: &Tensor) -> Result<u32> {
    let values = logits.flatten_all()?.to_vec1::<f32>()?;
    let (idx, _) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .context("empty logits")?;
    Ok(idx as u32)
}

fn prefill_ms(model: &mut Qwen2, device: &Device, tokens: usize) -> Result<f64> {
    model.clear_kv_cache();
    let input = dummy_tokens(tokens, device)?;
    let start = Instant::now();
    let _ = model.forward(&input, 0)?;
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn decode_ms(model: &mut Qwen2, device: &Device, tokens: usize) -> Result<f64> {
    model.clear_kv_cache();
    let seed = dummy_tokens(1, device)?;
    let logits = model.forward(&seed, 0)?;
    let mut next = argmax_u32(&logits)?;
    let start = Instant::now();
    for pos in 0..tokens {
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos + 1)?;
        next = argmax_u32(&logits)?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn summarize(name: &str, tokens: usize, samples_ms: &[f64]) -> Run {
    let tps: Vec<f64> = samples_ms
        .iter()
        .map(|ms| tokens as f64 / (ms / 1000.0))
        .collect();
    Run {
        name: name.to_string(),
        tokens,
        mean_ms: mean(samples_ms),
        stdev_ms: stdev(samples_ms),
        mean_tps: mean(&tps),
        stdev_tps: stdev(&tps),
        ms_per_token: mean(samples_ms) / tokens as f64,
    }
}

fn model_path() -> PathBuf {
    env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL))
}

fn load_model(path: &Path, device: &Device) -> Result<(Qwen2, usize, f64)> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let start = Instant::now();
    let content = gguf_file::Content::read(&mut file)?;
    let tensors = content.tensor_infos.len();
    let model = Qwen2::from_gguf(content, &mut file, device)?;
    Ok((model, tensors, start.elapsed().as_secs_f64() * 1000.0))
}

fn main() -> Result<()> {
    let threads = env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    unsafe {
        env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let path = model_path();
    let device = Device::Cpu;
    println!(
        "runtime=candle-0.11 rust decode loop | neon={} avx={} threads={} device=cpu",
        candle_core::utils::with_neon(),
        candle_core::utils::with_avx(),
        threads
    );
    println!("model={}", path.display());

    let (mut model, tensors, load_ms) = load_model(&path, &device)?;
    println!("loaded {tensors} tensors in {load_ms:.0} ms");

    // Warm the kernels so the timed repeats are not dominated by first-use allocs.
    let _ = prefill_ms(&mut model, &device, 8)?;
    let _ = decode_ms(&mut model, &device, 4)?;

    let mut runs = Vec::new();
    for tokens in PREFILL_LENS {
        let _ = prefill_ms(&mut model, &device, tokens)?;
        let mut samples = Vec::new();
        for _ in 0..REPEATS {
            samples.push(prefill_ms(&mut model, &device, tokens)?);
        }
        let run = summarize(&format!("pp{tokens}"), tokens, &samples);
        println!(
            "{:<8} {:>8.2} tok/s  ({:.0} ± {:.0} ms, {:.2} ms/tok)",
            run.name, run.mean_tps, run.mean_ms, run.stdev_ms, run.ms_per_token
        );
        runs.push(run);
    }
    for tokens in DECODE_LENS {
        let _ = decode_ms(&mut model, &device, tokens)?;
        let mut samples = Vec::new();
        for _ in 0..REPEATS {
            samples.push(decode_ms(&mut model, &device, tokens)?);
        }
        let run = summarize(&format!("tg{tokens}"), tokens, &samples);
        println!(
            "{:<8} {:>8.2} tok/s  ({:.0} ± {:.0} ms, {:.2} ms/tok)",
            run.name, run.mean_tps, run.mean_ms, run.stdev_ms, run.ms_per_token
        );
        runs.push(run);
    }

    let report = Report {
        runtime: "candle 0.11 quantized_qwen2 (Rust CPU)".to_string(),
        model: path.display().to_string(),
        threads,
        neon: candle_core::utils::with_neon(),
        avx: candle_core::utils::with_avx(),
        load_ms,
        tensors,
        runs,
    };
    let json = serde_json::to_string_pretty(&report)?;
    let dest = PathBuf::from(r"C:\Users\007sh\.finetuner\bench\qwen05b-rust-metrics.json");
    std::fs::write(&dest, &json)?;
    println!("wrote {}", dest.display());
    Ok(())
}
