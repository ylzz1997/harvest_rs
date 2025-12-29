
use harvest_rs::harvest_fast_2::{audio_path_to_harvest_f0, Args, HarvestOption};
use std::time::Instant;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TEST CODE
    let args = Args {
        input: "test.mp3".to_string(),
        mixdown: true,
        channel: 0,
        option: HarvestOption {
            f0_floor: 90.0,
            f0_ceil: 1600.0,
            frame_period: 10.0, // ms
        },
    };

    let mut seconds: f64 = 0.0;
    let mut min_seconds: f64 = f64::INFINITY;
    let num_runs = 5;
    let mut final_f0: Vec<f64> = vec![];
    for _ in 0..num_runs {
        let start = Instant::now();
        let (_t, f0) = audio_path_to_harvest_f0(&args)?;
        let dt = start.elapsed().as_secs_f64();
        std::hint::black_box(&f0);
        println!("Harvest 耗时: {:.6}s", dt);
        let voiced = f0.iter().filter(|&&v| v > 0.0).count();
        println!("f0_len={}, voiced_frames={}", f0.len(), voiced);
        seconds += dt;
        if dt < min_seconds {
            min_seconds = dt;
        }
        final_f0 = f0;
    }
    // 写入文件
    let mut file = File::create("f0.txt")?;
    for f in final_f0 {
        file.write_all(format!("{:.6},", f).as_bytes())?;
    }
    println!("平均耗时: {:.6}s", seconds / num_runs as f64);
    println!("最小耗时: {:.6}s", min_seconds);
    Ok(())
}


