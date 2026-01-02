use aus::{mixdown, read};
use num_complex::Complex64;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

const K_PI: f64 = 3.1415926535897932384;
const K_LOG2: f64 = 0.69314718055994529;
const K_MY_SAFE_GUARD_MINIMUM: f64 = 1e-12;

pub use crate::harvest::{Args, HarvestOption};

pub fn audio_path_to_harvest_f0(args: &Args) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let mut audio = read(&args.input).map_err(|e| format!("读取音频失败：{e:?}"))?;
    if args.mixdown {
        mixdown(&mut audio);
        audio.num_channels = 1;
        if !audio.samples.is_empty() {
            audio.num_frames = audio.samples[0].len();
        }
    }
    if audio.samples.is_empty() {
        return Err("音频没有任何 samples（samples 为空）".into());
    }
    if args.channel >= audio.samples.len() {
        return Err(format!(
            "channel 越界：{}，实际声道数是 {}",
            args.channel,
            audio.samples.len()
        )
        .into());
    }
    let samples = &audio.samples[args.channel];
    let x: Vec<f64> = samples.iter().copied().collect();
    let fs = audio.sample_rate as i32;

    Ok(harvest(&x, fs, &args.option))
}

fn my_min_i(x: i32, y: i32) -> i32 {
    if x < y { x } else { y }
}
fn my_max_i(x: i32, y: i32) -> i32 {
    if x > y { x } else { y }
}
fn my_min_f(x: f64, y: f64) -> f64 {
    if x < y { x } else { y }
}

fn get_suitable_fft_size(sample: usize) -> usize {
    let mut n = 1usize;
    while n < sample {
        n <<= 1;
    }
    n
}

fn matlab_round(x: f64) -> i32 {
    if x.is_nan() {
        return 0;
    }
    if x >= 0.0 {
        (x + 0.5).floor() as i32
    } else {
        (x - 0.5).ceil() as i32
    }
}

fn nuttall_window(y_len: usize, y: &mut [f64]) {
    if y_len == 0 {
        return;
    }
    let a0 = 0.355768_f64;
    let a1 = 0.487396_f64;
    let a2 = 0.144232_f64;
    let a3 = 0.012604_f64;
    if y_len == 1 {
        y[0] = 1.0;
        return;
    }
    let denom = (y_len - 1) as f64;
    for n in 0..y_len {
        let x = 2.0 * K_PI * (n as f64) / denom;
        y[n] = a0 - a1 * x.cos() + a2 * (2.0 * x).cos() - a3 * (3.0 * x).cos();
    }
}

// WORLD(matlabfunctions.cpp) 的 decimate: IIR + filtfilt + 边界扩展(kNFact=9) + 特定抽取起点。
fn filter_for_decimate(x: &[f64], r: usize, y: &mut [f64]) {
    debug_assert!(y.len() >= x.len());
    let (a0, a1, a2, b0, b1) = match r {
        11 => (2.450743295230728, -2.06794904601978, 0.59574774438332101, 0.0026822508007163792, 0.0080467524021491377),
        12 => (2.4981398605924205, -2.1368928194784025, 0.62187513816221485, 0.0021097275904709001, 0.0063291827714127002),
        10 => (2.3936475118069387, -1.9873904075111861, 0.5658879979027055, 0.0034818622251927556, 0.010445586675578267),
        9  => (2.3236003491759578, -1.8921545617463598, 0.53148928133729068, 0.0046331164041389372, 0.013899349212416812),
        8  => (2.2357462340187593, -1.7780899984041358, 0.49152555365968692, 0.0063522763407111993, 0.019056829022133598),
        7  => (2.1225239019534703, -1.6395144861046302, 0.44469707800587366, 0.0090366882681608418, 0.027110064804482525),
        6  => (1.9715352749512141, -1.4686795689225347, 0.3893908434965701, 0.013469181309343825, 0.040407543928031475),
        5  => (1.7610939654280557, -1.2554914843859768, 0.3237186507788215, 0.021334858522387423, 0.06400457556716227),
        4  => (1.4499664446880227, -0.98943497080950582, 0.24578252340690215, 0.036710750339322612, 0.11013225101796784),
        3  => (0.95039378983237421, -0.67429146741526791, 0.15412211621346475, 0.071221945171178636, 0.21366583551353591),
        2  => (0.041156734567757189, -0.42599112459189636, 0.041037215479961225, 0.16797464681802227, 0.50392394045406674),
        _  => (0.0, 0.0, 0.0, 0.0, 0.0),
    };

    let mut w0 = 0.0_f64;
    let mut w1 = 0.0_f64;
    let mut w2 = 0.0_f64;
    for (i, &v) in x.iter().enumerate() {
        let wt = v + a0 * w0 + a1 * w1 + a2 * w2;
        y[i] = b0 * wt + b1 * w0 + b1 * w1 + b0 * w2;
        w2 = w1;
        w1 = w0;
        w0 = wt;
    }
}

fn decimate_world(x: &[f64], r: usize, y: &mut [f64]) {
    if r <= 1 {
        let n = x.len().min(y.len());
        y[..n].copy_from_slice(&x[..n]);
        return;
    }

    const NFACT: usize = 9;
    if x.is_empty() || y.is_empty() {
        return;
    }
    let x_length = x.len();
    let ext_len = x_length + NFACT * 2;
    let mut tmp1 = vec![0.0_f64; ext_len];
    let mut tmp2 = vec![0.0_f64; ext_len];

    // 左侧扩展：tmp1[i] = 2*x[0] - x[NFACT - i]
    for i in 0..NFACT {
        let idx = (NFACT - i).min(x_length - 1);
        tmp1[i] = 2.0 * x[0] - x[idx];
    }
    // 原信号
    tmp1[NFACT..NFACT + x_length].copy_from_slice(x);
    // 右侧扩展：tmp1[...] = 2*x[last] - x[x_length-2 - j]
    for j in 0..NFACT {
        let src = if x_length >= 2 {
            (x_length - 2).saturating_sub(j)
        } else {
            0
        };
        tmp1[NFACT + x_length + j] = 2.0 * x[x_length - 1] - x[src];
    }

    filter_for_decimate(&tmp1, r, &mut tmp2);
    for i in 0..ext_len {
        tmp1[i] = tmp2[ext_len - i - 1];
    }
    filter_for_decimate(&tmp1, r, &mut tmp2);
    for i in 0..ext_len {
        tmp1[i] = tmp2[ext_len - i - 1];
    }

    // nout = (x_length - 1)/r + 1
    let nout = (x_length.saturating_sub(1)) / r + 1;
    // nbeg = r - r*nout + x_length
    let nbeg = r as isize - (r * nout) as isize + x_length as isize;

    let mut count = 0usize;
    let mut i = nbeg;
    while count < nout && i < (x_length + NFACT) as isize {
        if i >= 0 {
            // WORLD: y[count++] = tmp1[i + kNFact - 1];
            let idx = (i as usize) + NFACT - 1;
            // idx 在 [0, ext_len) 内
            if idx < tmp1.len() && count < y.len() {
                y[count] = tmp1[idx];
                count += 1;
            }
        }
        i += r as isize;
    }
}

// linear 
fn interp1(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
    if x.len() < 2 || x.len() != y.len() {
        yi.fill(0.0);
        return;
    }
    // WORLD(matlabfunctions.cpp) 的 interp1 行为：
    // - 线性插值
    // - xi 超出 x 范围时使用端点段进行线性外推（而不是置 0）
    let n = x.len();
    let mut x_idx = 0usize; // 指向满足 x[x_idx] <= q 的最大 idx

    for (k, &q) in xi.iter().enumerate().take(yi.len()) {
        if !q.is_finite() {
            yi[k] = 0.0;
            continue;
        }

        // 线性推进（xi 与 x 单调递增时非常快）
        while x_idx + 1 < n && x[x_idx + 1] <= q {
            x_idx += 1;
        }

        // 选择区间 [k0, k1]，其中 k1 = k0+1，且 clamp 到 [0..n-1]
        let k1 = if q < x[0] { 1 } else if x_idx >= n - 1 { n - 1 } else { x_idx + 1 };
        let k0 = k1 - 1;

        let x0 = x[k0];
        let x1 = x[k1];
        let y0 = y[k0];
        let y1 = y[k1];

        if (x1 - x0).abs() < 1e-12 {
            yi[k] = y0;
        } else {
            let s = (q - x0) / (x1 - x0);
            yi[k] = y0 + s * (y1 - y0);
        }
    }
}

#[derive(Clone)]
struct FftCtx {
    n: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    // 复用临时缓冲，避免每次 FFT 申请内存
    tmp_time: RefCell<Vec<f64>>,
    scratch_fwd: RefCell<Vec<Complex64>>,
    scratch_inv: RefCell<Vec<Complex64>>,
}

impl FftCtx {
    fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(n);
        let c2r = planner.plan_fft_inverse(n);

        // realfft 的 scratch 使用 Complex 缓冲
        let scratch_fwd = vec![Complex64::new(0.0, 0.0); r2c.get_scratch_len()];
        let scratch_inv = vec![Complex64::new(0.0, 0.0); c2r.get_scratch_len()];

        Self {
            n,
            r2c,
            c2r,
            tmp_time: RefCell::new(vec![0.0; n]),
            scratch_fwd: RefCell::new(scratch_fwd),
            scratch_inv: RefCell::new(scratch_inv),
        }
    }

    // 实数 -> 复数半谱，out.len() 必须为 n/2+1
    fn fft_real(&self, x: &[f64], out: &mut [Complex64]) {
        debug_assert_eq!(out.len(), self.n / 2 + 1);
        let mut tmp = self.tmp_time.borrow_mut();
        debug_assert_eq!(tmp.len(), self.n);

        // 拷贝 + 0 填充
        let copy_len = x.len().min(self.n);
        tmp[..copy_len].copy_from_slice(&x[..copy_len]);
        for v in &mut tmp[copy_len..] {
            *v = 0.0;
        }

        let mut scratch = self.scratch_fwd.borrow_mut();
        self.r2c
            .process_with_scratch(&mut tmp, out, &mut scratch)
            .expect("realfft forward failed");
    }

    // 复数半谱 -> 实数，x.len() 必须为 n/2+1；注意会原地修改 x
    fn ifft_to_real_in_place(&self, x: &mut [Complex64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n / 2 + 1);
        let mut tmp = self.tmp_time.borrow_mut();
        debug_assert_eq!(tmp.len(), self.n);

        let mut scratch = self.scratch_inv.borrow_mut();
        self.c2r
            .process_with_scratch(x, &mut tmp, &mut scratch)
            .expect("realfft inverse failed");

        // 与 rustfft 一样，inverse 不做归一化
        let scale = 1.0 / (self.n as f64);
        let n = out.len().min(self.n);
        for i in 0..n {
            out[i] = tmp[i] * scale;
        }
    }
}

thread_local! {
    // 每个 rayon worker 线程各自缓存 FFT plan，避免锁竞争，同时大幅减少 plan 生成开销
    static FFT_CACHE: RefCell<HashMap<usize, FftCtx>> = RefCell::new(HashMap::new());
}

#[derive(Default)]
struct FilterScratch {
    band_pass_filter: Vec<f64>,
    h_spec: Vec<Complex64>,
    prod: Vec<Complex64>,
    time: Vec<f64>,
}

#[derive(Default)]
struct ZeroCrossScratch {
    edges: Vec<usize>,
    fine_edges: Vec<f64>,
}

thread_local! {
    // 每个 rayon worker 线程各自复用临时缓冲，减少频繁 vec! 分配
    static FILTER_SCRATCH: RefCell<HashMap<usize, FilterScratch>> = RefCell::new(HashMap::new());
    static ZC_SCRATCH: RefCell<ZeroCrossScratch> = RefCell::new(ZeroCrossScratch::default());
}

#[inline]
fn with_filter_scratch<R>(fft_size: usize, f: impl FnOnce(&mut FilterScratch) -> R) -> R {
    FILTER_SCRATCH.with(|cache| {
        let mut map = cache.borrow_mut();
        let scratch = map.entry(fft_size).or_insert_with(FilterScratch::default);
        f(scratch)
    })
}

#[inline]
fn with_zc_scratch<R>(f: impl FnOnce(&mut ZeroCrossScratch) -> R) -> R {
    ZC_SCRATCH.with(|s| f(&mut s.borrow_mut()))
}

#[inline]
fn with_fft_ctx<R>(n: usize, f: impl FnOnce(&FftCtx) -> R) -> R {
    FFT_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        let ctx = map.entry(n).or_insert_with(|| FftCtx::new(n));
        f(ctx)
    })
}

fn get_waveform_and_spectrum_sub(
    x: &[f64],
    x_length: usize,
    y_length: usize,
    decimation_ratio: usize,
    y: &mut [f64],
) {
    if decimation_ratio == 1 {
        let n = x_length.min(y.len());
        y[..n].copy_from_slice(&x[..n]);
        return;
    }

    let lag = (((140.0 / decimation_ratio as f64).ceil() as usize) * decimation_ratio) as usize;
    let new_x_length = x_length + lag * 2;
    let mut new_y = vec![0.0_f64; new_x_length];
    let mut new_x = vec![0.0_f64; new_x_length];
    for i in 0..lag {
        new_x[i] = x[0];
    }
    for i in lag..(lag + x_length) {
        new_x[i] = x[i - lag];
    }
    for i in (lag + x_length)..new_x_length {
        new_x[i] = x[x_length - 1];
    }

    decimate_world(&new_x, decimation_ratio, &mut new_y);
    let start = lag / decimation_ratio;
    for i in 0..y_length {
        y[i] = new_y[start + i];
    }
}

fn get_waveform_and_spectrum(
    x: &[f64],
    x_length: usize,
    y_length: usize,
    fft_size: usize,
    decimation_ratio: usize,
    y: &mut [f64],
    y_spectrum: &mut [Complex64],
) {
    debug_assert_eq!(y_spectrum.len(), fft_size / 2 + 1);
    y.fill(0.0);
    get_waveform_and_spectrum_sub(x, x_length, y_length, decimation_ratio, y);

    let mut mean = 0.0_f64;
    for i in 0..y_length {
        mean += y[i];
    }
    mean /= y_length as f64;
    for i in 0..y_length {
        y[i] -= mean;
    }
    for i in y_length..fft_size {
        y[i] = 0.0;
    }

    with_fft_ctx(fft_size, |fft| {
        fft.fft_real(y, y_spectrum);
    });
}

fn get_filtered_signal(
    boundary_f0: f64,
    fft_size: usize,
    fs: f64,
    y_spectrum: &[Complex64],
    y_length: usize,
    filtered_signal: &mut [f64],
) {
    debug_assert_eq!(y_spectrum.len(), fft_size / 2 + 1);
    let filter_length_half = matlab_round(fs / boundary_f0 * 2.0).max(0) as usize;
    let win_len = filter_length_half * 2 + 1;
    let index_bias = filter_length_half + 1;

    // 纯复用临时缓冲，不改变任何数值路径（结果应与当前版本完全一致）
    with_filter_scratch(fft_size, |scratch| {
        scratch.band_pass_filter.resize(fft_size, 0.0);
        scratch.band_pass_filter.fill(0.0);
        nuttall_window(win_len, &mut scratch.band_pass_filter[..win_len]);
        for i in 0..win_len {
            let k = i as isize - filter_length_half as isize;
            scratch.band_pass_filter[i] *= (2.0 * K_PI * boundary_f0 * (k as f64) / fs).cos();
        }

        with_fft_ctx(fft_size, |fft| {
            let half = fft_size / 2 + 1;
            debug_assert_eq!(y_spectrum.len(), half);

            scratch.h_spec.resize(half, Complex64::new(0.0, 0.0));
            fft.fft_real(&scratch.band_pass_filter, &mut scratch.h_spec);

            scratch.prod.resize(half, Complex64::new(0.0, 0.0));
            for i in 0..half {
                scratch.prod[i] = y_spectrum[i] * scratch.h_spec[i];
            }

            scratch.time.resize(fft_size, 0.0);
            fft.ifft_to_real_in_place(&mut scratch.prod, &mut scratch.time);
        });

        for i in 0..y_length {
            filtered_signal[i] = scratch.time[i + index_bias];
        }
    });
    // filtered_signal 已在 with_fft_ctx 内写入
}

fn check_event(x: i32) -> i32 {
    if x > 0 { 1 } else { 0 }
}

fn zero_crossing_engine(
    filtered_signal: &[f64],
    y_length: usize,
    fs: f64,
    interval_locations: &mut Vec<f64>,
    intervals: &mut Vec<f64>,
) -> usize {
    // 纯复用临时缓冲，不改变任何数值路径（结果应与当前版本完全一致）
    with_zc_scratch(|scratch| {
        scratch.edges.clear();
        for i in 0..(y_length.saturating_sub(1)) {
            if 0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0 {
                scratch.edges.push(i + 1);
            }
        }
        if scratch.edges.len() < 2 {
            return 0;
        }

        scratch.fine_edges.resize(scratch.edges.len(), 0.0);
        for (i, &e) in scratch.edges.iter().enumerate() {
            let num = filtered_signal[e - 1];
            let den = filtered_signal[e] - filtered_signal[e - 1];
            scratch.fine_edges[i] = e as f64 - num / (den + K_MY_SAFE_GUARD_MINIMUM);
        }

        let out_len = scratch.fine_edges.len() - 1;
        interval_locations.clear();
        intervals.clear();
        interval_locations.reserve(out_len);
        intervals.reserve(out_len);
        for i in 0..out_len {
            let f = fs / (scratch.fine_edges[i + 1] - scratch.fine_edges[i] + K_MY_SAFE_GUARD_MINIMUM);
            let loc = (scratch.fine_edges[i] + scratch.fine_edges[i + 1]) / 2.0 / fs;
            intervals.push(f);
            interval_locations.push(loc);
        }
        out_len
    })
}

#[derive(Default)]
struct ZeroCrossings {
    negative_interval_locations: Vec<f64>,
    negative_intervals: Vec<f64>,
    number_of_negatives: usize,
    positive_interval_locations: Vec<f64>,
    positive_intervals: Vec<f64>,
    number_of_positives: usize,
    peak_interval_locations: Vec<f64>,
    peak_intervals: Vec<f64>,
    number_of_peaks: usize,
    dip_interval_locations: Vec<f64>,
    dip_intervals: Vec<f64>,
    number_of_dips: usize,
}

fn get_four_zero_crossing_intervals(filtered_signal: &mut [f64], y_length: usize, fs: f64) -> ZeroCrossings {
    let mut z = ZeroCrossings::default();

    z.number_of_negatives = zero_crossing_engine(
        filtered_signal,
        y_length,
        fs,
        &mut z.negative_interval_locations,
        &mut z.negative_intervals,
    );

    for v in filtered_signal.iter_mut().take(y_length) {
        *v = -*v;
    }
    z.number_of_positives = zero_crossing_engine(
        filtered_signal,
        y_length,
        fs,
        &mut z.positive_interval_locations,
        &mut z.positive_intervals,
    );

    for i in 0..(y_length.saturating_sub(1)) {
        filtered_signal[i] = filtered_signal[i] - filtered_signal[i + 1];
    }
    z.number_of_peaks = zero_crossing_engine(
        filtered_signal,
        y_length.saturating_sub(1),
        fs,
        &mut z.peak_interval_locations,
        &mut z.peak_intervals,
    );

    for i in 0..(y_length.saturating_sub(1)) {
        filtered_signal[i] = -filtered_signal[i];
    }
    z.number_of_dips = zero_crossing_engine(
        filtered_signal,
        y_length.saturating_sub(1),
        fs,
        &mut z.dip_interval_locations,
        &mut z.dip_intervals,
    );

    z
}

fn get_f0_candidate_contour_sub(
    interpolated: [&[f64]; 4],
    f0_floor: f64,
    f0_ceil: f64,
    boundary_f0: f64,
    out: &mut [f64],
) {
    let upper = boundary_f0 * 1.1;
    let lower = boundary_f0 * 0.9;
    for i in 0..out.len() {
        let v = (interpolated[0][i] + interpolated[1][i] + interpolated[2][i] + interpolated[3][i]) / 4.0;
        let ok = v <= upper && v >= lower && v <= f0_ceil && v >= f0_floor;
        out[i] = if ok { v } else { 0.0 };
    }
}

fn get_f0_candidate_contour(
    z: &ZeroCrossings,
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_candidate: &mut [f64],
) {
    if 0
        == check_event(z.number_of_negatives as i32 - 2)
            * check_event(z.number_of_positives as i32 - 2)
            * check_event(z.number_of_peaks as i32 - 2)
            * check_event(z.number_of_dips as i32 - 2)
    {
        f0_candidate.fill(0.0);
        return;
    }

    let mut i0 = vec![0.0_f64; f0_candidate.len()];
    let mut i1 = vec![0.0_f64; f0_candidate.len()];
    let mut i2 = vec![0.0_f64; f0_candidate.len()];
    let mut i3 = vec![0.0_f64; f0_candidate.len()];

    interp1(
        &z.negative_interval_locations,
        &z.negative_intervals,
        temporal_positions,
        &mut i0,
    );
    interp1(
        &z.positive_interval_locations,
        &z.positive_intervals,
        temporal_positions,
        &mut i1,
    );
    interp1(&z.peak_interval_locations, &z.peak_intervals, temporal_positions, &mut i2);
    interp1(&z.dip_interval_locations, &z.dip_intervals, temporal_positions, &mut i3);

    get_f0_candidate_contour_sub([&i0, &i1, &i2, &i3], f0_floor, f0_ceil, boundary_f0, f0_candidate);
}

fn get_f0_candidate_from_raw_event(
    boundary_f0: f64,
    fs: f64,
    y_spectrum: &[Complex64],
    y_length: usize,
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_candidate: &mut [f64],
) {
    let mut filtered_signal = vec![0.0_f64; fft_size];
    get_filtered_signal(boundary_f0, fft_size, fs, y_spectrum, y_length, &mut filtered_signal);
    let z = get_four_zero_crossing_intervals(&mut filtered_signal, y_length, fs);
    get_f0_candidate_contour(&z, boundary_f0, f0_floor, f0_ceil, temporal_positions, f0_candidate);
}

// ---------------------- Parallelized Section 1 ----------------------
// 使用 rayon 并行计算每个频段的候选值
fn get_raw_f0_candidates_par(
    boundary_f0_list: &[f64],
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    y_spectrum: &[Complex64],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    raw_f0_candidates: &mut [Vec<f64>],
) {
    // par_iter + zip_eq 进行并行
    raw_f0_candidates
        .par_iter_mut()
        .zip(boundary_f0_list.par_iter())
        .for_each(|(candidates, &bf0)| {
            get_f0_candidate_from_raw_event(
                bf0,
                actual_fs,
                y_spectrum,
                y_length,
                fft_size,
                f0_floor,
                f0_ceil,
                temporal_positions,
                candidates,
            );
        });
}

fn detect_official_f0_candidates_sub1(vuv: &[i32], st: &mut Vec<usize>, ed: &mut Vec<usize>) -> usize {
    st.clear();
    ed.clear();
    let mut number = 0usize;
    for i in 1..vuv.len() {
        let tmp = vuv[i] - vuv[i - 1];
        if tmp == 1 {
            st.push(i);
        }
        if tmp == -1 {
            ed.push(i);
            number += 1;
        }
    }
    number
}

fn detect_official_f0_candidates_sub2(
    raw_f0_candidates: &[Vec<f64>],
    index: usize,
    st: &[usize],
    ed: &[usize],
    max_candidates: usize,
    out: &mut [f64],
) -> usize {
    let mut number = 0usize;
    for i in 0..st.len().min(ed.len()) {
        if ed[i] <= st[i] || ed[i] - st[i] < 10 {
            continue;
        }
        let mut tmp_f0 = 0.0_f64;
        for ch in st[i]..ed[i] {
            tmp_f0 += raw_f0_candidates[ch][index];
        }
        tmp_f0 /= (ed[i] - st[i]) as f64;
        if number < max_candidates {
            out[number] = tmp_f0;
        }
        number += 1;
        if number >= max_candidates {
            break;
        }
    }
    for i in number..max_candidates {
        out[i] = 0.0;
    }
    number
}

fn detect_official_f0_candidates(
    raw_f0_candidates: &[Vec<f64>],
    number_of_channels: usize,
    f0_length: usize,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) -> usize {
    let mut number_of_candidates = 0usize;
    // 这里的循环依赖性较强且开销不大，维持串行或简单优化
    // 实际上这里也可以按 i (temporal pos) 并行，但需要处理 number_of_candidates 的归约
    // 简单起见，这里保持串行，因为耗时主要在 FFT
    let mut vuv = vec![0i32; number_of_channels];
    let mut st: Vec<usize> = Vec::new();
    let mut ed: Vec<usize> = Vec::new();
    for i in 0..f0_length {
        for j in 0..number_of_channels {
            vuv[j] = if raw_f0_candidates[j][i] > 0.0 { 1 } else { 0 };
        }
        if !vuv.is_empty() {
            vuv[0] = 0;
            vuv[number_of_channels - 1] = 0;
        }
        let voiced_sections = detect_official_f0_candidates_sub1(&vuv, &mut st, &mut ed);
        let n = detect_official_f0_candidates_sub2(raw_f0_candidates, i, &st[..voiced_sections], &ed[..voiced_sections], max_candidates, &mut f0_candidates[i]);
        number_of_candidates = number_of_candidates.max(n);
    }
    number_of_candidates
}

fn overlap_f0_candidates(f0_length: usize, number_of_candidates: usize, f0_candidates: &mut [Vec<f64>]) {
    let n = 3usize;
    for i in 1..=n {
        for j in 0..number_of_candidates {
            for k in i..f0_length {
                let dst = j + number_of_candidates * i;
                if dst < f0_candidates[k].len() {
                    f0_candidates[k][dst] = f0_candidates[k - i][j];
                }
            }
            for k in 0..(f0_length.saturating_sub(i)) {
                let dst = j + number_of_candidates * (i + n);
                if dst < f0_candidates[k].len() {
                    f0_candidates[k][dst] = f0_candidates[k + i][j];
                }
            }
        }
    }
}

fn get_base_index(current_position: f64, base_time: &[f64], fs: f64, base_index: &mut [i32]) {
    let basic_index = matlab_round((current_position + base_time[0]) * fs + 0.001);
    for (i, v) in base_index.iter_mut().enumerate() {
        *v = basic_index + i as i32;
    }
}

fn get_main_window(current_position: f64, base_index: &[i32], fs: f64, window_length_in_time: f64, main_window: &mut [f64]) {
    for i in 0..main_window.len() {
        let tmp = (base_index[i] as f64 - 1.0) / fs - current_position;
        main_window[i] = 0.42
            + 0.5 * (2.0 * K_PI * tmp / window_length_in_time).cos()
            + 0.08 * (4.0 * K_PI * tmp / window_length_in_time).cos();
    }
}

fn get_diff_window(main_window: &[f64], diff_window: &mut [f64]) {
    if main_window.len() != diff_window.len() || main_window.len() < 2 {
        diff_window.fill(0.0);
        return;
    }
    let n = main_window.len();
    diff_window[0] = -main_window[1] / 2.0;
    for i in 1..(n - 1) {
        diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    diff_window[n - 1] = main_window[n - 2] / 2.0;
}

fn get_spectra(
    x: &[f64],
    x_length: usize,
    fft_size: usize,
    base_index: &[i32],
    main_window: &[f64],
    diff_window: &[f64],
    main_spectrum: &mut [Complex64],
    diff_spectrum: &mut [Complex64],
    fft: &FftCtx,
) {
    let base_time_length = base_index.len();
    let mut buf = vec![0.0_f64; fft_size];

    for i in 0..base_time_length {
        let idx = my_max_i(0, my_min_i(x_length as i32 - 1, base_index[i] - 1)) as usize;
        buf[i] = x[idx] * main_window[i];
    }
    for i in base_time_length..fft_size {
        buf[i] = 0.0;
    }
    fft.fft_real(&buf, main_spectrum);

    for i in 0..base_time_length {
        let idx = my_max_i(0, my_min_i(x_length as i32 - 1, base_index[i] - 1)) as usize;
        buf[i] = x[idx] * diff_window[i];
    }
    for i in base_time_length..fft_size {
        buf[i] = 0.0;
    }
    fft.fft_real(&buf, diff_spectrum);
}

fn fix_f0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: f64,
    current_f0: f64,
    number_of_harmonics: usize,
) -> (f64, f64) {
    let mut amplitude_list = vec![0.0_f64; number_of_harmonics];
    let mut inst_freq_list = vec![0.0_f64; number_of_harmonics];

    for i in 0..number_of_harmonics {
        let idx = matlab_round(current_f0 * fft_size as f64 / fs * (i as f64 + 1.0)).max(0) as usize;
        let idx = idx.min(power_spectrum.len().saturating_sub(1));
        inst_freq_list[i] = if power_spectrum[idx] == 0.0 {
            0.0
        } else {
            idx as f64 * fs / fft_size as f64 + numerator_i[idx] / power_spectrum[idx] * fs / 2.0 / K_PI
        };
        amplitude_list[i] = power_spectrum[idx].sqrt();
    }

    let mut denom = 0.0_f64;
    let mut numer = 0.0_f64;
    let mut score = 0.0_f64;
    for i in 0..number_of_harmonics {
        numer += amplitude_list[i] * inst_freq_list[i];
        denom += amplitude_list[i] * (i as f64 + 1.0);
        score += ((inst_freq_list[i] / (i as f64 + 1.0) - current_f0) / current_f0).abs();
    }

    let refined_f0 = numer / (denom + K_MY_SAFE_GUARD_MINIMUM);
    let refined_score = 1.0 / (score / number_of_harmonics as f64 + K_MY_SAFE_GUARD_MINIMUM);
    (refined_f0, refined_score)
}

fn get_mean_f0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    current_f0: f64,
    fft_size: usize,
    window_length_in_time: f64,
    base_time: &[f64],
) -> (f64, f64) {
    with_fft_ctx(fft_size, |fft| {
        let base_time_length = base_time.len();
        let mut base_index = vec![0i32; base_time_length];
        let mut main_window = vec![0.0_f64; base_time_length];
        let mut diff_window = vec![0.0_f64; base_time_length];

        get_base_index(current_position, base_time, fs, &mut base_index);
        get_main_window(current_position, &base_index, fs, window_length_in_time, &mut main_window);
        get_diff_window(&main_window, &mut diff_window);

        let mut main_spectrum = vec![Complex64::new(0.0, 0.0); fft_size / 2 + 1];
        let mut diff_spectrum = vec![Complex64::new(0.0, 0.0); fft_size / 2 + 1];
        get_spectra(
            x,
            x_length,
            fft_size,
            &base_index,
            &main_window,
            &diff_window,
            &mut main_spectrum,
            &mut diff_spectrum,
            fft,
        );

        let mut power_spectrum = vec![0.0_f64; fft_size / 2 + 1];
        let mut numerator_i = vec![0.0_f64; fft_size / 2 + 1];
        for j in 0..=fft_size / 2 {
            numerator_i[j] = main_spectrum[j].re * diff_spectrum[j].im - main_spectrum[j].im * diff_spectrum[j].re;
            power_spectrum[j] = main_spectrum[j].re * main_spectrum[j].re + main_spectrum[j].im * main_spectrum[j].im;
        }

        let number_of_harmonics = my_min_i((fs / 2.0 / current_f0) as i32, 6).max(1) as usize;
        fix_f0(&power_spectrum, &numerator_i, fft_size, fs, current_f0, number_of_harmonics)
    })
}

fn get_refined_f0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    current_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> (f64, f64) {
    if current_f0 <= 0.0 {
        return (0.0, 0.0);
    }

    let half_window_length = (1.5 * fs / current_f0 + 1.0) as usize;
    let window_length_in_time = (2 * half_window_length + 1) as f64 / fs;
    let mut base_time = vec![0.0_f64; half_window_length * 2 + 1];
    for i in 0..base_time.len() {
        base_time[i] = (-(half_window_length as f64) + i as f64) / fs;
    }
    let len = (half_window_length * 2 + 1) as f64;
    let pow = 2u32 + ((len.ln() / K_LOG2) as u32);
    let fft_size = 1usize << pow;

    let (mut refined_f0, mut refined_score) = get_mean_f0(
        x,
        x_length,
        fs,
        current_position,
        current_f0,
        fft_size,
        window_length_in_time,
        &base_time,
    );

    if refined_f0 < f0_floor || refined_f0 > f0_ceil || refined_score < 2.5 {
        refined_f0 = 0.0;
        refined_score = 0.0;
    }
    (refined_f0, refined_score)
}

// ---------------------- Parallelized Section 2 ----------------------
// 最耗时的部分：对所有帧并行处理
fn refine_f0_candidates_par(
    x: &[f64],
    x_length: usize,
    fs: f64,
    temporal_positions: &[f64],
    f0_floor: f64,
    f0_ceil: f64,
    refined_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
    number_of_candidates: usize,
) {
    // par_iter_mut 替代串行
    refined_candidates
        .par_iter_mut()
        .zip(f0_scores.par_iter_mut())
        .zip(temporal_positions.par_iter())
        .for_each(|((candidates, scores), &tp)| {
            for j in 0..number_of_candidates {
                let (rf0, score) = get_refined_f0(
                    x,
                    x_length,
                    fs,
                    tp,
                    candidates[j],
                    f0_floor,
                    f0_ceil,
                );
                candidates[j] = rf0;
                scores[j] = score;
            }
        });
}

fn select_best_f0(reference_f0: f64, f0_candidates: &[f64], allowed_range: f64) -> (f64, f64) {
    let mut best_f0 = 0.0_f64;
    let mut best_error = allowed_range;
    for &c in f0_candidates {
        if c <= 0.0 {
            continue;
        }
        let tmp = (reference_f0 - c).abs() / (reference_f0 + K_MY_SAFE_GUARD_MINIMUM);
        if tmp > best_error {
            continue;
        }
        best_f0 = c;
        best_error = tmp;
    }
    (best_f0, best_error)
}

fn remove_unreliable_candidates(
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
) {
    let tmp: Vec<Vec<f64>> = f0_candidates.iter().map(|v| v[..number_of_candidates].to_vec()).collect();
    for i in 1..(f0_length.saturating_sub(1)) {
        for j in 0..number_of_candidates {
            let reference = f0_candidates[i][j];
            if reference == 0.0 {
                continue;
            }
            let (_b1, e1) = select_best_f0(reference, &tmp[i + 1], 1.0);
            let (_b2, e2) = select_best_f0(reference, &tmp[i - 1], 1.0);
            let min_error = my_min_f(e1, e2);
            if min_error <= 0.05 {
                continue;
            }
            f0_candidates[i][j] = 0.0;
            f0_scores[i][j] = 0.0;
        }
    }
}

fn search_f0_base(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
    base: &mut [f64],
) {
    for i in 0..f0_length {
        let mut best = 0.0_f64;
        let mut best_score = 0.0_f64;
        for j in 0..number_of_candidates {
            if f0_scores[i][j] > best_score {
                best = f0_candidates[i][j];
                best_score = f0_scores[i][j];
            }
        }
        base[i] = best;
    }
}

fn fix_step1(f0_base: &[f64], f0_length: usize, allowed_range: f64, out: &mut [f64]) {
    out.fill(0.0);
    for i in 2..f0_length {
        if f0_base[i] == 0.0 {
            continue;
        }
        let reference = f0_base[i - 1] * 2.0 - f0_base[i - 2];
        let cond1 = ((f0_base[i] - reference) / (reference + K_MY_SAFE_GUARD_MINIMUM)).abs() > allowed_range;
        let cond2 = ((f0_base[i] - f0_base[i - 1]) / (f0_base[i - 1] + K_MY_SAFE_GUARD_MINIMUM)).abs() > allowed_range;
        out[i] = if cond1 && cond2 { 0.0 } else { f0_base[i] };
    }
}

fn get_boundary_list(f0: &[f64], boundary_list: &mut Vec<usize>) -> usize {
    boundary_list.clear();
    let n = f0.len();
    if n < 2 {
        return 0;
    }
    let mut vuv = vec![0i32; n];
    for i in 0..n {
        vuv[i] = if f0[i] > 0.0 { 1 } else { 0 };
    }
    vuv[0] = 0;
    vuv[n - 1] = 0;
    for i in 1..n {
        if vuv[i] - vuv[i - 1] != 0 {
            let offset = boundary_list.len() % 2;
            boundary_list.push(i - offset);
        }
    }
    boundary_list.len()
}

fn fix_step2(f0_step1: &[f64], f0_length: usize, voice_range_minimum: usize, out: &mut [f64]) {
    out.copy_from_slice(f0_step1);
    let mut boundary_list = Vec::with_capacity(f0_length);
    let number = get_boundary_list(f0_step1, &mut boundary_list);
    for i in 0..(number / 2) {
        let st = boundary_list[i * 2];
        let ed = boundary_list[i * 2 + 1];
        if ed >= st && ed - st >= voice_range_minimum {
            continue;
        }
        for j in st..=ed.min(f0_length - 1) {
            out[j] = 0.0;
        }
    }
}

fn get_multi_channel_f0(f0: &[f64], f0_length: usize, boundary_list: &[usize], number_of_boundaries: usize, multi: &mut [Vec<f64>]) {
    for i in 0..(number_of_boundaries / 2) {
        let st = boundary_list[i * 2];
        let ed = boundary_list[i * 2 + 1];
        multi[i].resize(f0_length, 0.0);
        for j in 0..st.min(f0_length) {
            multi[i][j] = 0.0;
        }
        for j in st..=ed.min(f0_length - 1) {
            multi[i][j] = f0[j];
        }
        for j in (ed + 1).min(f0_length)..f0_length {
            multi[i][j] = 0.0;
        }
    }
}

fn my_abs_i(x: i32) -> i32 {
    if x > 0 { x } else { -x }
}

fn extend_f0(
    extended_f0: &mut [f64],
    origin: usize,
    last_point: usize,
    shift: i32,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
) -> usize {
    let threshold = 4usize;
    let mut tmp_f0 = extended_f0[origin];
    let mut shifted_origin = origin;
    let distance = my_abs_i(last_point as i32 - origin as i32) as usize;

    let mut count = 0usize;
    for i in 0..=distance {
        let idx = origin as i32 + shift * i as i32 + shift;
        if idx <= 0 || idx as usize >= extended_f0.len() {
            break;
        }
        let idxu = idx as usize;
        let (best, _err) = select_best_f0(tmp_f0, &f0_candidates[idxu][..number_of_candidates], allowed_range);
        extended_f0[idxu] = best;
        if best == 0.0 {
            count += 1;
        } else {
            tmp_f0 = best;
            count = 0;
            shifted_origin = idxu;
        }
        if count == threshold {
            break;
        }
    }
    shifted_origin
}

fn make_sorted_order(boundary_list: &[usize], number_of_sections: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..number_of_sections).collect();
    order.sort_by_key(|&i| boundary_list[i * 2]);
    order
}

fn search_score(f0: f64, f0_candidates: &[f64], f0_scores: &[f64], number_of_candidates: usize) -> f64 {
    let mut score = 0.0_f64;
    for i in 0..number_of_candidates {
        if f0 == f0_candidates[i] && score < f0_scores[i] {
            score = f0_scores[i];
        }
    }
    score
}

fn merge_f0_sub(
    merged: &mut [f64],
    st1: usize,
    ed1: usize,
    f0_2: &[f64],
    st2: usize,
    ed2: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_candidates: usize,
) -> usize {
    if st1 <= st2 && ed1 >= ed2 {
        return ed1;
    }
    let mut score1 = 0.0_f64;
    let mut score2 = 0.0_f64;
    for i in st2..=ed1.min(merged.len() - 1) {
        score1 += search_score(merged[i], &f0_candidates[i], &f0_scores[i], number_of_candidates);
        score2 += search_score(f0_2[i], &f0_candidates[i], &f0_scores[i], number_of_candidates);
    }
    if score1 > score2 {
        for i in ed1..=ed2.min(merged.len() - 1) {
            merged[i] = f0_2[i];
        }
    } else {
        for i in st2..=ed2.min(merged.len() - 1) {
            merged[i] = f0_2[i];
        }
    }
    ed2
}

fn merge_f0(
    multi_channel_f0: &[Vec<f64>],
    boundary_list: &mut [usize],
    number_of_channels: usize,
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_candidates: usize,
    merged_f0: &mut [f64],
) {
    let order = make_sorted_order(boundary_list, number_of_channels);
    for i in 0..f0_length {
        merged_f0[i] = multi_channel_f0[0][i];
    }
    for i in 1..number_of_channels {
        let st = boundary_list[order[i] * 2];
        let ed = boundary_list[order[i] * 2 + 1];
        if st as i32 - boundary_list[1] as i32 > 0 {
            for j in st..=ed.min(f0_length - 1) {
                merged_f0[j] = multi_channel_f0[order[i]][j];
            }
            boundary_list[0] = st;
            boundary_list[1] = ed;
        } else {
            let new_end = merge_f0_sub(
                merged_f0,
                boundary_list[0],
                boundary_list[1],
                &multi_channel_f0[order[i]],
                st,
                ed,
                f0_candidates,
                f0_scores,
                number_of_candidates,
            );
            boundary_list[1] = new_end;
        }
    }
}

fn extend(
    multi_channel_f0: &mut [Vec<f64>],
    number_of_sections: usize,
    f0_length: usize,
    boundary_list: &mut Vec<usize>,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
) -> usize {
    let threshold = 100usize;
    let mut shifted_boundary_list = boundary_list.clone();
    for i in 0..number_of_sections {
        let ed = boundary_list[i * 2 + 1];
        let st = boundary_list[i * 2];
        shifted_boundary_list[i * 2 + 1] = extend_f0(
            &mut multi_channel_f0[i],
            ed,
            my_min_i(f0_length as i32 - 2, (ed + threshold) as i32) as usize,
            1,
            f0_candidates,
            number_of_candidates,
            allowed_range,
        );
        shifted_boundary_list[i * 2] = extend_f0(
            &mut multi_channel_f0[i],
            st,
            my_max_i(1, st as i32 - threshold as i32) as usize,
            -1,
            f0_candidates,
            number_of_candidates,
            allowed_range,
        );
    }

    let mut count = 0usize;
    let threshold_sec = 2200.0_f64;
    let mut mean_f0 = 0.0_f64;
    for i in 0..number_of_sections {
        let st = shifted_boundary_list[i * 2];
        let ed = shifted_boundary_list[i * 2 + 1];
        for j in st..ed.min(f0_length) {
            mean_f0 += multi_channel_f0[i][j];
        }
        if ed > st {
            mean_f0 /= (ed - st) as f64;
        }
        if mean_f0 > 0.0 && threshold_sec / mean_f0 < (ed as f64 - st as f64) {
            multi_channel_f0.swap(count, i);
            shifted_boundary_list.swap(count * 2, i * 2);
            shifted_boundary_list.swap(count * 2 + 1, i * 2 + 1);
            count += 1;
        }
    }

    *boundary_list = shifted_boundary_list;
    count
}

fn fix_step3(
    f0_step2: &[f64],
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    f0_scores: &[Vec<f64>],
    out: &mut [f64],
) {
    out.copy_from_slice(f0_step2);
    let mut boundary_list = Vec::with_capacity(f0_length);
    let number_of_boundaries = get_boundary_list(f0_step2, &mut boundary_list);

    let sections = number_of_boundaries / 2;
    let mut multi_channel_f0 = vec![vec![0.0_f64; f0_length]; sections];
    get_multi_channel_f0(f0_step2, f0_length, &boundary_list, number_of_boundaries, &mut multi_channel_f0);

    let number_of_channels = if sections == 0 {
        0
    } else {
        extend(
            &mut multi_channel_f0,
            sections,
            f0_length,
            &mut boundary_list,
            f0_candidates,
            number_of_candidates,
            allowed_range,
        )
    };

    if number_of_channels != 0 {
        merge_f0(
            &multi_channel_f0,
            &mut boundary_list,
            number_of_channels,
            f0_length,
            f0_candidates,
            f0_scores,
            number_of_candidates,
            out,
        );
    }
}

fn fix_step4(f0_step3: &[f64], f0_length: usize, threshold: usize, out: &mut [f64]) {
    out.copy_from_slice(f0_step3);
    let mut boundary_list = Vec::with_capacity(f0_length);
    let number_of_boundaries = get_boundary_list(f0_step3, &mut boundary_list);
    if number_of_boundaries / 2 < 2 {
        return;
    }
    for i in 0..(number_of_boundaries / 2 - 1) {
        let prev_end = boundary_list[i * 2 + 1];
        let next_start = boundary_list[(i + 1) * 2];
        let distance = next_start.saturating_sub(prev_end + 1);
        if distance >= threshold {
            continue;
        }
        let tmp0 = f0_step3[prev_end] + 1.0;
        let tmp1 = f0_step3[next_start] - 1.0;
        let coef = (tmp1 - tmp0) / (distance as f64 + 1.0);
        let mut count = 1usize;
        for j in (prev_end + 1)..=(next_start.saturating_sub(1)) {
            out[j] = tmp0 + coef * count as f64;
            count += 1;
        }
    }
}

fn fix_f0_contour(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
    best_f0_contour: &mut [f64],
) {
    let mut tmp1 = vec![0.0_f64; f0_length];
    let mut tmp2 = vec![0.0_f64; f0_length];
    search_f0_base(f0_candidates, f0_scores, f0_length, number_of_candidates, &mut tmp1);
    fix_step1(&tmp1, f0_length, 0.008, &mut tmp2);
    fix_step2(&tmp2, f0_length, 6, &mut tmp1);
    fix_step3(&tmp1, f0_length, number_of_candidates, f0_candidates, 0.18, f0_scores, &mut tmp2);
    fix_step4(&tmp2, f0_length, 9, best_f0_contour);
}

fn filtering_f0(a: &[f64; 2], b: &[f64; 2], x: &mut [f64], x_length: usize, st: usize, ed: usize, y: &mut [f64]) {
    let mut w = [0.0_f64, 0.0_f64];
    let mut tmp_x = vec![0.0_f64; x_length];

    for i in 0..st {
        x[i] = x[st];
    }
    for i in (ed + 1)..x_length {
        x[i] = x[ed];
    }

    for i in 0..x_length {
        let wt = x[i] + a[0] * w[0] + a[1] * w[1];
        tmp_x[x_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
        w[1] = w[0];
        w[0] = wt;
    }

    w = [0.0, 0.0];
    for i in 0..x_length {
        let wt = tmp_x[i] + a[0] * w[0] + a[1] * w[1];
        y[x_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
        w[1] = w[0];
        w[0] = wt;
    }
}

fn smooth_f0_contour(f0: &[f64], f0_length: usize, smoothed_f0: &mut [f64]) {
    let b: [f64; 2] = [0.0078202080334971724, 0.015640416066994345];
    let a: [f64; 2] = [1.7347257688092754, -0.76600660094326412];
    let lag = 300usize;
    let new_len = f0_length + lag * 2;
    let mut f0_contour = vec![0.0_f64; new_len];
    for i in 0..lag {
        f0_contour[i] = 0.0;
    }
    for i in lag..(lag + f0_length) {
        f0_contour[i] = f0[i - lag];
    }
    for i in (lag + f0_length)..new_len {
        f0_contour[i] = 0.0;
    }

    let mut boundary_list = Vec::with_capacity(new_len);
    let number_of_boundaries = get_boundary_list(&f0_contour, &mut boundary_list);
    let sections = number_of_boundaries / 2;
    let mut multi_channel_f0 = vec![vec![0.0_f64; new_len]; sections];
    get_multi_channel_f0(&f0_contour, new_len, &boundary_list, number_of_boundaries, &mut multi_channel_f0);

    for i in 0..sections {
        let st = boundary_list[i * 2];
        let ed = boundary_list[i * 2 + 1];
        filtering_f0(&a, &b, &mut multi_channel_f0[i], new_len, st, ed, &mut f0_contour);
        for j in st..=ed.min(new_len - 1) {
            if j >= lag && (j - lag) < smoothed_f0.len() {
                smoothed_f0[j - lag] = f0_contour[j];
            }
        }
    }
}

fn harvest_general_body_sub(
    boundary_f0_list: &[f64],
    number_of_channels: usize,
    f0_length: usize,
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    y_spectrum: &[Complex64],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) -> usize {
    let mut raw_f0_candidates = vec![vec![0.0_f64; f0_length]; number_of_channels];
    
    // 使用并行版本
    get_raw_f0_candidates_par(
        boundary_f0_list,
        actual_fs,
        y_length,
        temporal_positions,
        y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        &mut raw_f0_candidates,
    );

    let number_of_candidates = detect_official_f0_candidates(
        &raw_f0_candidates,
        number_of_channels,
        f0_length,
        max_candidates,
        f0_candidates,
    );
    overlap_f0_candidates(f0_length, number_of_candidates, f0_candidates);
    number_of_candidates
}

fn harvest_general_body(
    x: &[f64],
    x_length: usize,
    fs: i32,
    frame_period: i32,
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    speed: i32,
    temporal_positions: &mut [f64],
    f0: &mut [f64],
) {
    let adjusted_f0_floor = f0_floor * 0.9;
    let adjusted_f0_ceil = f0_ceil * 1.1;
    let number_of_channels = 1
        + ((adjusted_f0_ceil / adjusted_f0_floor).ln() / K_LOG2 * channels_in_octave) as usize;

    let mut boundary_f0_list = vec![0.0_f64; number_of_channels];
    for i in 0..number_of_channels {
        boundary_f0_list[i] = adjusted_f0_floor * 2.0_f64.powf((i as f64 + 1.0) / channels_in_octave);
    }

    let decimation_ratio = my_max_i(my_min_i(speed, 12), 1) as usize;
    let y_length = ((x_length as f64) / (decimation_ratio as f64)).ceil() as usize;
    let actual_fs = fs as f64 / decimation_ratio as f64;
    let extra = 2 * ((2.0 * actual_fs / boundary_f0_list[0]) as usize);
    let fft_size = get_suitable_fft_size(y_length + 5 + extra);

    let mut y = vec![0.0_f64; fft_size];
    let mut y_spectrum = vec![Complex64::new(0.0, 0.0); fft_size / 2 + 1];
    get_waveform_and_spectrum(x, x_length, y_length, fft_size, decimation_ratio, &mut y, &mut y_spectrum);

    let f0_length = get_samples_for_harvest(fs, x_length, frame_period as f64);
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * frame_period as f64 / 1000.0;
        f0[i] = 0.0;
    }

    let overlap_parameter = 7usize;
    let max_candidates = matlab_round(number_of_channels as f64 / 10.0).max(1) as usize * overlap_parameter;
    let mut f0_candidates = vec![vec![0.0_f64; max_candidates]; f0_length];
    let mut f0_candidates_score = vec![vec![0.0_f64; max_candidates]; f0_length];

    let base_candidates = harvest_general_body_sub(
        &boundary_f0_list,
        number_of_channels,
        f0_length,
        actual_fs,
        y_length,
        temporal_positions,
        &y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        max_candidates,
        &mut f0_candidates,
    );
    let number_of_candidates = base_candidates * overlap_parameter;

    // 使用并行版本
    refine_f0_candidates_par(
        &y[..y_length],
        y_length,
        actual_fs,
        temporal_positions,
        f0_floor,
        f0_ceil,
        &mut f0_candidates,
        &mut f0_candidates_score,
        number_of_candidates,
    );
    
    remove_unreliable_candidates(&mut f0_candidates, &mut f0_candidates_score, f0_length, number_of_candidates);

    let mut best_f0_contour = vec![0.0_f64; f0_length];
    fix_f0_contour(
        &f0_candidates,
        &f0_candidates_score,
        f0_length,
        number_of_candidates,
        &mut best_f0_contour,
    );
    smooth_f0_contour(&best_f0_contour, f0_length, f0);
}

pub fn get_samples_for_harvest(fs: i32, x_length: usize, frame_period: f64) -> usize {
    ((1000.0 * x_length as f64) / (fs as f64) / frame_period) as usize + 1
}

pub fn harvest(x: &[f64], fs: i32, option: &HarvestOption) -> (Vec<f64>, Vec<f64>) {
    let x_length = x.len();
    let f0_length = get_samples_for_harvest(fs, x_length, option.frame_period);
    let mut temporal_positions = vec![0.0_f64; f0_length];
    let mut f0 = vec![0.0_f64; f0_length];

    let target_fs = 8000.0_f64;
    let dimension_ratio = matlab_round(fs as f64 / target_fs);
    let channels_in_octave = 40.0_f64;

    if (option.frame_period - 1.0).abs() < 1e-12 {
        harvest_general_body(
            x,
            x_length,
            fs,
            1,
            option.f0_floor,
            option.f0_ceil,
            channels_in_octave,
            dimension_ratio,
            &mut temporal_positions,
            &mut f0,
        );
        return (temporal_positions, f0);
    }

    let basic_frame_period = 1;
    let basic_len = get_samples_for_harvest(fs, x_length, basic_frame_period as f64);
    let mut basic_f0 = vec![0.0_f64; basic_len];
    let mut basic_t = vec![0.0_f64; basic_len];
    harvest_general_body(
        x,
        x_length,
        fs,
        basic_frame_period,
        option.f0_floor,
        option.f0_ceil,
        channels_in_octave,
        dimension_ratio,
        &mut basic_t,
        &mut basic_f0,
    );

    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * option.frame_period / 1000.0;
        let idx = matlab_round(temporal_positions[i] * 1000.0).max(0) as usize;
        let idx = idx.min(basic_len - 1);
        f0[i] = basic_f0[idx];
    }
    (temporal_positions, f0)
}

