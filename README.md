# High-Performance Harvest F0 Estimator (Rust)

**PS: 边界测试已完成，再有什么bug提issue吧，目前看最大误差在1e-27内**

这是一个基于 [WORLD](https://github.com/mmorise/World) 中 Harvest 算法的高性能 Rust 移植版本

感谢Gemini3pro提供了C++ to Rust的原始版本

本实现通过 **Rayon 并行计算** 在保持原有精度的前提下显著提升了运行速度

本仓库主要针对 **get_raw_f0_candidates_par(候选生成)** 和 **refine_f0_candidates_par(候选修正)** 进行了并行化，利用多核同时计算不同帧的 **get_refined_f0** 和不同的 **boundary_f0**

其次，对FFT的内存进行了缓存处理

```rust
thread_local! {
    static FFT_CACHE: RefCell<HashMap<usize, FftCtx>> = RefCell::new(HashMap::new());
}
```

每个工作线程维护自己的 FFT 上下文缓存。当并行任务请求特定长度（N）的 FFT 时，直接复用已创建的 Planner 和 Scratch Buffer，完全消除了重复分配内存和初始化规划器的开销

以上

## 咋用？

进入工作路径，运行`cargo run --release`就行

```bash
cd yourpath/harvest_rs
cargo run --release
```

`src/main.rs` 中改一下音频路径，即可测速

```rust
    let args = Args {
        input: "test.mp3".to_string(),  // 左边：音频路径
        mixdown: true,
        channel: 0,
        option: HarvestOption {
            f0_floor: 90.0,
            f0_ceil: 1600.0,
            frame_period: 10.0, // ms
        },
    };
```

顺便会把 `f0.txt` 存根目录


## Python 怎么个用法呢？

### 构建/安装（推荐 maturin）

在仓库根目录执行：

```bash
pip install -U maturin numpy
maturin develop -r
```

或者

```bash
pip install harvest_rs
```

安装发布版

### Python 调用示例

```python
import numpy as np
import harvest_rs
import librosa

opt = harvest_rs.HarvestOption(f0_floor=90.0, f0_ceil=1600.0, frame_period=10.0)

y, sr = librosa.load("test.mp3", sr=16000)
print(y.shape, sr)
t, f0 = harvest_rs.harvest(y, sr, opt)
print(t.shape, f0.shape) 
```
