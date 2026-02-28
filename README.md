# Scalet

`scalet` is a **Continuous Wavelet Transform (CWT)** library written in Rust, focused on
speed, numerical robustness, and high-quality visualization.

It provides fast CWT execution for `f32` and `f64`, multiple wavelet types, flexible scale generation.

# Example

```rust
let cwt = Scalet::make_cwt_f32(
    Arc::new(MorletWavelet::new(13.4)),
    signal.len(),
    CwtOptions {
    l1_norm: true,
    ..Default::default()
    },
)?;
let cwt_result = cwt.execute(&signal)?;
// optionally draw scalogram using spectrograph
use spectrograph::{Normalizer, SpectrographOptions, rgb_spectrograph_color_f32, SpectrographFrame};
let scalogram = rgb_spectrograph_color_f32(&SpectrographFrame {
    data: std::borrow::Cow::Borrowed(cwt_result.data.borrow()),
    width: cwt_result.width,
    height: cwt_result.height,
    },  SpectrographOptions {
    out_width: 1920,
    out_height: 1080,
    normalizer: Normalizer::PowerSqrt,
    colormap: spectrograph::Colormap::Turbo,
})?;
```

----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.