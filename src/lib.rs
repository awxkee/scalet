/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(clippy::excessive_precision)]
#![cfg_attr(
    all(feature = "fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod cwt_executor;
mod cwt_filter;
mod err;
mod factory;
mod freqs;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod sample;
mod scale_bounds;
mod scales;
#[cfg(feature = "scalogram")]
mod scalogram;
mod spetrum_arith;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse;
mod wavelets;

use crate::factory::create_cwt;
use crate::freqs::scale_to_frequencies_impl;
#[cfg(feature = "scalogram")]
use crate::scalogram::{draw_scalogram_color_impl_f32, draw_scalogram_color_impl_f64};
pub use cwt_filter::CwtWavelet;
pub use err::ScaletError;
use num_complex::Complex;
#[cfg(feature = "scalogram")]
pub use scalogram::Colormap;
use std::sync::Arc;
pub use wavelets::{CmhatWavelet, HhhatWavelet, MorletWavelet};

/// Configuration options for the Continuous Wavelet Transform (CWT).
///
/// `CwtOptions` controls how scales are generated and how the resulting
/// wavelet coefficients are normalized. These parameters affect the
/// time–frequency resolution, redundancy, and amplitude interpretation
/// of the transform.
#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct CwtOptions {
    /// Scale generation strategy.
    ///
    /// Determines how scales are distributed (e.g. logarithmic or linear)
    /// and how `nv` is interpreted.
    pub scale_type: ScaleType,
    /// Number of voices per octave **or** total number of scales.
    ///
    /// The exact meaning depends on `scale_type`:
    /// - For logarithmic scales, `nv` specifies the number of voices per octave.
    /// - For linear scales, `nv` specifies the total number of scales.
    ///
    /// Larger values increase frequency resolution and redundancy at the
    /// cost of higher computational complexity.
    pub nv: usize,
    /// Whether to L1-normalize the CWT, which yields a more representative
    /// distribution of energies and component amplitudes than L2 (see [3]).
    /// If False (default True), uses L2 norm.
    pub l1_norm: bool,
}

impl Default for CwtOptions {
    fn default() -> Self {
        Self {
            nv: 32,
            scale_type: ScaleType::Log,
            l1_norm: true,
        }
    }
}

/// Defines the core functionality for executing a Continuous Wavelet Transform (CWT).
///
/// Implementors of this trait handle the pre-calculation of wavelet filters
/// and the efficient execution of the CWT against an input signal.
pub trait CwtExecutor<T> {
    /// Executes the Continuous Wavelet Transform on the input signal.
    ///
    /// The output is a 2D vector representing the scalogram. Each inner `Vec<Complex<T>>`
    /// corresponds to the wavelet coefficients for one scale (row), containing coefficients
    /// across the time axis (columns).
    /// The resulting dimensions are: `[num_scales, input_length]`.
    fn execute(&self, input: &[T]) -> Result<Vec<Vec<Complex<T>>>, ScaletError>;
    /// Executes the Continuous Wavelet Transform on a **complex-valued** input signal.
    ///
    /// This method allows direct analysis of analytic signals or signals that
    /// have already been transformed into the complex domain.
    ///
    /// # Parameters
    /// - `input`: Complex-valued time-domain signal to be analyzed.
    ///
    /// # Returns
    /// A two-dimensional vector representing the **scalogram**, with the same
    /// layout and interpretation as [`execute`](Self::execute).
    ///
    /// # Errors
    /// Returns `ScaletError` if the input length is incompatible with the
    /// executor configuration or if an internal FFT operation fails.
    fn execute_complex(&self, input: &[Complex<T>]) -> Result<Vec<Vec<Complex<T>>>, ScaletError>;
    /// Returns the expected length of the input signal this executor was built for.
    ///
    /// This is typically used to pre-calculate necessary internal parameters or
    /// check against the input signal length during `execute`.
    fn length(&self) -> usize;
    /// Provides a zero-copy view of the scale values used for this CWT instance.
    ///
    /// These values represent the dilation parameter 'a' for each step in the transform,
    /// and they are inversely proportional to the pseudo-frequency.
    ///
    /// # Returns
    ///
    /// An immutable slice (`&[T]`) containing the pre-calculated scale values.
    fn view_scales(&self) -> &[T];
}

/// The main entry point for constructing CWT executors.
///
/// `Scalet` provides convenient factory methods for creating pre-configured CWT executors,
/// handling the initialization of the chosen wavelet (e.g., Morlet) and the scale generation.
pub struct Scalet {}

impl Scalet {
    /// Creates a CWT executor configured for single-precision floating-point numbers (`f32`)
    /// using the default **Morlet Wavelet**.
    ///
    /// The resulting `CwtExecutor` is wrapped in an `Arc` for thread-safe sharing and
    /// object-safe dynamic dispatch.
    ///
    /// # Arguments
    ///
    /// * `length` - The expected length of the signal the executor will process.
    ///
    /// # Returns
    ///
    /// A `Result` containing an `Arc<dyn CwtExecutor<f32>>` or a `ScaletError`.
    pub fn make_morlet_f32(
        length: usize,
        options: CwtOptions,
    ) -> Result<Arc<dyn CwtExecutor<f32> + Send + Sync>, ScaletError> {
        create_cwt(
            Arc::new(MorletWavelet::default()),
            length,
            options.scale_type,
            options,
        )
    }

    /// Creates a CWT executor configured for double-precision floating-point numbers (`f64`)
    /// using the default **Morlet Wavelet**.
    ///
    /// This is suitable for applications requiring higher precision. See `make_morlet_f32`
    /// for argument details.
    ///
    /// # Arguments
    ///
    /// * `length` - The expected length of the signal the executor will process.
    ///
    /// # Returns
    ///
    /// A `Result` containing an `Arc<dyn CwtExecutor<f64>>` or a `ScaletError`.
    pub fn make_morlet_f64(
        length: usize,
        options: CwtOptions,
    ) -> Result<Arc<dyn CwtExecutor<f64> + Send + Sync>, ScaletError> {
        create_cwt(
            Arc::new(MorletWavelet::default()),
            length,
            options.scale_type,
            options,
        )
    }

    /// Creates a CWT executor for **single-precision (`f32`)** using a custom wavelet.
    ///
    /// # Arguments
    ///
    /// * `wavelet` – A reference-counted, thread-safe wavelet implementing `CwtWavelet<f32>`.
    /// * `length` – The expected length of the signal the executor will process.
    /// * `options` – CWT configuration parameters controlling scales, voices, and normalization.
    ///
    /// # Returns
    ///
    /// A `Result` containing an `Arc<dyn CwtExecutor<f32> + Send + Sync>` on success,
    /// or a `ScaletError` if creation fails.
    pub fn make_cwt_f32(
        wavelet: Arc<dyn CwtWavelet<f32> + Send + Sync>,
        length: usize,
        options: CwtOptions,
    ) -> Result<Arc<dyn CwtExecutor<f32> + Send + Sync>, ScaletError> {
        create_cwt(wavelet, length, options.scale_type, options)
    }

    /// Creates a CWT executor for **double-precision (`f64`)** using a custom wavelet.
    ///
    /// # Arguments
    ///
    /// * `wavelet` – A reference-counted, thread-safe wavelet implementing `CwtWavelet<f64>`.
    /// * `length` – The expected length of the signal the executor will process.
    /// * `options` – CWT configuration parameters controlling scales, voices, and normalization.
    ///
    /// # Returns
    ///
    /// A `Result` containing an `Arc<dyn CwtExecutor<f64> + Send + Sync>` on success,
    /// or a `ScaletError` if creation fails.
    pub fn make_cwt_f64(
        wavelet: Arc<dyn CwtWavelet<f64> + Send + Sync>,
        length: usize,
        options: CwtOptions,
    ) -> Result<Arc<dyn CwtExecutor<f64> + Send + Sync>, ScaletError> {
        create_cwt(wavelet, length, options.scale_type, options)
    }

    /// Converts wavelet scales to corresponding frequencies (f32 version).
    ///
    /// # Arguments
    ///
    /// * `wavelet` - An `Arc` to a type implementing `CwtWavelet<f32>`, used to determine the wavelet's center frequency.
    /// * `scales` - Slice of wavelet scales. Smaller scales correspond to higher frequencies.
    /// * `filter_length` - Length of the wavelet filter used in the CWT computation.
    /// * `sampling_frequency` - Sampling frequency of the original signal.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a `Vec<f32>` of frequencies corresponding to the input scales,
    /// or a `ScaletError` if the computation fails.
    ///
    /// # Behavior
    ///
    /// The output frequencies are in **natural order**, meaning that if `scales` are provided in
    /// **ascending order**, the resulting frequencies will be in **descending order** (high → low),
    /// because frequency is inversely proportional to scale.
    pub fn scales_to_frequencies_f32(
        wavelet: Arc<dyn CwtWavelet<f32> + Send + Sync>,
        scales: &[f32],
        filter_length: usize,
        sampling_frequency: f32,
    ) -> Result<Vec<f32>, ScaletError> {
        scale_to_frequencies_impl(wavelet, scales, sampling_frequency, filter_length)
    }

    /// Converts wavelet scales to corresponding frequencies (f64 version).
    ///
    /// Same behavior and parameters as [`scales_to_frequencies_f32`], but for `f64` data.
    pub fn scales_to_frequencies_f64(
        wavelet: Arc<dyn CwtWavelet<f64> + Send + Sync>,
        scales: &[f64],
        filter_length: usize,
        sampling_frequency: f64,
    ) -> Result<Vec<f64>, ScaletError> {
        scale_to_frequencies_impl(wavelet, scales, sampling_frequency, filter_length)
    }

    /// Draws a colorful scaleogram from CWT coefficients (f32 version).
    ///
    /// This function generates a color image representing the magnitude of the
    /// complex wavelet coefficients. The image is returned as a `Vec<u8>` in
    /// RGB format (3 bytes per pixel: R, G, B).
    ///
    /// # Parameters
    ///
    /// * `coeffs` - 2D slice of complex wavelet coefficients. Outer index corresponds to scales,
    ///   inner index corresponds to time. Typically, scales are in ascending order (low → high).
    /// * `out_width` - Width of the output image in pixels (time axis).
    /// * `out_height` - Height of the output image in pixels (scale axis).
    /// * `colormap` - The `Colormap` to use for mapping magnitude values to colors. This can
    ///   be any predefined colormap (e.g., Turbo, Jet) or custom.
    #[cfg(feature = "scalogram")]
    pub fn draw_scalogram_color_f32(
        coeffs: &[Vec<Complex<f32>>],
        out_width: usize,
        out_height: usize,
        colormap: Colormap,
    ) -> Result<Vec<u8>, ScaletError> {
        draw_scalogram_color_impl_f32(coeffs, out_width, out_height, colormap)
    }

    /// Draws a colorful scaleogram from CWT coefficients (f32 version).
    ///
    /// This function generates a color image representing the magnitude of the
    /// complex wavelet coefficients. The image is returned as a `Vec<u8>` in
    /// RGB format (3 bytes per pixel: R, G, B).
    ///
    /// # Parameters
    ///
    /// * `coeffs` - 2D slice of complex wavelet coefficients. Outer index corresponds to scales,
    ///   inner index corresponds to time. Typically, scales are in ascending order (low → high).
    /// * `out_width` - Width of the output image in pixels (time axis).
    /// * `out_height` - Height of the output image in pixels (scale axis).
    /// * `colormap` - The `Colormap` to use for mapping magnitude values to colors. This can
    ///   be any predefined colormap (e.g., Turbo, Jet) or custom.
    #[cfg(feature = "scalogram")]
    pub fn draw_scalogram_color_f64(
        coeffs: &[Vec<Complex<f64>>],
        out_width: usize,
        out_height: usize,
        colormap: Colormap,
    ) -> Result<Vec<u8>, ScaletError> {
        draw_scalogram_color_impl_f64(coeffs, out_width, out_height, colormap)
    }
}

/// Specifies how the wavelet scales are distributed in a Continuous Wavelet Transform (CWT).
///
/// The choice of `ScaleType` affects the time–frequency resolution of the transform
/// and how the `nv` parameter in `CwtOptions` is interpreted.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ScaleType {
    /// **Logarithmic spacing (Geometric):** Scales are spaced by powers of two (octaves).
    /// This is the standard choice for multi-resolution analysis, providing better frequency
    /// resolution at lower frequencies. The `nv` parameter represents the **voices per octave**.
    Log,
    /// **Linear spacing (Uniform):** Scales are spaced with a constant step size.
    /// This is typically used for narrowband analysis where a uniform resolution in the
    /// scale parameter is desired. The `nv` parameter represents the **total number of scales**.
    Linear,
}
