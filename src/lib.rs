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
#![deny(clippy::unwrap_used)]
#![cfg_attr(
    all(feature = "fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod complex_arith;
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
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;
mod wavelets;

use crate::factory::create_cwt;
use crate::freqs::scale_to_frequencies_impl;
pub use cwt_filter::CwtWavelet;
pub use err::ScaletError;
use num_complex::Complex;
use std::sync::Arc;
pub use wavelets::{CmhatWavelet, GaborWavelet, HhhatWavelet, MorletWavelet};

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
    /// distribution of energies and component amplitudes than L2.
    /// If false (default true), uses L2 norm.
    pub l1_norm: bool,
    /// Whether to cache wavelet as well, or only FFTs.
    /// Depending on your data this may require notable amount on memory.
    /// Roughly `size_of<data_type>*(~300)*data_length*2` is required.
    pub full_cache: bool,
}

impl Default for CwtOptions {
    fn default() -> Self {
        Self {
            nv: 32,
            scale_type: ScaleType::Log,
            l1_norm: true,
            full_cache: false,
        }
    }
}

pub struct SynchrosqueezeOptions<T> {
    pub sample_rate: T,
    pub num_freq_bins: usize,
    pub threshold: T,
    pub nv: usize,
}

/// Defines the core functionality for executing a Continuous Wavelet Transform (CWT).
///
/// Implementors of this trait handle the pre-calculation of wavelet filters
/// and the efficient execution of the CWT against an input signal.
pub trait CwtExecutor<T>
where
    [Complex<T>]: ToOwned<Owned = Vec<Complex<T>>>,
{
    /// Executes the Continuous Wavelet Transform on the input signal.
    ///
    /// The output is a 2D vector representing the drawing. Each inner `Vec<Complex<T>>`
    /// corresponds to the wavelet coefficients for one scale (row), containing coefficients
    /// across the time axis (columns).
    /// The resulting dimensions are: `[num_scales, input_length]`.
    fn execute(&self, input: &[T]) -> Result<ScaletFrameMut<'_, Complex<T>>, ScaletError>;
    /// Executes the Continuous Wavelet Transform using caller-provided buffers.
    ///
    /// This method avoids internal allocations and allows the caller
    /// to reuse memory for performance-sensitive workloads.
    ///
    /// # Parameters
    /// - `input`: Real-valued time-domain signal.
    /// - `into_frame`: Preallocated output frame that will receive the coefficients.
    /// - `scratch`: Temporary working buffer. Must be at least
    ///   [`Self::scratch_length`] elements long.
    fn execute_with_scratch(
        &self,
        input: &[T],
        into_frame: &mut ScaletFrameMut<'_, Complex<T>>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), ScaletError>;
    /// Executes the Continuous Wavelet Transform on a **complex-valued** input signal.
    ///
    /// This method allows direct analysis of analytic signals or signals that
    /// have already been transformed into the complex domain.
    ///
    /// # Parameters
    /// - `input`: Complex-valued time-domain signal to be analyzed.
    ///
    /// # Returns
    /// A two-dimensional vector representing the **drawing**, with the same
    /// layout and interpretation as [`execute`](Self::execute).
    ///
    /// # Errors
    /// Returns `ScaletError` if the input length is incompatible with the
    /// executor configuration or if an internal FFT operation fails.
    fn execute_complex(
        &self,
        input: &[Complex<T>],
    ) -> Result<ScaletFrameMut<'_, Complex<T>>, ScaletError>;
    /// Executes the Continuous Wavelet Transform on a complex-valued signal
    /// using caller-provided buffers.
    ///
    /// This variant avoids internal allocations and is intended for
    /// high-performance or real-time scenarios.
    ///
    /// # Parameters
    /// - `input`: Complex-valued time-domain signal.
    /// - `into`: Preallocated output frame that will receive the coefficients.
    /// - `scratch`: Temporary working buffer. Must be at least
    ///   [`Self::scratch_length`] elements long.
    fn execute_complex_with_scratch(
        &self,
        input: &[Complex<T>],
        into: &mut ScaletFrameMut<'_, Complex<T>>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), ScaletError>;

    fn synchrosqueeze(
        &self,
        cwt_frame: &ScaletFrame<'_, Complex<T>>,
        options: SynchrosqueezeOptions<T>,
    ) -> Result<ScaletFrameMut<'_, Complex<T>>, ScaletError>;

    fn synchrosqueeze_into(
        &self,
        cwt_frame: &ScaletFrame<'_, Complex<T>>,
        into: &mut ScaletFrameMut<'_, Complex<T>>,
        options: SynchrosqueezeOptions<T>,
    ) -> Result<(), ScaletError>;

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
    /// Returns the required scratch buffer length for `_with_scratch` methods.
    ///
    /// The caller must allocate a scratch slice of at least this size
    /// before invoking `execute_with_scratch` or
    /// `execute_complex_with_scratch`.
    fn scratch_length(&self) -> usize;
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
    /// Same behavior and parameters as [`Scalet::scales_to_frequencies_f32`], but for `f64` data.
    pub fn scales_to_frequencies_f64(
        wavelet: Arc<dyn CwtWavelet<f64> + Send + Sync>,
        scales: &[f64],
        filter_length: usize,
        sampling_frequency: f64,
    ) -> Result<Vec<f64>, ScaletError> {
        scale_to_frequencies_impl(wavelet, scales, sampling_frequency, filter_length)
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

pub struct ScaletFrameMut<'a, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub data: BufferStoreMut<'a, T>,
    pub width: usize,
    pub height: usize,
}

pub struct ScaletFrame<'a, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub data: std::borrow::Cow<'a, [T]>,
    pub width: usize,
    pub height: usize,
}

impl<'a, T> ScaletFrameMut<'a, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub fn validate(&self) -> Result<(), ScaletError> {
        let total_size = isize::try_from(self.width)
            .map_err(|_| ScaletError::PointerOverlow)?
            .checked_mul(isize::try_from(self.height).map_err(|_| ScaletError::PointerOverlow)?)
            .ok_or(ScaletError::PointerOverlow)?;

        _ = total_size
            .checked_mul(size_of::<T>() as isize)
            .ok_or(ScaletError::PointerOverlow)? as usize;
        if self.data.borrow().len() != total_size as usize {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "Invalid frame size, expected {} but it was {}",
                    total_size,
                    self.data.borrow().len()
                )
                .to_string(),
            ));
        }
        Ok(())
    }

    pub fn as_ref(&'a self) -> ScaletFrame<'a, T>
    where
        T: Clone,
    {
        ScaletFrame {
            data: std::borrow::Cow::Borrowed(self.data.borrow()),
            width: self.width,
            height: self.height,
        }
    }

    pub fn into_ref(self) -> ScaletFrame<'a, T>
    where
        T: Clone,
        <[T] as ToOwned>::Owned: From<Vec<T>>,
    {
        ScaletFrame {
            data: std::borrow::Cow::Owned(self.data.borrow().to_vec()),
            width: self.width,
            height: self.height,
        }
    }

    pub fn matches(&self, other: &ScaletFrame<'_, T>) -> Result<(), ScaletError> {
        if self.width == other.width && self.height == other.height {
            Ok(())
        } else {
            Err(ScaletError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            })
        }
    }

    pub fn matches_transposed(&self, other: &ScaletFrame<'_, T>) -> Result<(), ScaletError> {
        if self.width == other.height && self.height == other.width {
            Ok(())
        } else {
            Err(ScaletError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            })
        }
    }
}

impl<'a, T> ScaletFrame<'a, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub fn validate(&self) -> Result<(), ScaletError> {
        let total_size = isize::try_from(self.width)
            .map_err(|_| ScaletError::PointerOverlow)?
            .checked_mul(isize::try_from(self.height).map_err(|_| ScaletError::PointerOverlow)?)
            .ok_or(ScaletError::PointerOverlow)?;

        _ = total_size
            .checked_mul(size_of::<T>() as isize)
            .ok_or(ScaletError::PointerOverlow)? as usize;
        if self.data.as_ref().len() != total_size as usize {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "Invalid frame size, expected {} but it was {}",
                    total_size,
                    self.data.as_ref().len()
                )
                .to_string(),
            ));
        }
        Ok(())
    }

    pub fn matches(&self, other: &ScaletFrameMut<'_, T>) -> Result<(), ScaletError> {
        if self.width == other.width && self.height == other.height {
            Ok(())
        } else {
            Err(ScaletError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            })
        }
    }

    pub fn matches_transposed(&self, other: &ScaletFrameMut<'_, T>) -> Result<(), ScaletError> {
        if self.width == other.height && self.height == other.width {
            Ok(())
        } else {
            Err(ScaletError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            })
        }
    }
}

/// Shared storage type
pub enum BufferStoreMut<'a, T> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T> BufferStoreMut<'_, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg(test)]
macro_rules! platform_test {
    ($(#[$meta:meta])* fn $name:ident() $body:block) => {
        #[wasm_bindgen_test::wasm_bindgen_test]
        $(#[$meta])*
        fn $name() $body
    };
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
macro_rules! platform_test {
    ($(#[$meta:meta])* fn $name:ident() $body:block) => {
        #[test]
        $(#[$meta])*
        fn $name() $body
    };
}

#[cfg(test)]
mod cwt_tests {
    use super::*;
    use num_complex::Complex;
    use std::f32::consts::PI as PI32;
    use std::f64::consts::PI as PI64;
    use std::sync::Arc;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    // ------------------------------------------------------------------ //
    // Helpers                                                              //
    // ------------------------------------------------------------------ //

    fn sine_f32(freq: f32, sample_rate: f32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| (2.0 * PI32 * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    fn sine_f64(freq: f64, sample_rate: f64, length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| (2.0 * PI64 * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    fn complex_sine_f32(freq: f32, sample_rate: f32, length: usize) -> Vec<Complex<f32>> {
        (0..length)
            .map(|i| {
                let phase = 2.0 * PI32 * freq * i as f32 / sample_rate;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    fn energy(frame: &ScaletFrameMut<'_, Complex<f32>>) -> Vec<f32> {
        let data = frame.data.borrow();
        let w = frame.width;
        (0..frame.height)
            .map(|row| {
                data[row * w..(row + 1) * w]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f32>()
            })
            .collect()
    }

    fn energy_f64(frame: &ScaletFrameMut<'_, Complex<f64>>) -> Vec<f64> {
        let data = frame.data.borrow();
        let w = frame.width;
        (0..frame.height)
            .map(|row| {
                data[row * w..(row + 1) * w]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
            })
            .collect()
    }

    fn peak_scale_index(energies: &[f32]) -> usize {
        energies
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn default_opts() -> CwtOptions {
        CwtOptions {
            nv: 16,
            scale_type: ScaleType::Log,
            l1_norm: true,
            full_cache: false,
        }
    }

    // ------------------------------------------------------------------ //
    // Basic construction                                                   //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_make_morlet_f32_succeeds() {
            let result = Scalet::make_morlet_f32(256, default_opts());
            assert!(result.is_ok(), "make_morlet_f32 should succeed");
        }
    }

    platform_test! {
        fn test_make_morlet_f64_succeeds() {
            let result = Scalet::make_morlet_f64(256, default_opts());
            assert!(result.is_ok(), "make_morlet_f64 should succeed");
        }
    }

    platform_test! {
        fn test_reported_length_matches_input() {
            let n = 512;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            assert_eq!(cwt.length(), n);
        }
    }

    platform_test! {
        fn test_scales_nonempty() {
            let cwt = Scalet::make_morlet_f32(256, default_opts()).unwrap();
            assert!(!cwt.view_scales().is_empty(), "scales should be non-empty");
        }
    }

    platform_test! {
        fn test_scales_all_positive() {
            let cwt = Scalet::make_morlet_f32(256, default_opts()).unwrap();
            assert!(
                cwt.view_scales().iter().all(|&s| s > 0.0),
                "all scales must be positive"
            );
        }
    }

    // ------------------------------------------------------------------ //
    // Output shape                                                         //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_output_width_equals_input_length() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(10.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            assert_eq!(frame.width, n, "output width must equal signal length");
        }
    }

    platform_test! {
        fn test_output_height_equals_num_scales() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(10.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            assert_eq!(
                frame.height,
                cwt.view_scales().len(),
                "output height must equal number of scales"
            );
        }
    }

    platform_test! {
        fn test_output_data_length_is_width_times_height() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(10.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            assert_eq!(
                frame.data.borrow().len(),
                frame.width * frame.height,
                "data buffer size must be width*height"
            );
        }
    }

    platform_test! {
        fn test_frame_validates_ok() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(10.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            assert!(frame.validate().is_ok(), "frame.validate() must pass");
        }
    }

    // ------------------------------------------------------------------ //
    // Zero signal                                                          //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_zero_signal_produces_near_zero_output_f32() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = vec![0.0f32; n];
            let frame = cwt.execute(&signal).unwrap();
            let max_amp = frame
                .data
                .borrow()
                .iter()
                .map(|c| c.norm())
                .fold(0.0f32, f32::max);
            assert!(
                max_amp < 1e-6,
                "zero input should produce near-zero output, got {max_amp}"
            );
        }
    }

    platform_test! {
        fn test_zero_signal_produces_near_zero_output_f64() {
            let n = 256;
            let cwt = Scalet::make_morlet_f64(n, default_opts()).unwrap();
            let signal = vec![0.0f64; n];
            let frame = cwt.execute(&signal).unwrap();
            let max_amp = frame
                .data
                .borrow()
                .iter()
                .map(|c| c.norm())
                .fold(0.0f64, f64::max);
            assert!(max_amp < 1e-12, "zero f64 input should give ~zero output");
        }
    }

    // ------------------------------------------------------------------ //
    // Frequency localisation                                               //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_peak_energy_at_correct_frequency() {
            let n = 512;
            let sample_rate = 512.0f32;
            let target_freq = 32.0f32;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(target_freq, sample_rate, n);
            let frame = cwt.execute(&signal).unwrap();
            let energies = energy(&frame);
            let peak_idx = peak_scale_index(&energies);

            let scales = cwt.view_scales().to_vec();
            let freqs = Scalet::scales_to_frequencies_f32(
                Arc::new(MorletWavelet::default()),
                &scales,
                n,
                sample_rate,
            )
            .unwrap();

            let peak_freq = freqs[peak_idx];
            let ratio = (peak_freq - target_freq).abs() / target_freq;
            assert!(
                ratio < 0.25,
                "peak frequency {peak_freq:.2} Hz should be within 25% of {target_freq} Hz"
            );
        }
    }

    platform_test! {
        fn test_two_frequencies_produce_two_peaks() {
            let n = 512;
            let sample_rate = 512.0f32;
            let (f1, f2) = (16.0f32, 64.0f32);
            let signal: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32 / sample_rate;
                    (2.0 * PI32 * f1 * t).sin() + (2.0 * PI32 * f2 * t).sin()
                })
                .collect();

            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let frame = cwt.execute(&signal).unwrap();
            let energies = energy(&frame);

            let mut sorted: Vec<(usize, f32)> = energies.iter().cloned().enumerate().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let peak1 = sorted[0].0;
            let peak2 = sorted
                .iter()
                .find(|&&(idx, _)| idx.abs_diff(peak1) > 2)
                .map(|&(idx, _)| idx);

            assert!(
                peak2.is_some(),
                "expected two separated energy peaks for two-frequency input"
            );
        }
    }

    // ------------------------------------------------------------------ //
    // Linearity                                                            //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_linearity_f32() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let s1 = sine_f32(10.0, 256.0, n);
            let s2 = sine_f32(30.0, 256.0, n);
            let sum: Vec<f32> = s1.iter().zip(s2.iter()).map(|(a, b)| a + b).collect();

            let f1 = cwt.execute(&s1).unwrap();
            let f2 = cwt.execute(&s2).unwrap();
            let fsum = cwt.execute(&sum).unwrap();

            let d1 = f1.data.borrow();
            let d2 = f2.data.borrow();
            let ds = fsum.data.borrow();

            let max_err = d1
                .iter()
                .zip(d2.iter())
                .zip(ds.iter())
                .map(|((a, b), s)| ((*a + *b) - *s).norm())
                .fold(0.0f32, f32::max);

            assert!(max_err < 1e-4, "CWT must be linear: max deviation was {max_err}");
        }
    }

    // ------------------------------------------------------------------ //
    // Scaling                                                              //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_double_amplitude_doubles_output_f32() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let s = sine_f32(20.0, 256.0, n);
            let s2: Vec<f32> = s.iter().map(|x| x * 2.0).collect();

            let f1 = cwt.execute(&s).unwrap();
            let f2 = cwt.execute(&s2).unwrap();

            let norm1: f32 = f1.data.borrow().iter().map(|c| c.norm()).sum();
            let norm2: f32 = f2.data.borrow().iter().map(|c| c.norm()).sum();

            let ratio = norm2 / norm1;
            assert!(
                (ratio - 2.0).abs() < 0.1,
                "doubling amplitude should double CWT magnitude, got ratio {ratio:.3}"
            );
        }
    }

    // ------------------------------------------------------------------ //
    // Complex input                                                        //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_execute_complex_returns_correct_shape() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = complex_sine_f32(20.0, 256.0, n);
            let frame = cwt.execute_complex(&signal).unwrap();
            assert_eq!(frame.width, n);
            assert_eq!(frame.height, cwt.view_scales().len());
        }
    }

    platform_test! {
        fn test_execute_complex_zero_input() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = vec![Complex::new(0.0f32, 0.0); n];
            let frame = cwt.execute_complex(&signal).unwrap();
            let max_amp = frame
                .data
                .borrow()
                .iter()
                .map(|c| c.norm())
                .fold(0.0f32, f32::max);
            assert!(max_amp < 1e-6);
        }
    }

    // ------------------------------------------------------------------ //
    // Scratch buffer API                                                   //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_execute_with_scratch_matches_execute() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(20.0, 256.0, n);
            let ref_frame = cwt.execute(&signal).unwrap();

            let num_scales = cwt.view_scales().len();
            let mut out_data = vec![Complex::new(0.0f32, 0.0); n * num_scales];
            let mut scratch = vec![Complex::new(0.0f32, 0.0); cwt.scratch_length()];
            let mut frame = ScaletFrameMut {
                data: BufferStoreMut::Borrowed(out_data.as_mut_slice()),
                width: n,
                height: num_scales,
            };

            cwt.execute_with_scratch(&signal, &mut frame, &mut scratch).unwrap();

            let max_err = ref_frame
                .data
                .borrow()
                .iter()
                .zip(frame.data.borrow().iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);

            assert!(max_err < 1e-5, "scratch API must match execute(), max diff: {max_err}");
        }
    }

    platform_test! {
        fn test_execute_complex_with_scratch_matches_execute_complex() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = complex_sine_f32(20.0, 256.0, n);
            let ref_frame = cwt.execute_complex(&signal).unwrap();

            let num_scales = cwt.view_scales().len();
            let mut out_data = vec![Complex::new(0.0f32, 0.0); n * num_scales];
            let mut scratch = vec![Complex::new(0.0f32, 0.0); cwt.scratch_length()];
            let mut frame = ScaletFrameMut {
                data: BufferStoreMut::Borrowed(out_data.as_mut_slice()),
                width: n,
                height: num_scales,
            };

            cwt.execute_complex_with_scratch(&signal, &mut frame, &mut scratch).unwrap();

            let max_err = ref_frame
                .data
                .borrow()
                .iter()
                .zip(frame.data.borrow().iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);

            assert!(max_err < 1e-5);
        }
    }

    // ------------------------------------------------------------------ //
    // ScaletFrame / validation helpers                                     //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_frame_as_ref_preserves_dimensions() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(10.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            let r = frame.as_ref();
            assert_eq!(r.width, frame.width);
            assert_eq!(r.height, frame.height);
        }
    }

    platform_test! {
        fn test_frame_matches_itself() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let signal = sine_f32(10.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            let r = frame.as_ref();
            assert!(frame.matches(&r).is_ok());
        }
    }

    // ------------------------------------------------------------------ //
    // Scale type: Linear                                                   //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_linear_scale_type_succeeds() {
            let opts = CwtOptions {
                nv: 32,
                scale_type: ScaleType::Linear,
                l1_norm: true,
                full_cache: false,
            };
            let cwt = Scalet::make_morlet_f32(256, opts).unwrap();
            let signal = sine_f32(10.0, 256.0, 256);
            let frame = cwt.execute(&signal).unwrap();
            assert!(frame.validate().is_ok());
        }
    }

    // ------------------------------------------------------------------ //
    // L1 vs L2 norm                                                        //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_l1_and_l2_norm_both_succeed() {
            let signal = sine_f32(10.0, 256.0, 256);
            for l1_norm in [true, false] {
                let opts = CwtOptions {
                    nv: 16,
                    scale_type: ScaleType::Log,
                    l1_norm,
                    full_cache: false,
                };
                let cwt = Scalet::make_morlet_f32(256, opts).unwrap();
                let frame = cwt.execute(&signal).unwrap();
                assert!(frame.validate().is_ok(), "failed with l1_norm={l1_norm}");
            }
        }
    }

    // ------------------------------------------------------------------ //
    // scales_to_frequencies                                                //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_scales_to_frequencies_f32_length_matches() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let scales = cwt.view_scales().to_vec();
            let freqs = Scalet::scales_to_frequencies_f32(
                Arc::new(MorletWavelet::default()),
                &scales,
                n,
                256.0,
            )
            .unwrap();
            assert_eq!(freqs.len(), scales.len());
        }
    }

    platform_test! {
        fn test_scales_to_frequencies_f32_all_positive() {
            let n = 256;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let scales = cwt.view_scales().to_vec();
            let freqs = Scalet::scales_to_frequencies_f32(
                Arc::new(MorletWavelet::default()),
                &scales,
                n,
                256.0,
            )
            .unwrap();
            assert!(freqs.iter().all(|&f| f > 0.0), "all converted frequencies must be positive");
        }
    }

    platform_test! {
        fn test_scales_to_frequencies_f64_length_matches() {
            let n = 256;
            let cwt = Scalet::make_morlet_f64(n, default_opts()).unwrap();
            let scales = cwt.view_scales().to_vec();
            let freqs = Scalet::scales_to_frequencies_f64(
                Arc::new(MorletWavelet::default()),
                &scales,
                n,
                256.0,
            )
            .unwrap();
            assert_eq!(freqs.len(), scales.len());
        }
    }

    platform_test! {
        fn test_scales_ascending_frequencies_descending() {
            let n = 512;
            let cwt = Scalet::make_morlet_f32(n, default_opts()).unwrap();
            let scales = cwt.view_scales().to_vec();
            let scales_asc = scales.windows(2).all(|w| w[0] <= w[1]);
            if !scales_asc {
                return;
            }
            let freqs = Scalet::scales_to_frequencies_f32(
                Arc::new(MorletWavelet::default()),
                &scales,
                n,
                512.0,
            )
            .unwrap();
            let freqs_desc = freqs.windows(2).all(|w| w[0] >= w[1]);
            assert!(freqs_desc, "ascending scales must yield descending frequencies");
        }
    }

    // ------------------------------------------------------------------ //
    // full_cache option                                                    //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_full_cache_produces_same_result_as_no_cache() {
            let n = 256;
            let signal = sine_f32(20.0, 256.0, n);
            let cwt_no_cache = Scalet::make_morlet_f32(n, CwtOptions { full_cache: false, ..default_opts() }).unwrap();
            let cwt_cache    = Scalet::make_morlet_f32(n, CwtOptions { full_cache: true,  ..default_opts() }).unwrap();
            let f1 = cwt_no_cache.execute(&signal).unwrap();
            let f2 = cwt_cache.execute(&signal).unwrap();
            let max_err = f1.data.borrow()
                .iter()
                .zip(f2.data.borrow().iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-5, "full_cache should not change results, max diff: {max_err}");
        }
    }

    // ------------------------------------------------------------------ //
    // f64 smoke test                                                       //
    // ------------------------------------------------------------------ //

    platform_test! {
        fn test_f64_execute_basic() {
            let n = 256;
            let cwt = Scalet::make_morlet_f64(n, default_opts()).unwrap();
            let signal = sine_f64(20.0, 256.0, n);
            let frame = cwt.execute(&signal).unwrap();
            assert_eq!(frame.width, n);
            assert!(frame.validate().is_ok());
            let total_energy: f64 = frame.data.borrow().iter().map(|c| c.norm_sqr()).sum();
            assert!(total_energy > 0.0, "f64 CWT output should have nonzero energy");
        }
    }

    platform_test! {
        fn test_f64_peak_at_correct_frequency() {
            let n = 512;
            let sample_rate = 512.0f64;
            let target_freq = 32.0f64;
            let cwt = Scalet::make_morlet_f64(n, default_opts()).unwrap();
            let signal = sine_f64(target_freq, sample_rate, n);
            let frame = cwt.execute(&signal).unwrap();

            let energies = energy_f64(&frame);
            let peak_idx = energies
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let scales = cwt.view_scales().to_vec();
            let freqs = Scalet::scales_to_frequencies_f64(
                Arc::new(MorletWavelet::default()),
                &scales,
                n,
                sample_rate,
            )
            .unwrap();

            let peak_freq = freqs[peak_idx];
            let ratio = (peak_freq - target_freq).abs() / target_freq;
            assert!(ratio < 0.25, "f64 peak at {peak_freq:.2} Hz, expected ~{target_freq} Hz");
        }
    }
}
