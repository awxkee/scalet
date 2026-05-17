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
use crate::complex_arith::ComplexArithmetic;
use crate::err::try_vec;
use crate::mla::fmla;
use crate::sample::CwtSample;
use crate::{
    BufferStoreMut, CwtExecutor, CwtWavelet, ScaletError, ScaletFrame, ScaletFrameMut,
    SynchrosqueezeOptions,
};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;
use zaft::FftExecutor;

pub(crate) struct CommonCwtExecutor<T> {
    pub(crate) wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    pub(crate) fft_forward: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) fft_inverse: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) complex_arithmetic: Arc<dyn ComplexArithmetic<T> + Send + Sync>,
    pub(crate) scales: Vec<T>,
    pub(crate) psi: Vec<T>,
    pub(crate) execution_length: usize,
    pub(crate) l1_norm: bool,
    pub(crate) scratch_length: usize,
    pub(crate) built_wavelets: Vec<Complex<T>>,
}

impl<T: CwtSample> CommonCwtExecutor<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn execute_impl(
        &self,
        into: &mut ScaletFrameMut<'_, Complex<T>>,
        signal_fft: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ScaletError> {
        if self.execution_length != signal_fft.len() {
            return Err(ScaletError::InvalidInputSize(
                self.execution_length,
                signal_fft.len(),
            ));
        }

        if scratch.len() < self.scratch_length {
            return Err(ScaletError::InvalidScratchSize(
                self.scratch_length,
                scratch.len(),
            ));
        }

        let (scratch, _) = scratch.split_at_mut(self.scratch_length);

        // 1. Transform the input signal into the frequency domain (Spectral Domain).
        // This is the first step of the FFT-based convolution theorem.
        self.fft_forward
            .execute_with_scratch(signal_fft, scratch)
            .map_err(|x| ScaletError::FftError(x.to_string()))?;

        // Frequency vector
        let scales = self.view_scales();

        // Initialize temporary vectors and the final result structure.
        // current_psi: Workspace for the wavelet filter in the frequency domain for the current scale.
        let mut current_psi = try_vec![T::zero(); self.execution_length];
        // result: The final CWT drawing [num_scales][signal_length], storing complex coefficients.

        if self.execution_length != into.width {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "Invalid frame width, expected {} but it was {}",
                    self.execution_length, into.width
                )
                .to_string(),
            ));
        }
        if scales.len() != into.height {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "Invalid frame height, expected {} but it was {}",
                    scales.len(),
                    into.height
                )
                .to_string(),
            ));
        }
        into.validate()?;

        if self.built_wavelets.is_empty() {
            for (&scale, v_dst) in scales.iter().zip(
                into.data
                    .borrow_mut()
                    .chunks_exact_mut(self.execution_length),
            ) {
                // --- Step 1: Prepare Wavelet Filter for Convolution ---

                // Adjust the pre-calculated base phases (self.psi) by the current scale 'a'.
                // This implements the dilation property of the wavelet in the frequency domain.
                // The frequency-domain wavelet is scaled by 1/a, and its amplitude is scaled by 'a'.
                for (dst, &psi) in current_psi.iter_mut().zip(self.psi.iter()) {
                    *dst = psi * scale;
                }

                // Generate the final complex FFT filter for the current scale 'a'.
                let wavelet_fft = self.wavelet.make_wavelet(&current_psi)?;

                if wavelet_fft.len() != self.execution_length {
                    return Err(ScaletError::WaveletInvalidSize(
                        self.execution_length,
                        wavelet_fft.len(),
                    ));
                }

                // --- Step 2: Perform Convolution via Frequency-Domain Multiplication ---

                // Multiply the Signal FFT by the (conjugate of the) Wavelet FFT element-wise.
                // This is the core convolution theorem: IFFT(F(x) * F(y)) = x * y
                // additionally we'll normalize in this step as a part of optimization

                // Calculate the overall normalization factor (including the IFFT factor and CWT factor).
                let norm_factor = if self.l1_norm {
                    // L1 Normalization (Amplitude/Area): Typically divides by 'a' (scale).
                    // This current implementation only corrects for the unscaled IFFT (1/N).
                    1.0f64.as_() / v_dst.len().as_()
                } else {
                    // L2 Normalization (Energy)
                    1.0f64.as_() / (v_dst.len().as_() * scale.sqrt())
                };

                // input * other.conj() * normalize_value
                self.complex_arithmetic.mul_by_b_conj_normalize(
                    v_dst,
                    signal_fft,
                    &wavelet_fft,
                    norm_factor,
                );

                // --- Step 3: Inverse Transform to the Time Domain ---

                // Perform the Inverse FFT (IFFT) to transform the resulting spectrum back to the time domain.
                // The result in v_dst is the complex CWT coefficients Wx(a, b) at the current scale 'a'.
                self.fft_inverse
                    .execute_with_scratch(v_dst, scratch)
                    .map_err(|x| ScaletError::FftError(x.to_string()))?;
            }
        } else {
            for ((&scale, wavelet_fft), v_dst) in self
                .scales
                .iter()
                .zip(self.built_wavelets.chunks_exact(self.execution_length))
                .zip(
                    into.data
                        .borrow_mut()
                        .chunks_exact_mut(self.execution_length),
                )
            {
                // --- Step 2: Perform Convolution via Frequency-Domain Multiplication ---

                // Multiply the Signal FFT by the (conjugate of the) Wavelet FFT element-wise.
                // This is the core convolution theorem: IFFT(F(x) * F(y)) = x * y
                // additionally we'll normalize in this step as a part of optimization

                // Calculate the overall normalization factor (including the IFFT factor and CWT factor).
                let norm_factor = if self.l1_norm {
                    // L1 Normalization (Amplitude/Area): Typically divides by 'a' (scale).
                    // This current implementation only corrects for the unscaled IFFT (1/N).
                    1.0f64.as_() / v_dst.len().as_()
                } else {
                    // L2 Normalization (Energy)
                    1.0f64.as_() / (v_dst.len().as_() * scale.sqrt())
                };

                // input * other.conj() * normalize_value
                self.complex_arithmetic.mul_by_b_conj_normalize(
                    v_dst,
                    signal_fft,
                    wavelet_fft,
                    norm_factor,
                );

                // --- Step 3: Inverse Transform to the Time Domain ---

                // Perform the Inverse FFT (IFFT) to transform the resulting spectrum back to the time domain.
                // The result in v_dst is the complex CWT coefficients Wx(a, b) at the current scale 'a'.
                self.fft_inverse
                    .execute_with_scratch(v_dst, scratch)
                    .map_err(|x| ScaletError::FftError(x.to_string()))?;
            }
        }

        Ok(())
    }
}

impl<T: CwtSample> CwtExecutor<T> for CommonCwtExecutor<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn execute(&self, input: &[T]) -> Result<ScaletFrameMut<'_, Complex<T>>, ScaletError> {
        let mut frame = ScaletFrameMut {
            data: BufferStoreMut::Owned(
                try_vec![Complex::zero(); self.view_scales().len() * self.execution_length],
            ),
            height: self.view_scales().len(),
            width: self.execution_length,
        };

        let mut scratch = try_vec![Complex::zero(); self.scratch_length];

        self.execute_with_scratch(input, &mut frame, &mut scratch)?;
        Ok(frame)
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        into_frame: &mut ScaletFrameMut<'_, Complex<T>>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), ScaletError> {
        if self.execution_length != input.len() {
            return Err(ScaletError::InvalidInputSize(
                self.execution_length,
                input.len(),
            ));
        }

        let mut signal_fft: Vec<Complex<T>> = try_vec![Complex::<T>::default(); input.len()];
        for (dst, &src) in signal_fft.iter_mut().zip(input.iter()) {
            *dst = Complex::new(src, Zero::zero());
        }
        self.execute_impl(into_frame, &mut signal_fft, scratch)?;
        Ok(())
    }

    fn execute_complex(
        &self,
        input: &[Complex<T>],
    ) -> Result<ScaletFrameMut<'_, Complex<T>>, ScaletError> {
        if self.execution_length != input.len() {
            return Err(ScaletError::InvalidInputSize(
                self.execution_length,
                input.len(),
            ));
        }

        let mut frame = ScaletFrameMut {
            data: BufferStoreMut::Owned(
                try_vec![Complex::zero(); self.view_scales().len() * self.execution_length],
            ),
            height: self.view_scales().len(),
            width: self.execution_length,
        };

        let mut scratch = try_vec![Complex::zero(); self.scratch_length];

        let mut signal_fft = input.to_vec();
        self.execute_impl(&mut frame, &mut signal_fft, &mut scratch)?;
        Ok(frame)
    }

    fn execute_complex_with_scratch(
        &self,
        input: &[Complex<T>],
        into: &mut ScaletFrameMut<'_, Complex<T>>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), ScaletError> {
        if self.execution_length != input.len() {
            return Err(ScaletError::InvalidInputSize(
                self.execution_length,
                input.len(),
            ));
        }

        let mut signal_fft = input.to_vec();
        self.execute_impl(into, &mut signal_fft, scratch)?;
        Ok(())
    }

    fn synchrosqueeze(
        &self,
        cwt_frame: &ScaletFrame<'_, Complex<T>>,
        options: SynchrosqueezeOptions<T>,
    ) -> Result<ScaletFrameMut<'_, Complex<T>>, ScaletError> {
        let total_frame_size = isize::try_from(self.execution_length)
            .map_err(|_| ScaletError::PointerOverlow)?
            .checked_mul(
                isize::try_from(options.num_freq_bins).map_err(|_| ScaletError::PointerOverlow)?,
            )
            .ok_or(ScaletError::PointerOverlow)? as usize;
        let mut frame = ScaletFrameMut {
            data: BufferStoreMut::Owned(try_vec![Complex::zero(); total_frame_size]),
            height: options.num_freq_bins,
            width: self.execution_length,
        };
        self.synchrosqueeze_into(cwt_frame, &mut frame, options)?;
        Ok(frame)
    }

    fn synchrosqueeze_into(
        &self,
        cwt_frame: &ScaletFrame<'_, Complex<T>>,
        into: &mut ScaletFrameMut<'_, Complex<T>>,
        options: SynchrosqueezeOptions<T>,
    ) -> Result<(), ScaletError> {
        if self.execution_length != cwt_frame.width {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "Frame width should be {}, but it is {}",
                    self.execution_length, cwt_frame.width
                )
                .to_string(),
            ));
        }
        cwt_frame.validate()?;
        if into.height != options.num_freq_bins || into.width != self.execution_length {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "Frame height should be {} and width {} but it was {}x{}",
                    options.num_freq_bins, self.execution_length, into.width, into.height
                )
                .to_string(),
            ));
        }
        into.validate()?;
        let n = self.execution_length;
        let scales = self.view_scales();
        if scales.len() != cwt_frame.height {
            return Err(ScaletError::InvalidFrame(
                format_args!(
                    "CWT frame height is expected to be {} but it was {}",
                    scales.len(),
                    cwt_frame.height
                )
                .to_string(),
            ));
        }
        let ns = scales.len();

        let data = cwt_frame.data.as_ref();

        let dt: T = 1.0f64.as_() / options.sample_rate;

        let center_freq = self.wavelet.center_frequency();
        let f_min: T = center_freq / (scales[ns - 1] / options.sample_rate);
        let f_max: T = center_freq / (scales[0] / options.sample_rate);

        let log_ratio = (f_max / f_min).ln() / (options.num_freq_bins - 1).as_();
        let const_weight: T = (2.0f64.ln() / options.nv as f64).as_();

        for row in data.chunks_exact(cwt_frame.width) {
            for (b, &w_ab) in row.iter().enumerate() {
                let norm_sqr = fmla(w_ab.re, w_ab.re, w_ab.im * w_ab.im);
                let magnitude = norm_sqr.sqrt();

                if magnitude < options.threshold {
                    continue;
                }

                let (w_prev, w_next, divisor) = match b {
                    0 => (row[b], row[b + 1], dt),
                    b if b == n - 1 => (row[b - 1], row[b], dt),
                    _ => (row[b - 1], row[b + 1], 2.0f64.as_() * dt),
                };

                let dw_db = (w_next - w_prev) / divisor;

                let conj_w = w_ab.conj();
                let product = conj_w * dw_db;
                let omega_hz: T = (product.im / (T::TWO_PI * norm_sqr)).abs();

                if omega_hz < f_min || omega_hz > f_max {
                    continue; // outside our frequency range
                }

                let bin = (options.num_freq_bins - 1)
                    - ((omega_hz / f_min).ln() / log_ratio)
                        .round_to_usize()
                        .clamp(0, options.num_freq_bins - 1);

                let sst_data = into.data.borrow_mut();

                // inside loop, replace da_weight with:
                sst_data[bin * n + b].re = fmla(w_ab.re, const_weight, sst_data[bin * n + b].re);
                sst_data[bin * n + b].im = fmla(w_ab.im, const_weight, sst_data[bin * n + b].im);
            }
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn view_scales(&self) -> &[T] {
        &self.scales
    }

    fn scratch_length(&self) -> usize {
        self.scratch_length
    }
}
