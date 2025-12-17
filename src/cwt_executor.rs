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
use crate::err::try_vec;
use crate::sample::CwtSample;
use crate::spetrum_arith::SpectrumArithmetic;
use crate::{CwtExecutor, CwtWavelet, ScaletError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;
use zaft::FftExecutor;

pub(crate) struct CommonCwtExecutor<T> {
    pub(crate) wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    pub(crate) fft_forward: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) fft_inverse: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) spectrum_arithmetic: Arc<dyn SpectrumArithmetic<T> + Send + Sync>,
    pub(crate) scales: Vec<T>,
    pub(crate) psi: Vec<T>,
    pub(crate) execution_length: usize,
    pub(crate) l1_norm: bool,
}

impl<T: CwtSample> CommonCwtExecutor<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn execute_impl(
        &self,
        signal_fft: &mut [Complex<T>],
    ) -> Result<Vec<Vec<Complex<T>>>, ScaletError> {
        if self.execution_length != signal_fft.len() {
            return Err(ScaletError::InvalidInputSize(
                self.execution_length,
                signal_fft.len(),
            ));
        }

        // 1. Transform the input signal into the frequency domain (Spectral Domain).
        // This is the first step of the FFT-based convolution theorem.
        self.fft_forward
            .execute(signal_fft)
            .map_err(|x| ScaletError::FftError(x.to_string()))?;

        // Frequency vector
        let scales = self.view_scales();

        // Initialize temporary vectors and the final result structure.
        // current_psi: Workspace for the wavelet filter in the frequency domain for the current scale.
        let mut current_psi = try_vec![T::zero(); self.execution_length];
        // result: The final CWT scalogram [num_scales][signal_length], storing complex coefficients.
        let mut result = try_vec![try_vec![Complex::new(T::zero(), T::zero()); self.execution_length]; scales.len()];

        for (&scale, v_dst) in scales.iter().zip(result.iter_mut()) {
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
            self.spectrum_arithmetic.mul_by_b_conj_normalize(
                v_dst,
                &signal_fft,
                &wavelet_fft,
                norm_factor,
            );

            // --- Step 3: Inverse Transform to the Time Domain ---

            // Perform the Inverse FFT (IFFT) to transform the resulting spectrum back to the time domain.
            // The result in v_dst is the complex CWT coefficients Wx(a, b) at the current scale 'a'.
            self.fft_inverse
                .execute(v_dst)
                .map_err(|x| ScaletError::FftError(x.to_string()))?;
        }

        Ok(result)
    }
}

impl<T: CwtSample> CwtExecutor<T> for CommonCwtExecutor<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn execute(&self, input: &[T]) -> Result<Vec<Vec<Complex<T>>>, ScaletError> {
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
        self.execute_impl(&mut signal_fft)
    }

    fn execute_complex(&self, input: &[Complex<T>]) -> Result<Vec<Vec<Complex<T>>>, ScaletError> {
        if self.execution_length != input.len() {
            return Err(ScaletError::InvalidInputSize(
                self.execution_length,
                input.len(),
            ));
        }

        let mut signal_fft = input.to_vec();
        self.execute_impl(&mut signal_fft)
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn view_scales(&self) -> &[T] {
        &self.scales
    }
}
