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
use crate::mla::fmla;
use crate::sample::CwtSample;
use crate::{CwtWavelet, ScaletError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};

/// Represents the **Morlet Wavelet** (or Gabor Wavelet), a fundamental analytic wavelet
/// used for Continuous Wavelet Transform (CWT) analysis.
///
/// The Morlet wavelet is defined as a complex sinusoid modulated by a Gaussian envelope.
/// This implementation calculates the frequency-domain filter, including a correction
/// term to ensure the zero-mean condition, which is critical for a valid wavelet.
#[derive(Debug, Copy, Clone, Hash)]
pub struct MorletWavelet<T> {
    /// The center frequency (mu or omega_0) of the complex exponential.
    /// This parameter controls the location of the filter's maximum energy in the frequency spectrum.
    mu: T,
    ks: T,
    c1: T,
}

impl<T: CwtSample> MorletWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// Creates a new `MorletWavelet` instance with the specified center frequency (mu).
    ///
    /// This constructor pre-calculates the internal DC-correction factor (`ks`) and the
    /// normalization factor based on standard analytical formulas for the Morlet wavelet.
    ///
    /// # Arguments
    ///
    /// * `mu` - The center frequency ($\omega_0$) of the wavelet. Common values are 5 or 6.
    pub fn new(mu: T) -> Self {
        let cs = (1f64.as_() + (-mu * mu).exp()
            - 2f64.as_() * ((-3f64 / 4f64).as_() * mu * mu).exp())
        .rsqrt();
        let ks = -(-0.5f64.as_() * mu * mu).exp();

        let c1 = cs * T::TWO_SQRT_BY_PI_POWER_0_25;

        Self { mu, ks, c1 }
    }
}

impl<T: CwtSample> Default for MorletWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// Provides a default, recommended instance of the Morlet Wavelet.
    ///
    /// The default implementation uses a center frequency (mu) of **13.4**.
    /// This value is often chosen
    /// because it yields a mu / sigma ratio that optimizes the balance between
    /// **frequency resolution** and **time localization**.
    fn default() -> Self {
        Self::new(13.4.as_())
    }
}

impl<T: CwtSample> CwtWavelet<T> for MorletWavelet<T>
where
    f64: AsPrimitive<T>,
{
    fn make_wavelet(&self, omegas: &[T]) -> Result<Vec<Complex<T>>, ScaletError> {
        let mut out = try_vec![Complex::<T>::zero(); omegas.len()];

        let c0 = -0.5f64.as_();

        for (dst, &w) in out.iter_mut().zip(omegas.iter()) {
            let dwmu = w - self.mu;
            let a = self.c1 * fmla(self.ks, (c0 * w * w).exp(), (c0 * dwmu * dwmu).exp());
            *dst = Complex::new(a, T::zero());
        }

        Ok(out)
    }
}
