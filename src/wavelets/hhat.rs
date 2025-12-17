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
use crate::{CwtWavelet, ScaletError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};

/// HHhat (Hilbert–Hermitian Hat) wavelet.
///
/// The **HHhat wavelet** is defined as the *analytic (Hilbert-transformed)*
/// version of the **Hermitian Hat** wavelet. It is a **complex, one-sided**
/// band-pass wavelet commonly used in Continuous Wavelet Transforms (CWT).
#[derive(Debug, Copy, Clone, Hash)]
pub struct HhhatWavelet<T> {
    /// The center frequency of the wavelet bandpass filter in the frequency domain.
    /// This parameter controls the location of the maximum energy in the frequency spectrum.
    mu: T,
}

impl<T: CwtSample> HhhatWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// Creates a new HHhat wavelet with the specified center frequency `mu`.
    ///
    /// # Parameters
    /// - `mu`: Center frequency of the wavelet in the frequency domain.
    pub fn new(mu: T) -> Self {
        Self { mu }
    }
}

impl<T: CwtSample> Default for HhhatWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// Returns a default HHhat wavelet with a center frequency of `mu = 5.0`.
    ///
    /// This value provides a reasonable balance between time and frequency
    /// localization for many signals.
    fn default() -> Self {
        Self::new(5.0f64.as_())
    }
}

impl<T: CwtSample> CwtWavelet<T> for HhhatWavelet<T>
where
    f64: AsPrimitive<T>,
{
    fn make_wavelet(&self, omegas: &[T]) -> Result<Vec<Complex<T>>, ScaletError> {
        let mut out = try_vec![Complex::<T>::zero(); omegas.len()];

        let c1 = T::TWO_OVER_5_SQ_PI_POWER_M0_25;
        let c0 = -0.5f64.as_();

        for (dst, &w) in out.iter_mut().zip(omegas.iter()) {
            let dwmu = w - self.mu;
            let dw2 = dwmu * dwmu;
            let sign_dw = if dwmu == T::zero() {
                T::zero()
            } else {
                1f64.as_().copysign(dwmu)
            };
            let a = c1 * (dwmu * (1f64.as_() + dwmu) * (c0 * dw2).exp()) * (1f64.as_() + sign_dw);
            *dst = Complex::new(a, T::zero());
        }

        Ok(out)
    }
}
