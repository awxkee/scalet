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

/// Represents the **Complex Mexican Hat (Cmhat)** Wavelet, an analytic-like filter
/// used for Continuous Wavelet Transform (CWT) analysis.
///
/// The Complex Mexican Hat is a variant of the Difference of Gaussians (DOG) wavelet,
/// implemented in the frequency domain to approximate an analytic wavelet by zeroing out
/// negative frequencies, similar to the method often applied to the Morlet wavelet.
#[derive(Debug, Copy, Clone, Hash)]
pub struct CmhatWavelet<T> {
    /// The center frequency of the wavelet bandpass filter in the frequency domain.
    /// This parameter controls the location of the maximum energy in the frequency spectrum.
    mu: T,
    q0: T,
    q1: T,
}

impl<T: CwtSample> CmhatWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// Creates a new `CmhatWavelet` instance with the specified center frequency mu.
    ///
    /// This constructor pre-calculates internal constants  based on
    /// a default unit scale and an assumed order/exponent (5/2 = 2.5).
    /// These constants are then scaled correctly during the CWT execution loop.
    pub fn new(mu: T, s: T) -> Self {
        let c0 = (5f64 / 2f64).as_();
        let q0 = s.pow(c0);
        let q1 = -s * s;
        Self { mu, q0, q1 }
    }
}

impl<T: CwtSample> Default for CmhatWavelet<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self::new(1.0f64.as_(), 1.0f64.as_())
    }
}

impl<T: CwtSample> CwtWavelet<T> for CmhatWavelet<T>
where
    f64: AsPrimitive<T>,
{
    fn make_wavelet(&self, omegas: &[T]) -> Result<Vec<Complex<T>>, ScaletError> {
        let mut out = try_vec![Complex::<T>::zero(); omegas.len()];

        let c1 = T::TWO_S2_OVER_3_PI_POWER_M0_25;

        for (dst, &w) in out.iter_mut().zip(omegas.iter()) {
            let dwmu = w - self.mu;
            let dw2 = dwmu * dwmu;
            let nullifier = if dwmu >= T::zero() {
                1f64.as_()
            } else {
                0f64.as_()
            };
            let a = c1 * (self.q0 * dw2 * (self.q1 * dw2 * 0.5f64.as_()).exp() * nullifier);
            *dst = Complex::new(a, T::zero());
        }

        Ok(out)
    }
}
