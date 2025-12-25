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
use std::cmp::Ordering;

/// Frequency–domain **Gabor wavelet**.
///
/// This structure represents the canonical analytic Gabor atom defined in the
/// Fourier domain as:
///
/// ```text
/// ψ̂(k) = α · exp( -α · (k − μ)² ) · exp( −i · x₀ · (k − μ) )
/// ```
#[derive(Debug, Copy, Clone, Hash)]
pub struct GaborWavelet<T> {
    /// Center frequency μ (ω₀) of the wavelet.
    mu: T,
    /// Bandwidth parameter α.
    alpha: T,
    /// Precomputed α².
    alpha_s2: T,
    /// −x₀, the negative time translation used in the frequency-domain phase term.
    mx0: T,
}

impl<T: CwtSample> GaborWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// Creates a frequency–domain Gabor wavelet
    ///
    /// ```text
    /// ψ̂(k) = α · exp( -α · (k − μ)² ) · exp( −i · x₀ · (k − μ) )
    /// ```
    ///
    /// - `alpha` — bandwidth / scale parameter (controls time–frequency spread)
    /// - `mu`    — center frequency μ (ω₀)
    /// - `x0`    — time translation (usually 0)
    pub fn new(alpha: T, mu: T, x0: T) -> Self {
        Self {
            mu,
            alpha,
            alpha_s2: alpha * alpha,
            mx0: -x0,
        }
    }
}

impl<T: CwtSample> Default for GaborWavelet<T>
where
    f64: AsPrimitive<T>,
{
    /// α = 1, μ = 13.4, x₀ = 0 — a well-localized general-purpose Gabor wavelet.
    fn default() -> Self {
        Self::new(1f64.as_(), 13.4.as_(), 0.0.as_())
    }
}

impl<T: CwtSample> CwtWavelet<T> for GaborWavelet<T>
where
    f64: AsPrimitive<T>,
{
    fn make_wavelet(&self, omegas: &[T]) -> Result<Vec<Complex<T>>, ScaletError> {
        let mut out = try_vec![Complex::<T>::zero(); omegas.len()];

        if self.mx0.partial_cmp(&T::zero()).unwrap_or(Ordering::Equal) == Ordering::Equal {
            for (dst, &w) in out.iter_mut().zip(omegas.iter()) {
                let dwmu = w - self.mu;
                let z0 = (-dwmu * dwmu * self.alpha_s2).exp();

                *dst = Complex::new(self.alpha * z0, T::zero());
            }
        } else {
            for (dst, &w) in out.iter_mut().zip(omegas.iter()) {
                let dwmu = w - self.mu;
                let z0 = (-dwmu * dwmu * self.alpha_s2).exp();

                let ubd = (self.mx0 * dwmu).sincos();

                let z1 = Complex::new(ubd.1, ubd.0);
                *dst = z1 * self.alpha * z0;
            }
        }

        Ok(out)
    }
}
