/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use num_complex::Complex;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct SseStoreD {
    pub(crate) v: __m128d,
}

impl SseStoreD {
    #[inline]
    pub(crate) fn raw(v: __m128d) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn conj_flags() -> Self {
        static CONJ_FACTORS: [f64; 2] = [0.0, -0.0];
        unsafe {
            Self {
                v: _mm_loadu_pd(CONJ_FACTORS.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn dup(v: f64) -> Self {
        Self { v: _mm_set1_pd(v) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load1(data: &Complex<f64>) -> Self {
        unsafe { Self::raw(_mm_loadu_pd(data as *const Complex<f64> as *const f64)) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        let mut temp1 = _mm_unpacklo_pd(other.v, other.v);
        let mut temp2 = _mm_unpackhi_pd(other.v, other.v);
        temp1 = _mm_mul_pd(temp1, self.v);
        temp2 = _mm_mul_pd(temp2, self.v);
        temp2 = _mm_shuffle_pd::<0x01>(temp2, temp2);
        Self::raw(_mm_addsub_pd(temp1, temp2))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self::raw(_mm_xor_pd(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write1(&self, data: &mut Complex<f64>) {
        unsafe { _mm_storeu_pd(data as *mut Complex<f64> as *mut f64, self.v) }
    }
}

impl Mul<SseStoreD> for SseStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: SseStoreD) -> Self::Output {
        unsafe { Self::raw(_mm_mul_pd(self.v, rhs.v)) }
    }
}
