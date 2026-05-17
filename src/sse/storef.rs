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
pub(crate) struct SseStoreF {
    pub(crate) v: __m128,
}

impl SseStoreF {
    #[inline]
    pub(crate) fn raw(v: __m128) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn conj_flags() -> Self {
        static CONJ_FACTORS: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
        unsafe {
            Self {
                v: _mm_loadu_ps(CONJ_FACTORS.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn dup(v: f32) -> Self {
        Self { v: _mm_set1_ps(v) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load(data: &[Complex<f32>]) -> Self {
        unsafe { Self::raw(_mm_loadu_ps(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load1(data: &Complex<f32>) -> Self {
        unsafe {
            Self::raw(_mm_castsi128_ps(_mm_loadu_si64(
                data as *const Complex<f32> as *const _,
            )))
        }
    }

    // a * b
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        let mut temp1 = _mm_shuffle_ps::<0xA0>(other.v, other.v);
        let mut temp2 = _mm_shuffle_ps::<0xF5>(other.v, other.v);
        temp1 = _mm_mul_ps(temp1, self.v);
        temp2 = _mm_mul_ps(temp2, self.v);
        temp2 = _mm_shuffle_ps::<0xB1>(temp2, temp2);
        Self::raw(_mm_addsub_ps(temp1, temp2))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self::raw(_mm_xor_ps(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write(&self, data: &mut [Complex<f32>]) {
        unsafe { _mm_storeu_ps(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write1(&self, data: &mut Complex<f32>) {
        unsafe {
            _mm_storeu_si64(
                data as *mut Complex<f32> as *mut _,
                _mm_castps_si128(self.v),
            )
        }
    }
}

impl Mul<SseStoreF> for SseStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: SseStoreF) -> Self::Output {
        unsafe { Self::raw(_mm_mul_ps(self.v, rhs.v)) }
    }
}
