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
use std::arch::aarch64::*;
use std::ops::Mul;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct NeonStoreD {
    pub(crate) v: float64x2_t,
}

impl NeonStoreD {
    #[inline]
    pub(crate) fn raw(v: float64x2_t) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn conj_flags() -> Self {
        static CONJ_FACTORS: [f64; 2] = [0.0, -0.0];
        unsafe {
            Self {
                v: vld1q_f64(CONJ_FACTORS.as_ptr()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn dup(v: f64) -> Self {
        Self { v: vdupq_n_f64(v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load(data: &[Complex<f64>]) -> Self {
        unsafe { Self::raw(vld1q_f64(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load1(data: &Complex<f64>) -> Self {
        unsafe { Self::raw(vld1q_f64(data as *const Complex<f64> as *const f64)) }
    }

    // a * b
    // Layout: [re, im]
    // re = ar*br - ai*bi
    // im = ar*bi + ai*br
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        let temp = vcombine_f64(vneg_f64(vget_high_f64(self.v)), vget_low_f64(self.v));
        let sum = vmulq_laneq_f64::<0>(self.v, other.v);
        Self::raw(vfmaq_laneq_f64::<1>(sum, temp, other.v))
    }

    // a * b.conj()
    // re = ar*br + ai*bi
    // im = ai*br - ar*bi
    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn mul_by_conj_b(self, other: Self) -> Self {
        Self::raw(vcmlaq_rot270_f64(
            vcmlaq_f64(vdupq_n_f64(0.), other.v, self.v),
            other.v,
            self.v,
        ))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self::raw(vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(self.v),
            vreinterpretq_u64_f64(other.v),
        )))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn write(&self, data: &mut [Complex<f64>]) {
        unsafe { vst1q_f64(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn write1(&self, data: &mut Complex<f64>) {
        unsafe { vst1q_f64(data as *mut Complex<f64> as *mut f64, self.v) }
    }
}

impl Mul<NeonStoreD> for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: NeonStoreD) -> Self::Output {
        unsafe { Self::raw(vmulq_f64(self.v, rhs.v)) }
    }
}
