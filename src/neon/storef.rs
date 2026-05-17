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
use std::arch::aarch64::float32x4_t;
use std::arch::aarch64::*;
use std::ops::Mul;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct NeonStoreF {
    pub(crate) v: float32x4_t,
}

impl NeonStoreF {
    #[inline]
    pub(crate) fn raw(v: float32x4_t) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn conj_flags() -> Self {
        static CONJ_FACTORS: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
        unsafe {
            Self {
                v: vld1q_f32(CONJ_FACTORS.as_ptr()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn dup(v: f32) -> Self {
        Self { v: vdupq_n_f32(v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load(data: &[Complex<f32>]) -> Self {
        unsafe { Self::raw(vld1q_f32(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn load1(data: &Complex<f32>) -> Self {
        unsafe {
            let ptr = data as *const Complex<f32> as *const f32;
            Self::raw(vcombine_f32(vld1_f32(ptr), vdup_n_f32(0.0)))
        }
    }

    // a * b
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        let temp1 = vtrn1q_f32(other.v, other.v);
        let temp2 = vtrn2q_f32(other.v, vnegq_f32(other.v));
        let temp3 = vmulq_f32(temp2, self.v);
        let temp4 = vrev64q_f32(temp3);
        Self::raw(vfmaq_f32(temp4, temp1, self.v))
    }

    // a * b.conj()
    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn mul_by_conj_b(self, other: Self) -> Self {
        Self::raw(vcmlaq_rot270_f32(
            vcmlaq_f32(vdupq_n_f32(0.), other.v, self.v),
            other.v,
            self.v,
        ))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self::raw(vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(self.v),
            vreinterpretq_u32_f32(other.v),
        )))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn write(&self, data: &mut [Complex<f32>]) {
        unsafe { vst1q_f32(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn write1(&self, data: &mut Complex<f32>) {
        unsafe { vst1_f32(data as *mut Complex<f32> as *mut f32, vget_low_f32(self.v)) }
    }
}

impl Mul<NeonStoreF> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: NeonStoreF) -> Self::Output {
        unsafe { Self::raw(vmulq_f32(self.v, rhs.v)) }
    }
}
