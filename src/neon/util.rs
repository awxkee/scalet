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
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) fn vcmulq_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    unsafe {
        let temp1 = vtrn1q_f32(rhs, rhs);
        let temp2 = vtrn2q_f32(rhs, vnegq_f32(rhs));
        let temp3 = vmulq_f32(temp2, lhs);
        let temp4 = vrev64q_f32(temp3);
        vfmaq_f32(temp4, temp1, lhs)
    }
}

#[inline(always)]
pub(crate) fn vcmul_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    unsafe {
        let temp1 = vtrn1_f32(rhs, rhs);
        let temp2 = vtrn2_f32(rhs, vneg_f32(rhs));
        let temp3 = vmul_f32(temp2, lhs);
        let temp4 = vrev64_f32(temp3);
        vfma_f32(temp4, temp1, lhs)
    }
}

#[inline]
#[cfg(feature = "fcma")]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmulq_conj_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    vcmlaq_rot270_f32(vcmlaq_f32(vdupq_n_f32(0.), rhs, lhs), rhs, lhs)
}

#[inline]
#[cfg(feature = "fcma")]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmul_conj_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    vcmla_rot270_f32(vcmla_f32(vdup_n_f32(0.), rhs, lhs), rhs, lhs)
}

#[inline(always)]
pub(crate) fn vcmulq_f64(lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {
    unsafe {
        let temp = vcombine_f64(vneg_f64(vget_high_f64(lhs)), vget_low_f64(lhs));
        let sum = vmulq_laneq_f64::<0>(lhs, rhs);
        vfmaq_laneq_f64::<1>(sum, temp, rhs)
    }
}

#[inline]
#[cfg(feature = "fcma")]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmul_conj_f64(lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {
    vcmlaq_rot270_f64(vcmlaq_f64(vdupq_n_f64(0.), rhs, lhs), rhs, lhs)
}
