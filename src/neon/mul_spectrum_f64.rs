/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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

use crate::neon::util::vcmulq_f64;
use crate::spetrum_arith::SpectrumArithmetic;
use num_complex::Complex;
use std::arch::aarch64::*;

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct NeonSpectrumF64 {}

impl SpectrumArithmetic<f64> for NeonSpectrumF64 {
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        unsafe {
            let v_norm_factor = vdupq_n_f64(normalize_value);

            static CONJ_FACTORS: [f64; 2] = [0.0, -0.0];
            let conj_factors = vreinterpretq_u64_f64(vld1q_f64(CONJ_FACTORS.as_ptr()));

            for ((dst, input), other) in dst
                .chunks_exact_mut(4)
                .zip(input.chunks_exact(4))
                .zip(other.chunks_exact(4))
            {
                let vd0 = vld1q_f64(input.as_ptr().cast());
                let vd1 = vld1q_f64(input.get_unchecked(1..).as_ptr().cast());
                let vd2 = vld1q_f64(input.get_unchecked(2..).as_ptr().cast());
                let vd3 = vld1q_f64(input.get_unchecked(3..).as_ptr().cast());

                let mut vk0 = vld1q_f64(other.as_ptr().cast());
                let mut vk1 = vld1q_f64(other.get_unchecked(1..).as_ptr().cast());
                let mut vk2 = vld1q_f64(other.get_unchecked(2..).as_ptr().cast());
                let mut vk3 = vld1q_f64(other.get_unchecked(3..).as_ptr().cast());

                vk0 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk0), conj_factors));
                vk1 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk1), conj_factors));
                vk2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk2), conj_factors));
                vk3 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk3), conj_factors));

                let p0 = vmulq_f64(vcmulq_f64(vd0, vk0), v_norm_factor);
                let p1 = vmulq_f64(vcmulq_f64(vd1, vk1), v_norm_factor);
                let p2 = vmulq_f64(vcmulq_f64(vd2, vk2), v_norm_factor);
                let p3 = vmulq_f64(vcmulq_f64(vd3, vk3), v_norm_factor);

                vst1q_f64(dst.as_mut_ptr().cast(), p0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), p1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), p3);
            }

            let dst_rem = dst.chunks_exact_mut(4).into_remainder();
            let other_rem = other.chunks_exact(4).remainder();
            let input_rem = input.chunks_exact(4).remainder();

            for ((dst, input), other) in dst_rem
                .iter_mut()
                .zip(input_rem.iter())
                .zip(other_rem.iter())
            {
                let v0 = vld1q_f64(input as *const Complex<f64> as *const f64);
                let mut v1 = vld1q_f64(other as *const Complex<f64> as *const f64);

                v1 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v1), conj_factors));

                let p1 = vmulq_f64(vcmulq_f64(v0, v1), v_norm_factor);
                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p1);
            }
        }
    }
}
