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
use crate::spetrum_arith::SpectrumArithmetic;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.2")]
fn _mm_fcmul_ps(a: __m128, b: __m128) -> __m128 {
    let mut temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let mut temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    temp1 = _mm_mul_ps(temp1, a);
    temp2 = _mm_mul_ps(temp2, a);
    temp2 = _mm_shuffle_ps(temp2, temp2, 0xB1);
    _mm_addsub_ps(temp1, temp2)
}

#[derive(Default)]
pub(crate) struct Sse42SpectrumF32 {}

impl SpectrumArithmetic<f32> for Sse42SpectrumF32 {
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<f32>],
        input: &[Complex<f32>],
        other: &[Complex<f32>],
        normalize_value: f32,
    ) {
        unsafe { self.mul_by_b_conj_normalize_impl(dst, input, other, normalize_value) }
    }
}

impl Sse42SpectrumF32 {
    #[target_feature(enable = "sse4.2")]
    fn mul_by_b_conj_normalize_impl(
        &self,
        dst: &mut [Complex<f32>],
        input: &[Complex<f32>],
        other: &[Complex<f32>],
        normalize_value: f32,
    ) {
        unsafe {
            let v_norm_factor = _mm_set1_ps(normalize_value);

            static CONJ_FACTORS: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            let conj_factors = _mm_loadu_ps(CONJ_FACTORS.as_ptr());

            for ((dst, input), other) in dst
                .chunks_exact_mut(8)
                .zip(input.chunks_exact(8))
                .zip(other.chunks_exact(8))
            {
                let vd0 = _mm_loadu_ps(input.as_ptr().cast());
                let vd1 = _mm_loadu_ps(input.get_unchecked(2..).as_ptr().cast());
                let vd2 = _mm_loadu_ps(input.get_unchecked(4..).as_ptr().cast());
                let vd3 = _mm_loadu_ps(input.get_unchecked(6..).as_ptr().cast());

                let mut vk0 = _mm_loadu_ps(other.as_ptr().cast());
                let mut vk1 = _mm_loadu_ps(other.get_unchecked(2..).as_ptr().cast());
                let mut vk2 = _mm_loadu_ps(other.get_unchecked(4..).as_ptr().cast());
                let mut vk3 = _mm_loadu_ps(other.get_unchecked(6..).as_ptr().cast());

                vk0 = _mm_xor_ps(vk0, conj_factors);
                vk1 = _mm_xor_ps(vk1, conj_factors);
                vk2 = _mm_xor_ps(vk2, conj_factors);
                vk3 = _mm_xor_ps(vk3, conj_factors);

                let p0 = _mm_mul_ps(_mm_fcmul_ps(vd0, vk0), v_norm_factor);
                let p1 = _mm_mul_ps(_mm_fcmul_ps(vd1, vk1), v_norm_factor);
                let p2 = _mm_mul_ps(_mm_fcmul_ps(vd2, vk2), v_norm_factor);
                let p3 = _mm_mul_ps(_mm_fcmul_ps(vd3, vk3), v_norm_factor);

                _mm_storeu_ps(dst.as_mut_ptr().cast(), p0);
                _mm_storeu_ps(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                _mm_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                _mm_storeu_ps(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst_rem = dst.chunks_exact_mut(8).into_remainder();
            let input_rem = input.chunks_exact(8).remainder();
            let other_rem = other.chunks_exact(8).remainder();

            for ((dst, input), other) in dst_rem
                .chunks_exact_mut(2)
                .zip(input_rem.chunks_exact(2))
                .zip(other_rem.chunks_exact(2))
            {
                let v0 = _mm_loadu_ps(input.as_ptr().cast());
                let mut v1 = _mm_loadu_ps(other.as_ptr().cast());

                v1 = _mm_xor_ps(v1, conj_factors);

                let p1 = _mm_mul_ps(_mm_fcmul_ps(v0, v1), v_norm_factor);
                _mm_storeu_ps(dst.as_mut_ptr().cast(), p1);
            }

            let dst_rem = dst_rem.chunks_exact_mut(2).into_remainder();
            let other_rem = other_rem.chunks_exact(2).remainder();
            let input_rem = input_rem.chunks_exact(2).remainder();

            for ((dst, input), other) in dst_rem
                .iter_mut()
                .zip(input_rem.iter())
                .zip(other_rem.iter())
            {
                let v0 = _mm_castsi128_ps(_mm_loadu_si64((input as *const Complex<f32>).cast()));
                let mut v1 =
                    _mm_castsi128_ps(_mm_loadu_si64((other as *const Complex<f32>).cast()));

                v1 = _mm_xor_ps(v1, conj_factors);

                let p1 = _mm_mul_ps(_mm_fcmul_ps(v0, v1), v_norm_factor);
                _mm_storeu_si64((dst as *mut Complex<f32>).cast(), _mm_castps_si128(p1));
            }
        }
    }
}
