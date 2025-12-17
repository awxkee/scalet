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

// a * b.conj()
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn _m256_fcmul_a_by_b_conj(a: __m256, b: __m256) -> __m256 {
    // Extract real and imag parts from a
    let ar = _mm256_moveldup_ps(a); // duplicate even lanes (re parts)
    let ai = _mm256_movehdup_ps(a); // duplicate odd lanes (im parts)

    // Swap real/imag of b for cross terms
    let bswap = _mm256_permute_ps::<0b10110001>(b); // [im, re, im, re, ...]

    // re = ar*br - ai*bi
    // im = ar*bi + ai*br
    _mm256_fmsubadd_ps(ai, bswap, _mm256_mul_ps(ar, b))
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn _mm_fcmul_a_by_b_conj(a: __m128, b: __m128) -> __m128 {
    let temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    let mul2 = _mm_mul_ps(a, temp2);
    let mul2 = _mm_shuffle_ps::<0xB1>(mul2, mul2);
    _mm_fmsubadd_ps(a, temp1, mul2)
}

#[derive(Default)]
pub(crate) struct AvxSpectrumF32 {}

impl SpectrumArithmetic<f32> for AvxSpectrumF32 {
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

impl AvxSpectrumF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_by_b_conj_normalize_impl(
        &self,
        dst: &mut [Complex<f32>],
        input: &[Complex<f32>],
        other: &[Complex<f32>],
        normalize_value: f32,
    ) {
        unsafe {
            let v_norm_factor = _mm256_set1_ps(normalize_value);

            for ((dst, input), other) in dst
                .chunks_exact_mut(16)
                .zip(input.chunks_exact(16))
                .zip(other.chunks_exact(16))
            {
                let vd0 = _mm256_loadu_ps(input.as_ptr().cast());
                let vd1 = _mm256_loadu_ps(input.get_unchecked(4..).as_ptr().cast());
                let vd2 = _mm256_loadu_ps(input.get_unchecked(8..).as_ptr().cast());
                let vd3 = _mm256_loadu_ps(input.get_unchecked(12..).as_ptr().cast());

                let vk0 = _mm256_loadu_ps(other.as_ptr().cast());
                let vk1 = _mm256_loadu_ps(other.get_unchecked(4..).as_ptr().cast());
                let vk2 = _mm256_loadu_ps(other.get_unchecked(8..).as_ptr().cast());
                let vk3 = _mm256_loadu_ps(other.get_unchecked(12..).as_ptr().cast());

                let mut d0 = _m256_fcmul_a_by_b_conj(vd0, vk0);
                let mut d1 = _m256_fcmul_a_by_b_conj(vd1, vk1);
                let mut d2 = _m256_fcmul_a_by_b_conj(vd2, vk2);
                let mut d3 = _m256_fcmul_a_by_b_conj(vd3, vk3);

                d0 = _mm256_mul_ps(d0, v_norm_factor);
                d1 = _mm256_mul_ps(d1, v_norm_factor);
                d2 = _mm256_mul_ps(d2, v_norm_factor);
                d3 = _mm256_mul_ps(d3, v_norm_factor);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), d0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), d1);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), d2);
                _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), d3);
            }

            let dst_rem = dst.chunks_exact_mut(16).into_remainder();
            let input_rem = input.chunks_exact(16).remainder();
            let other_rem = other.chunks_exact(16).remainder();

            for ((dst, input), other) in dst_rem
                .chunks_exact_mut(2)
                .zip(input_rem.chunks_exact(2))
                .zip(other_rem.chunks_exact(2))
            {
                let v0 = _mm_loadu_ps(input.as_ptr().cast());
                let v1 = _mm_loadu_ps(other.as_ptr().cast());

                let p1 = _mm_fcmul_a_by_b_conj(v0, v1);
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
                let v1 = _mm_castsi128_ps(_mm_loadu_si64((other as *const Complex<f32>).cast()));

                let p1 = _mm_fcmul_a_by_b_conj(v0, v1);
                _mm_storeu_si64((dst as *mut Complex<f32>).cast(), _mm_castps_si128(p1));
            }
        }
    }
}
