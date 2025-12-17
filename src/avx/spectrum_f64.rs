/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
use crate::avx::AvxSpectrumF32;
use crate::spetrum_arith::SpectrumArithmetic;
use num_complex::Complex;
use std::arch::x86_64::*;

// a * b.conj()
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn _mm256_fcmul_pd_conj_b(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_permute_pd::<0b0101>(a); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_permute_pd::<0b0000>(b); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_permute_pd::<0b1111>(b); // [c_im, c_im, d_im, d_im]

    _mm256_fmsubadd_pd(a, b_xx, _mm256_mul_pd(a_yx, b_yy))
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn _mm_fcmul_pd_conj_b(a: __m128d, b: __m128d) -> __m128d {
    let temp1 = _mm_unpacklo_pd(b, b);
    let mut temp2 = _mm_unpackhi_pd(b, b);
    temp2 = _mm_mul_pd(temp2, a);
    temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
    _mm_fmsubadd_pd(temp1, a, temp2)
}

#[derive(Copy, Clone, Default)]
pub(crate) struct AvxSpectrumF64 {}

impl SpectrumArithmetic<f64> for AvxSpectrumF64 {
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        unsafe {
            self.mul_by_b_conj_normalize_impl(dst, input, other, normalize_value);
        }
    }
}

impl AvxSpectrumF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_by_b_conj_normalize_impl(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        unsafe {
            let v_norm_factor = _mm256_set1_pd(normalize_value);

            for ((dst, input), other) in dst
                .chunks_exact_mut(8)
                .zip(input.chunks_exact(8))
                .zip(other.chunks_exact(8))
            {
                let vd0 = _mm256_loadu_pd(input.as_ptr().cast());
                let vd1 = _mm256_loadu_pd(input.get_unchecked(2..).as_ptr().cast());
                let vd2 = _mm256_loadu_pd(input.get_unchecked(4..).as_ptr().cast());
                let vd3 = _mm256_loadu_pd(input.get_unchecked(6..).as_ptr().cast());

                let vk0 = _mm256_loadu_pd(other.as_ptr().cast());
                let vk1 = _mm256_loadu_pd(other.get_unchecked(2..).as_ptr().cast());
                let vk2 = _mm256_loadu_pd(other.get_unchecked(4..).as_ptr().cast());
                let vk3 = _mm256_loadu_pd(other.get_unchecked(6..).as_ptr().cast());

                let d0 = _mm256_mul_pd(_mm256_fcmul_pd_conj_b(vd0, vk0), v_norm_factor);
                let d1 = _mm256_mul_pd(_mm256_fcmul_pd_conj_b(vd1, vk1), v_norm_factor);
                let d2 = _mm256_mul_pd(_mm256_fcmul_pd_conj_b(vd2, vk2), v_norm_factor);
                let d3 = _mm256_mul_pd(_mm256_fcmul_pd_conj_b(vd3, vk3), v_norm_factor);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), d0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), d1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), d2);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), d3);
            }

            let dst_rem = dst.chunks_exact_mut(8).into_remainder();
            let input_rem = input.chunks_exact(8).remainder();
            let other_rem = other.chunks_exact(8).remainder();

            for ((dst, input), other) in dst_rem
                .chunks_exact_mut(8)
                .zip(input_rem.chunks_exact(8))
                .zip(other_rem.chunks_exact(8))
            {
                let a0 = _mm256_loadu_pd(input.as_ptr().cast());
                let b0 = _mm256_loadu_pd(other.as_ptr().cast());

                let d0 = _mm256_mul_pd(_mm256_fcmul_pd_conj_b(a0, b0), v_norm_factor);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), d0);
            }

            let dst_rem = dst_rem.chunks_exact_mut(8).into_remainder();
            let input_rem = input_rem.chunks_exact(8).remainder();
            let other_rem = other_rem.chunks_exact(8).remainder();

            for ((dst, input), other) in dst_rem
                .iter_mut()
                .zip(input_rem.iter())
                .zip(other_rem.iter())
            {
                let v0 = _mm_loadu_pd(input as *const Complex<f64> as *const _);
                let v1 = _mm_loadu_pd(other as *const Complex<f64> as *const _);

                let lo = _mm_mul_pd(
                    _mm_fcmul_pd_conj_b(v0, v1),
                    _mm256_castpd256_pd128(v_norm_factor),
                );

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut _, lo);
            }
        }
    }
}
