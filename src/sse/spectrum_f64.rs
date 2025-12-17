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
use crate::spetrum_arith::SpectrumArithmetic;
use num_complex::Complex;
use std::arch::x86_64::*;

#[derive(Copy, Clone, Default)]
pub(crate) struct Sse42SpectrumF64 {}

impl SpectrumArithmetic<f64> for Sse42SpectrumF64 {
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        unsafe {
            self.mul_by_b_conj_normalize(dst, input, other, normalize_value);
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn _mm_fcmul_pd(a: __m128d, b: __m128d) -> __m128d {
    let mut temp1 = _mm_unpacklo_pd(b, b);
    let mut temp2 = _mm_unpackhi_pd(b, b);
    temp1 = _mm_mul_pd(temp1, a);
    temp2 = _mm_mul_pd(temp2, a);
    temp2 = _mm_shuffle_pd::<0x01>(temp2, temp2);
    _mm_addsub_pd(temp1, temp2)
}

impl Sse42SpectrumF64 {
    #[target_feature(enable = "sse4.2")]
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        unsafe {
            static CONJ_FACTORS: [f64; 2] = [0.0, -0.0];
            let conj_factors = _mm_loadu_pd(CONJ_FACTORS.as_ptr());

            let v_norm_factor = _mm_set1_pd(normalize_value);

            for ((dst, input), other) in dst.iter_mut().zip(input.iter()).zip(other.iter()) {
                let v0 = _mm_loadu_pd(input as *const Complex<f64> as *const _);
                let mut v1 = _mm_loadu_pd(other as *const Complex<f64> as *const _);

                v1 = _mm_xor_pd(v1, conj_factors);

                let lo = _mm_mul_pd(_mm_fcmul_pd(v0, v1), v_norm_factor);

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut _, lo);
            }
        }
    }
}
