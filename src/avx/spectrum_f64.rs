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
use crate::avx::stored::AvxStoreD;
use crate::complex_arith::ComplexArithmetic;
use num_complex::Complex;
use std::arch::x86_64::*;
use std::ops::Mul;

// a * b.conj()
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn _mm256_fcmul_pd_conj_b(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_shuffle_pd::<0b0101>(a, a); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_shuffle_pd::<0b0000>(b, b); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_shuffle_pd::<0b1111>(b, b); // [c_im, c_im, d_im, d_im]

    _mm256_fmsubadd_pd(a, b_xx, _mm256_mul_pd(a_yx, b_yy))
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn _mm_fcmul_pd_conj_b(a: __m128d, b: __m128d) -> __m128d {
    let temp1 = _mm_unpacklo_pd(b, b);
    let mut temp2 = _mm_unpackhi_pd(b, b);
    temp2 = _mm_mul_pd(temp2, a);
    temp2 = _mm_shuffle_pd::<0x01>(temp2, temp2);
    _mm_fmsubadd_pd(temp1, a, temp2)
}

#[derive(Copy, Clone, Default)]
pub(crate) struct AvxSpectrumF64 {}

impl ComplexArithmetic<f64> for AvxSpectrumF64 {
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
        let v_norm_factor = AvxStoreD::dup(normalize_value);

        for ((dst, input), other) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<8>().0.iter())
            .zip(other.as_chunks::<8>().0.iter())
        {
            let vd0 = AvxStoreD::load(input);
            let vd1 = AvxStoreD::load(&input[2..]);
            let vd2 = AvxStoreD::load(&input[4..]);
            let vd3 = AvxStoreD::load(&input[6..]);

            let vk0 = AvxStoreD::load(other);
            let vk1 = AvxStoreD::load(&other[2..]);
            let vk2 = AvxStoreD::load(&other[4..]);
            let vk3 = AvxStoreD::load(&other[6..]);

            let d0 = AvxStoreD::mul(AvxStoreD::mul_by_conj_b(vd0, vk0), v_norm_factor);
            let d1 = AvxStoreD::mul(AvxStoreD::mul_by_conj_b(vd1, vk1), v_norm_factor);
            let d2 = AvxStoreD::mul(AvxStoreD::mul_by_conj_b(vd2, vk2), v_norm_factor);
            let d3 = AvxStoreD::mul(AvxStoreD::mul_by_conj_b(vd3, vk3), v_norm_factor);

            d0.write(dst);
            d1.write(&mut dst[2..]);
            d2.write(&mut dst[4..]);
            d3.write(&mut dst[6..]);
        }

        let dst_rem = dst.as_chunks_mut::<8>().1;
        let input_rem = input.as_chunks::<8>().1;
        let other_rem = other.as_chunks::<8>().1;

        for ((dst, input), other) in dst_rem
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(input_rem.as_chunks::<2>().0.iter())
            .zip(other_rem.as_chunks::<2>().0.iter())
        {
            let a0 = AvxStoreD::load(input);
            let b0 = AvxStoreD::load(other);

            let d0 = AvxStoreD::mul(AvxStoreD::mul_by_conj_b(a0, b0), v_norm_factor);

            d0.write(dst);
        }

        let dst_rem = dst_rem.as_chunks_mut::<2>().1;
        let input_rem = input_rem.as_chunks::<2>().1;
        let other_rem = other_rem.as_chunks::<2>().1;

        for ((dst, input), other) in dst_rem
            .iter_mut()
            .zip(input_rem.iter())
            .zip(other_rem.iter())
        {
            let v0 = AvxStoreD::load1(input);
            let v1 = AvxStoreD::load1(other);

            let d0 = AvxStoreD::mul(AvxStoreD::mul_by_conj_b(v0, v1), v_norm_factor);

            d0.write1(dst);
        }
    }
}
