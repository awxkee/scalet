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
use crate::complex_arith::ComplexArithmetic;
use crate::sse::storef::SseStoreF;
use num_complex::Complex;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

#[inline]
#[target_feature(enable = "sse4.2")]
fn _mm_fcmul_ps(a: __m128, b: __m128) -> __m128 {
    let mut temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let mut temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    temp1 = _mm_mul_ps(temp1, a);
    temp2 = _mm_mul_ps(temp2, a);
    temp2 = _mm_shuffle_ps::<0xB1>(temp2, temp2);
    _mm_addsub_ps(temp1, temp2)
}

#[derive(Default)]
pub(crate) struct Sse42SpectrumF32 {}

impl ComplexArithmetic<f32> for Sse42SpectrumF32 {
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
        let v_norm_factor = SseStoreF::dup(normalize_value);

        let conj_factors = SseStoreF::conj_flags();

        for ((dst, input), other) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<8>().0.iter())
            .zip(other.as_chunks::<8>().0.iter())
        {
            let vd0 = SseStoreF::load(input);
            let vd1 = SseStoreF::load(&input[2..]);
            let vd2 = SseStoreF::load(&input[4..]);
            let vd3 = SseStoreF::load(&input[6..]);

            let mut vk0 = SseStoreF::load(other);
            let mut vk1 = SseStoreF::load(&other[2..]);
            let mut vk2 = SseStoreF::load(&other[4..]);
            let mut vk3 = SseStoreF::load(&other[6..]);

            vk0 = vk0.xor(conj_factors);
            vk1 = vk1.xor(conj_factors);
            vk2 = vk2.xor(conj_factors);
            vk3 = vk3.xor(conj_factors);

            let p0 = SseStoreF::mul(SseStoreF::mul_by_complex(vd0, vk0), v_norm_factor);
            let p1 = SseStoreF::mul(SseStoreF::mul_by_complex(vd1, vk1), v_norm_factor);
            let p2 = SseStoreF::mul(SseStoreF::mul_by_complex(vd2, vk2), v_norm_factor);
            let p3 = SseStoreF::mul(SseStoreF::mul_by_complex(vd3, vk3), v_norm_factor);

            p0.write(dst);
            p1.write(&mut dst[2..]);
            p2.write(&mut dst[4..]);
            p3.write(&mut dst[6..]);
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
            let v0 = SseStoreF::load(input);
            let mut v1 = SseStoreF::load(other);

            v1 = v1.xor(conj_factors);

            let p1 = SseStoreF::mul(SseStoreF::mul_by_complex(v0, v1), v_norm_factor);
            p1.write(dst);
        }

        let dst_rem = dst_rem.as_chunks_mut::<2>().1;
        let other_rem = other_rem.as_chunks::<2>().1;
        let input_rem = input_rem.as_chunks::<2>().1;

        for ((dst, input), other) in dst_rem
            .iter_mut()
            .zip(input_rem.iter())
            .zip(other_rem.iter())
        {
            let v0 = SseStoreF::load1(input);
            let mut v1 = SseStoreF::load1(other);

            v1 = v1.xor(conj_factors);

            let p1 = SseStoreF::mul(SseStoreF::mul_by_complex(v0, v1), v_norm_factor);
            p1.write1(dst);
        }
    }
}
