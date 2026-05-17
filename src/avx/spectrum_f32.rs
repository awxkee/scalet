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
use crate::avx::storef::AvxStoreF;
use crate::complex_arith::ComplexArithmetic;
use num_complex::Complex;
use std::ops::Mul;

#[derive(Default)]
pub(crate) struct AvxSpectrumF32 {}

impl ComplexArithmetic<f32> for AvxSpectrumF32 {
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
        let v_norm_factor = AvxStoreF::dup(normalize_value);

        for ((dst, input), other) in dst
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<16>().0.iter())
            .zip(other.as_chunks::<16>().0.iter())
        {
            let vd0 = AvxStoreF::load(input);
            let vd1 = AvxStoreF::load(&input[4..]);
            let vd2 = AvxStoreF::load(&input[8..]);
            let vd3 = AvxStoreF::load(&input[12..]);

            let vk0 = AvxStoreF::load(other);
            let vk1 = AvxStoreF::load(&other[4..]);
            let vk2 = AvxStoreF::load(&other[8..]);
            let vk3 = AvxStoreF::load(&other[12..]);

            let mut d0 = AvxStoreF::mul_by_conj_b(vd0, vk0);
            let mut d1 = AvxStoreF::mul_by_conj_b(vd1, vk1);
            let mut d2 = AvxStoreF::mul_by_conj_b(vd2, vk2);
            let mut d3 = AvxStoreF::mul_by_conj_b(vd3, vk3);

            d0 = AvxStoreF::mul(d0, v_norm_factor);
            d1 = AvxStoreF::mul(d1, v_norm_factor);
            d2 = AvxStoreF::mul(d2, v_norm_factor);
            d3 = AvxStoreF::mul(d3, v_norm_factor);

            d0.write(dst);
            d1.write(&mut dst[4..]);
            d2.write(&mut dst[8..]);
            d3.write(&mut dst[12..]);
        }

        let dst_rem = dst.as_chunks_mut::<16>().1;
        let input_rem = input.as_chunks::<16>().1;
        let other_rem = other.as_chunks::<16>().1;

        for ((dst, input), other) in dst_rem
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(input_rem.as_chunks::<2>().0.iter())
            .zip(other_rem.as_chunks::<2>().0.iter())
        {
            let v0 = AvxStoreF::load2(input);
            let v1 = AvxStoreF::load2(other);

            let mut p1 = AvxStoreF::mul_by_conj_b(v0, v1);
            p1 = AvxStoreF::mul(p1, v_norm_factor);

            p1.write2(dst);
        }

        let dst_rem = dst_rem.as_chunks_mut::<2>().1;
        let input_rem = input_rem.as_chunks::<2>().1;
        let other_rem = other_rem.as_chunks::<2>().1;

        for ((dst, input), other) in dst_rem
            .iter_mut()
            .zip(input_rem.iter())
            .zip(other_rem.iter())
        {
            let v0 = AvxStoreF::load1(input);
            let v1 = AvxStoreF::load1(other);

            let mut p1 = AvxStoreF::mul_by_conj_b(v0, v1);
            p1 = AvxStoreF::mul(p1, v_norm_factor);

            p1.write1(dst);
        }
    }
}
