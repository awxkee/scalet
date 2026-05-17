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
use crate::neon::storef::NeonStoreF;
use num_complex::Complex;
use std::ops::Mul;

#[derive(Default)]
pub(crate) struct FcmaSpectrumF32 {}

impl ComplexArithmetic<f32> for FcmaSpectrumF32 {
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

impl FcmaSpectrumF32 {
    #[target_feature(enable = "fcma")]
    fn mul_by_b_conj_normalize_impl(
        &self,
        dst: &mut [Complex<f32>],
        input: &[Complex<f32>],
        other: &[Complex<f32>],
        normalize_value: f32,
    ) {
        let v_norm_factor = NeonStoreF::dup(normalize_value);

        for ((dst, input), other) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<8>().0.iter())
            .zip(other.as_chunks::<8>().0.iter())
        {
            let vd0 = NeonStoreF::load(input);
            let vd1 = NeonStoreF::load(&input[2..]);
            let vd2 = NeonStoreF::load(&input[4..]);
            let vd3 = NeonStoreF::load(&input[6..]);

            let vk0 = NeonStoreF::load(other);
            let vk1 = NeonStoreF::load(&other[2..]);
            let vk2 = NeonStoreF::load(&other[4..]);
            let vk3 = NeonStoreF::load(&other[6..]);

            let p0 = NeonStoreF::mul(NeonStoreF::mul_by_conj_b(vd0, vk0), v_norm_factor);
            let p1 = NeonStoreF::mul(NeonStoreF::mul_by_conj_b(vd1, vk1), v_norm_factor);
            let p2 = NeonStoreF::mul(NeonStoreF::mul_by_conj_b(vd2, vk2), v_norm_factor);
            let p3 = NeonStoreF::mul(NeonStoreF::mul_by_conj_b(vd3, vk3), v_norm_factor);

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
            let v0 = NeonStoreF::load(input);
            let v1 = NeonStoreF::load(other);

            let p1 = NeonStoreF::mul(NeonStoreF::mul_by_conj_b(v0, v1), v_norm_factor);
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
            let v0 = NeonStoreF::load1(input);
            let v1 = NeonStoreF::load1(other);

            let p1 = NeonStoreF::mul(NeonStoreF::mul_by_conj_b(v0, v1), v_norm_factor);
            p1.write1(dst);
        }
    }
}
