/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::wasm::stored::WasmStoreD;
use num_complex::Complex;
use std::ops::Mul;

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct WasmSpectrumF64 {}

impl ComplexArithmetic<f64> for WasmSpectrumF64 {
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        self.mul_by_b_conj_normalize_impl(dst, input, other, normalize_value)
    }
}

impl WasmSpectrumF64 {
    #[target_feature(enable = "simd128")]
    fn mul_by_b_conj_normalize_impl(
        &self,
        dst: &mut [Complex<f64>],
        input: &[Complex<f64>],
        other: &[Complex<f64>],
        normalize_value: f64,
    ) {
        let v_norm_factor = WasmStoreD::dup(normalize_value);
        let conj_factors = WasmStoreD::conj_flags();

        for ((dst, input), other) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<4>().0.iter())
            .zip(other.as_chunks::<4>().0.iter())
        {
            let vd0 = WasmStoreD::load(input);
            let vd1 = WasmStoreD::load(&input[1..]);
            let vd2 = WasmStoreD::load(&input[2..]);
            let vd3 = WasmStoreD::load(&input[3..]);

            let mut vk0 = WasmStoreD::load(other);
            let mut vk1 = WasmStoreD::load(&other[1..]);
            let mut vk2 = WasmStoreD::load(&other[2..]);
            let mut vk3 = WasmStoreD::load(&other[3..]);

            vk0 = vk0.xor(conj_factors);
            vk1 = vk1.xor(conj_factors);
            vk2 = vk2.xor(conj_factors);
            vk3 = vk3.xor(conj_factors);

            let p0 = WasmStoreD::mul(WasmStoreD::mul_by_complex(vd0, vk0), v_norm_factor);
            let p1 = WasmStoreD::mul(WasmStoreD::mul_by_complex(vd1, vk1), v_norm_factor);
            let p2 = WasmStoreD::mul(WasmStoreD::mul_by_complex(vd2, vk2), v_norm_factor);
            let p3 = WasmStoreD::mul(WasmStoreD::mul_by_complex(vd3, vk3), v_norm_factor);

            p0.write(dst);
            p1.write(&mut dst[1..]);
            p2.write(&mut dst[2..]);
            p3.write(&mut dst[3..]);
        }

        let dst_rem = dst.as_chunks_mut::<4>().1;
        let other_rem = other.as_chunks::<4>().1;
        let input_rem = input.as_chunks::<4>().1;

        for ((dst, input), other) in dst_rem
            .iter_mut()
            .zip(input_rem.iter())
            .zip(other_rem.iter())
        {
            let v0 = WasmStoreD::load1(input);
            let mut v1 = WasmStoreD::load1(other);

            v1 = v1.xor(conj_factors);

            let p1 = WasmStoreD::mul(WasmStoreD::mul_by_complex(v0, v1), v_norm_factor);
            p1.write1(dst);
        }
    }
}
