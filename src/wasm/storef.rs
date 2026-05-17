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

use num_complex::Complex;
use std::arch::wasm32::*;
use std::ops::Mul;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct WasmStoreF {
    pub(crate) v: v128,
}

impl WasmStoreF {
    #[inline]
    pub(crate) fn raw(v: v128) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn conj_flags() -> Self {
        // [0.0, -0.0, 0.0, -0.0] — xor with this to conjugate
        Self {
            v: f32x4(0.0, -0.0, 0.0, -0.0),
        }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn dup(v: f32) -> Self {
        Self { v: f32x4_splat(v) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load(data: &[Complex<f32>]) -> Self {
        unsafe { Self::raw(v128_load(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load1(data: &Complex<f32>) -> Self {
        unsafe { Self::raw(v128_load64_zero(data as *const Complex<f32> as *const u64)) }
    }

    // a * b
    // re = ar*br - ai*bi
    // im = ar*bi + ai*br
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        let temp1 = i32x4_shuffle::<0, 0, 2, 2>(other.v, other.v);
        let neg_other = f32x4_neg(other.v);
        let temp2 = i32x4_shuffle::<1, 5, 3, 7>(other.v, neg_other);
        let temp3 = f32x4_mul(temp2, self.v);
        let temp4 = i32x4_shuffle::<1, 0, 3, 2>(temp3, temp3);
        Self::raw(f32x4_add(f32x4_mul(temp1, self.v), temp4))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self::raw(v128_xor(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write(&self, data: &mut [Complex<f32>]) {
        unsafe { v128_store(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write1(&self, data: &mut Complex<f32>) {
        unsafe { v128_store64_lane::<0>(self.v, data as *mut Complex<f32> as *mut u64) }
    }
}

impl Mul<WasmStoreF> for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: WasmStoreF) -> Self::Output {
        Self::raw(f32x4_mul(self.v, rhs.v))
    }
}

#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::*;
    use num_complex::Complex;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn ref_mul_conj(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
        a * b.conj()
    }

    fn ref_mul_complex(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
        a * b
    }

    fn wasm_mul_conj(a: &[Complex<f32>; 2], b: &[Complex<f32>; 2]) -> [Complex<f32>; 2] {
        let va = WasmStoreF::load(a.as_slice());
        let vb = WasmStoreF::load(b.as_slice());
        let vc = va.mul_by_complex(vb.xor(WasmStoreF::conj_flags()));
        let mut out = [Complex::new(0.0f32, 0.0); 2];
        vc.write(out.as_mut_slice());
        out
    }

    fn wasm_mul_complex(a: &[Complex<f32>; 2], b: &[Complex<f32>; 2]) -> [Complex<f32>; 2] {
        let va = WasmStoreF::load(a.as_slice());
        let vb = WasmStoreF::load(b.as_slice());
        let vc = va.mul_by_complex(vb);
        let mut out = [Complex::new(0.0f32, 0.0); 2];
        vc.write(out.as_mut_slice());
        out
    }

    fn assert_approx(got: Complex<f32>, want: Complex<f32>, label: &str) {
        let eps = 1e-5_f32;
        assert!(
            (got.re - want.re).abs() < eps && (got.im - want.im).abs() < eps,
            "{label}: got ({}, {}i), want ({}, {}i)",
            got.re,
            got.im,
            want.re,
            want.im
        );
    }

    fn run_conj(a: [Complex<f32>; 2], b: [Complex<f32>; 2]) {
        let result = wasm_mul_conj(&a, &b);
        for i in 0..2 {
            assert_approx(
                result[i],
                ref_mul_conj(a[i], b[i]),
                &format!("conj lane {i}"),
            );
        }
    }

    fn run_mul(a: [Complex<f32>; 2], b: [Complex<f32>; 2]) {
        let result = wasm_mul_complex(&a, &b);
        for i in 0..2 {
            assert_approx(
                result[i],
                ref_mul_complex(a[i], b[i]),
                &format!("mul lane {i}"),
            );
        }
    }

    // --- mul_by_conj_b ---

    #[wasm_bindgen_test]
    fn test_conj_unit() {
        run_conj([Complex::new(1., 0.); 2], [Complex::new(1., 0.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_pure_imaginary() {
        // i * conj(i) = 1
        run_conj([Complex::new(0., 1.); 2], [Complex::new(0., 1.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_general() {
        // (3+4i) * conj(1+2i) = 11-2i
        run_conj([Complex::new(3., 4.); 2], [Complex::new(1., 2.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_mixed_lanes() {
        run_conj(
            [Complex::new(1., 2.), Complex::new(3., -1.)],
            [Complex::new(4., 1.), Complex::new(-1., 2.)],
        );
    }

    #[wasm_bindgen_test]
    fn test_conj_b_is_real() {
        run_conj([Complex::new(2., 3.); 2], [Complex::new(5., 0.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_a_is_real() {
        run_conj([Complex::new(4., 0.); 2], [Complex::new(1., 2.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_zero_a() {
        run_conj([Complex::new(0., 0.); 2], [Complex::new(3., 4.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_zero_b() {
        run_conj([Complex::new(3., 4.); 2], [Complex::new(0., 0.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_conj_negatives() {
        run_conj(
            [Complex::new(-1., -1.), Complex::new(-3., 2.)],
            [Complex::new(-2., 3.), Complex::new(4., -1.)],
        );
    }

    #[wasm_bindgen_test]
    fn test_conj_self_magnitude_squared() {
        // a * conj(a) = |a|² + 0i
        let a = Complex::new(3., 4.);
        let result = wasm_mul_conj(&[a; 2], &[a; 2]);
        for lane in &result {
            assert_approx(*lane, Complex::new(25., 0.), "self-conj");
        }
    }

    // --- mul_by_complex ---

    #[wasm_bindgen_test]
    fn test_mul_unit() {
        run_mul([Complex::new(1., 0.); 2], [Complex::new(1., 0.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_mul_pure_imaginary() {
        // i * i = -1
        run_mul([Complex::new(0., 1.); 2], [Complex::new(0., 1.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_mul_general() {
        // (3+4i) * (1+2i) = -5+10i
        run_mul([Complex::new(3., 4.); 2], [Complex::new(1., 2.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_mul_mixed_lanes() {
        run_mul(
            [Complex::new(1., 2.), Complex::new(-3., 1.)],
            [Complex::new(3., -1.), Complex::new(2., 4.)],
        );
    }

    #[wasm_bindgen_test]
    fn test_mul_zero() {
        run_mul([Complex::new(0., 0.); 2], [Complex::new(3., 4.); 2]);
        run_mul([Complex::new(3., 4.); 2], [Complex::new(0., 0.); 2]);
    }

    #[wasm_bindgen_test]
    fn test_mul_negatives() {
        run_mul(
            [Complex::new(-1., -1.), Complex::new(-3., 2.)],
            [Complex::new(-2., 3.), Complex::new(4., -1.)],
        );
    }

    #[wasm_bindgen_test]
    fn test_roundtrip() {
        // (a * b) * conj(b) = a * |b|²
        let a = [Complex::new(2., 3.); 2];
        let b = [Complex::new(1., 2.); 2];
        let ab = wasm_mul_complex(&a, &b);
        let back = wasm_mul_conj(&ab, &b);
        let want: [Complex<f32>; 2] = std::array::from_fn(|i| a[i] * b[i].norm_sqr());
        for i in 0..2 {
            assert_approx(back[i], want[i], &format!("roundtrip lane {i}"));
        }
    }
}
