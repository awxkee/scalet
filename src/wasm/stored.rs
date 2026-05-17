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
pub(crate) struct WasmStoreD {
    pub(crate) v: v128,
}

impl WasmStoreD {
    #[inline]
    pub(crate) fn raw(v: v128) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn conj_flags() -> Self {
        // [0.0, -0.0] — xor with this to conjugate im lane
        Self {
            v: f64x2(0.0, -0.0),
        }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn dup(v: f64) -> Self {
        Self { v: f64x2_splat(v) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load(data: &[Complex<f64>]) -> Self {
        unsafe { Self::raw(v128_load(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load1(data: &Complex<f64>) -> Self {
        unsafe { Self::raw(v128_load(data as *const Complex<f64> as *const v128)) }
    }

    // a * b
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        // [-ai, ar]
        let neg_im = f64x2_neg(self.v); // [-ar, -ai]
        let temp = i64x2_shuffle::<1, 2>(neg_im, self.v); // [-ai, ar]

        let br = i64x2_shuffle::<0, 0>(other.v, other.v);
        let bi = i64x2_shuffle::<1, 1>(other.v, other.v);
        let sum = f64x2_mul(self.v, br);
        Self::raw(f64x2_add(sum, f64x2_mul(temp, bi)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self::raw(v128_xor(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write(&self, data: &mut [Complex<f64>]) {
        unsafe { v128_store(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write1(&self, data: &mut Complex<f64>) {
        unsafe { v128_store(data as *mut Complex<f64> as *mut v128, self.v) }
    }
}

impl Mul<WasmStoreD> for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: WasmStoreD) -> Self::Output {
        Self::raw(f64x2_mul(self.v, rhs.v))
    }
}

#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::*;
    use num_complex::Complex;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn ref_mul_conj(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
        a * b.conj()
    }

    fn ref_mul_complex(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
        a * b
    }

    fn wasm_d_mul_conj(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
        let va = WasmStoreD::load1(&a);
        let vb = WasmStoreD::load1(&b);
        let vc = va.mul_by_complex(vb.xor(WasmStoreD::conj_flags()));
        let mut out = Complex::new(0.0f64, 0.0);
        vc.write1(&mut out);
        out
    }

    fn wasm_d_mul_complex(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
        let va = WasmStoreD::load1(&a);
        let vb = WasmStoreD::load1(&b);
        let vc = va.mul_by_complex(vb);
        let mut out = Complex::new(0.0f64, 0.0);
        vc.write1(&mut out);
        out
    }

    fn assert_approx(got: Complex<f64>, want: Complex<f64>, label: &str) {
        let eps = 1e-12_f64;
        assert!(
            (got.re - want.re).abs() < eps && (got.im - want.im).abs() < eps,
            "{label}: got ({}, {}i), want ({}, {}i)",
            got.re,
            got.im,
            want.re,
            want.im
        );
    }

    fn run_conj(a: Complex<f64>, b: Complex<f64>) {
        let got = wasm_d_mul_conj(a, b);
        assert_approx(got, ref_mul_conj(a, b), &format!("({a}) * conj({b})"));
    }

    fn run_mul(a: Complex<f64>, b: Complex<f64>) {
        let got = wasm_d_mul_complex(a, b);
        assert_approx(got, ref_mul_complex(a, b), &format!("({a}) * ({b})"));
    }

    // --- mul_by_conj_b ---

    #[wasm_bindgen_test]
    fn test_conj_unit() {
        run_conj(Complex::new(1., 0.), Complex::new(1., 0.));
    }

    #[wasm_bindgen_test]
    fn test_conj_pure_imaginary() {
        // i * conj(i) = 1
        run_conj(Complex::new(0., 1.), Complex::new(0., 1.));
    }

    #[wasm_bindgen_test]
    fn test_conj_general() {
        // (3+4i) * conj(1+2i) = 11-2i
        run_conj(Complex::new(3., 4.), Complex::new(1., 2.));
    }

    #[wasm_bindgen_test]
    fn test_conj_b_is_real() {
        run_conj(Complex::new(2., 3.), Complex::new(5., 0.));
    }

    #[wasm_bindgen_test]
    fn test_conj_a_is_real() {
        run_conj(Complex::new(4., 0.), Complex::new(1., 2.));
    }

    #[wasm_bindgen_test]
    fn test_conj_zero_a() {
        run_conj(Complex::new(0., 0.), Complex::new(3., 4.));
    }

    #[wasm_bindgen_test]
    fn test_conj_zero_b() {
        run_conj(Complex::new(3., 4.), Complex::new(0., 0.));
    }

    #[wasm_bindgen_test]
    fn test_conj_negatives() {
        run_conj(Complex::new(-1., -1.), Complex::new(-2., 3.));
        run_conj(Complex::new(-3., 2.), Complex::new(4., -1.));
    }

    #[wasm_bindgen_test]
    fn test_conj_self_magnitude_squared() {
        // a * conj(a) = |a|² + 0i
        let a = Complex::new(3., 4.);
        let got = wasm_d_mul_conj(a, a);
        assert_approx(got, Complex::new(25., 0.), "self-conj");
    }

    #[wasm_bindgen_test]
    fn test_conj_large() {
        run_conj(
            Complex::new(1e10_f64, -1e10_f64),
            Complex::new(1e10_f64, 1e10_f64),
        );
    }

    #[wasm_bindgen_test]
    fn test_conj_small() {
        run_conj(
            Complex::new(1e-10_f64, -1e-10_f64),
            Complex::new(1e-10_f64, 1e-10_f64),
        );
    }

    // --- mul_by_complex ---

    #[wasm_bindgen_test]
    fn test_mul_unit() {
        run_mul(Complex::new(1., 0.), Complex::new(1., 0.));
    }

    #[wasm_bindgen_test]
    fn test_mul_pure_imaginary() {
        // i * i = -1
        run_mul(Complex::new(0., 1.), Complex::new(0., 1.));
    }

    #[wasm_bindgen_test]
    fn test_mul_general() {
        // (3+4i) * (1+2i) = -5+10i
        run_mul(Complex::new(3., 4.), Complex::new(1., 2.));
    }

    #[wasm_bindgen_test]
    fn test_mul_negatives() {
        run_mul(Complex::new(-1., -1.), Complex::new(-2., 3.));
    }

    #[wasm_bindgen_test]
    fn test_mul_zero() {
        run_mul(Complex::new(0., 0.), Complex::new(3., 4.));
        run_mul(Complex::new(3., 4.), Complex::new(0., 0.));
    }

    #[wasm_bindgen_test]
    fn test_roundtrip() {
        // (a * b) * conj(b) = a * |b|²
        let a = Complex::new(2., 3.);
        let b = Complex::new(1., 2.);
        let ab = wasm_d_mul_complex(a, b);
        let back = wasm_d_mul_conj(ab, b);
        let want = a * b.norm_sqr();
        assert_approx(back, want, "roundtrip");
    }
}
