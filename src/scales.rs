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
use crate::ScaletError;
use crate::err::try_vec;
use crate::sample::CwtSample;
use crate::scale_bounds::linspace;
use num_traits::AsPrimitive;

pub(crate) fn log_piecewise_scales<T: CwtSample>(
    min_scale: T,
    max_scale: T,
    nv: T,
) -> Result<Vec<T>, ScaletError>
where
    usize: AsPrimitive<T>,
    isize: AsPrimitive<T>,
{
    // 1. Calculate na (number of scales needed)
    // na = int(ceil(nv * log2(max_scale / min_scale)))
    let na: isize = (nv * (max_scale / min_scale).log2()).ceil().as_();

    // 2. Calculate mn_pow and mx_pow (min and max exponents for the 2^ distribution)
    // mn_pow = int(floor(nv * log2(min_scale)))
    let mn_pow: isize = (nv * min_scale.log2()).floor().as_();

    // mx_pow = mn_pow + na
    let mx_pow = mn_pow + na;

    // 3. Generate the base log-spaced scales (2 ** (arange(mn_pow, mx_pow) / nv))
    // let base_scales: Vec<T> = (mn_pow..mx_pow).map(|p| (p.as_() / nv).exp2()).collect();
    // base_scales

    // 1. Initialize an empty mutable vector to hold the scales.
    if mx_pow < mn_pow {
        panic!("Something went wrong and impossible condition has occur");
    }
    let mut base_scales = try_vec![T::zero(); (mx_pow - mn_pow) as usize];

    // 2. Iterate through the range of powers.
    // The range (mn_pow..mx_pow) is exclusive of mx_pow, matching the map iterator.
    for (i, dst) in base_scales.iter_mut().enumerate() {
        let p = i as isize + mn_pow;
        // 3. Calculate the exponent argument: (p as f64) / nv
        let p_f64 = p.as_();
        let exponent = p_f64 / nv;

        let scale_value = exponent.exp2();

        *dst = scale_value;
    }

    // 6. Return the populated vector.
    Ok(base_scales)
}

pub(crate) fn linear_scales<T: CwtSample>(
    min_scale: T,
    max_scale: T,
    nv: T,
) -> Result<Vec<T>, ScaletError>
where
    usize: AsPrimitive<T>,
    isize: AsPrimitive<T>,
{
    // 1. Calculate na (number of scales needed)
    // na = int(ceil(nv * log2(max_scale / min_scale)))
    let na: isize = (nv * (max_scale / min_scale).log2()).ceil().as_();

    // 2. Calculate mn_pow and mx_pow (min and max exponents for the 2^ distribution)
    // mn_pow = int(floor(nv * log2(min_scale)))
    let mn_pow: isize = (nv * min_scale.log2()).floor().as_();

    // mx_pow = mn_pow + na
    let mx_pow = mn_pow + na;

    // mx_pow = mn_pow + na
    let min_scale = (mn_pow.as_() / nv).exp2();
    let max_scale = (mx_pow.as_() / nv).exp2();
    let na: usize = (max_scale / min_scale).ceil().as_();
    linspace(min_scale, max_scale, na)
}
