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
use crate::err::try_vec;
use crate::mla::fmla;
use crate::sample::CwtSample;
use crate::{CwtWavelet, ScaletError};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) fn linspace<T: CwtSample>(
    start: T,
    end: T,
    samples: usize,
) -> Result<Vec<T>, ScaletError>
where
    usize: AsPrimitive<T>,
{
    if samples == 0 {
        return Ok(Vec::new());
    }

    // Handle the trivial case where only one sample is requested.
    if samples == 1 {
        return Ok(vec![start]);
    }

    // Calculate the step size (delta)
    // NumPy calculates the step size as (end - start) / (samples - 1)
    let delta = (end - start) / ((samples - 1).as_());

    // Generate the vector by iterating and adding the step
    let mut result = try_vec![T::zero(); samples];

    for (i, dst) in result.iter_mut().enumerate() {
        let value = fmla(i.as_(), delta, start);

        // Use a slight correction for the last element to ensure it precisely equals 'end',
        if i == samples - 1 {
            *dst = end;
        } else {
            *dst = value;
        }
    }

    Ok(result)
}

fn linspace_exclusive<T: CwtSample>(start: T, end: T, samples: usize) -> Result<Vec<T>, ScaletError>
where
    usize: AsPrimitive<T>,
{
    if samples == 0 {
        return Ok(Vec::new());
    }

    let step = (end - start) / samples.as_();
    let mut result = try_vec![T::zero(); samples];

    for (i, dst) in result.iter_mut().enumerate() {
        *dst = fmla(i.as_(), step, start);
    }

    Ok(result)
}

fn find_maximum<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    step_size: T,
    steps_per_search: usize,
    step_start: T,
    step_limit: T,
    min_value: T,
) -> Result<(T, T), ScaletError>
where
    usize: AsPrimitive<T>,
{
    let increment = steps_per_search.as_() * step_size;

    let mut largest_max = min_value;
    let mut input_value = step_start;

    let mut search_idx = 0usize;

    loop {
        let start = step_start + increment * search_idx.as_();
        let end = start + increment;

        // linspace(start, end, steps_per_search, endpoint=False)
        let input_values = linspace_exclusive(start, end, steps_per_search)?;

        // output_values = abs(fn(input_values))
        let output_values = wavelet
            .make_wavelet(&input_values)?
            .into_iter()
            .map(|v| fmla(v.re, v.re, v.im * v.im).sqrt())
            .collect::<Vec<_>>();

        if input_values.len() != output_values.len() {
            return Err(ScaletError::WaveletInvalidSize(
                input_values.len(),
                output_values.len(),
            ));
        }

        // find max and argmax
        let mut output_max = T::NEG_INFINITY;
        let mut argmax = 0usize;
        for (i, &v) in output_values.iter().enumerate() {
            if v > output_max {
                output_max = v;
                argmax = i;
            }
        }

        if output_max > largest_max {
            largest_max = output_max;
            input_value = input_values[argmax];
        } else if output_max < largest_max {
            break;
        }

        search_idx += 1;

        if *input_values.last().unwrap_or(&T::default()) > step_limit {
            return Err(ScaletError::Generic(format!(
                "could not find function maximum with given \
(step_size, steps_per_search, step_start, step_limit, min_value)=({}, {}, {}, {}, {})",
                step_size, steps_per_search, step_start, step_limit, min_value
            )));
        }
    }

    Ok((input_value, largest_max))
}

fn find_first_occurrence<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    value: T,
    step_size: T,
    steps_per_search: usize,
    step_start: T,
    step_limit: T,
) -> Result<(T, T), ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let increment = steps_per_search.as_() * step_size;

    let mut search_idx = 0usize;
    let mut step_limit_exceeded = false;

    let mut input_values: Vec<T>;
    let mut output_values: Vec<T>;

    let idx;

    loop {
        let start = fmla(increment, search_idx.as_(), step_start);
        let end = start + increment;

        input_values = linspace_exclusive(start, end, steps_per_search)?;

        // clip to step_limit if exceeded
        if *input_values.last().unwrap_or(&T::default()) > step_limit {
            step_limit_exceeded = true;
            for v in &mut input_values {
                *v = v.min(step_limit);
            }
        }

        // output_values = abs(fn(input_values))
        output_values = wavelet
            .make_wavelet(&input_values)?
            .into_iter()
            .map(|v| fmla(v.re, v.re, v.im * v.im).sqrt())
            .collect();

        if input_values.len() != output_values.len() {
            return Err(ScaletError::WaveletInvalidSize(
                input_values.len(),
                output_values.len(),
            ));
        }

        // mxdiff = max(abs(diff(output_values)))
        let mut mxdiff: T = 0.0f64.as_();
        for w in output_values.windows(2) {
            let d = (w[1] - w[0]).abs();
            mxdiff = mxdiff.max(d);
        }

        // check if any |output - value| <= mxdiff
        let mut found = false;
        let mut best_idx = 0usize;
        let mut best_err = T::INFINITY;

        for (i, &v) in output_values.iter().enumerate() {
            let err = (v - value).abs();
            if err <= mxdiff && err < best_err {
                best_err = err;
                best_idx = i;
                found = true;
            }
        }

        if found {
            idx = best_idx;
            break;
        }

        search_idx += 1;

        if step_limit_exceeded {
            return Err(ScaletError::Generic(format!(
                "could not find input value to yield function output value={} \
with given (step_size, steps_per_search, step_start, step_limit)=({}, {}, {}, {})",
                value, step_size, steps_per_search, step_start, step_limit
            )));
        }
    }

    Ok((input_values[idx], output_values[idx]))
}

pub(crate) fn find_min_scale<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    cutoff: T,
) -> Result<T, ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let (w_peak, peak) = find_maximum(
        wavelet.clone(),
        1e-3f64.as_(),
        10000,
        T::zero(),
        1000f64.as_(),
        -1f64.as_(),
    )?;
    let (step_start, step_limit) = if cutoff > T::zero() {
        (w_peak, 10f64.as_() * w_peak)
    } else {
        (T::zero(), w_peak)
    };
    let (w_cutoff, _) = find_first_occurrence(
        wavelet.clone(),
        cutoff.abs() * peak,
        1e-3f64.as_(),
        10000,
        step_start,
        step_limit,
    )?;
    let min_scale = w_cutoff * T::FRAC_1_PI;
    Ok(min_scale)
}

fn find_wrap_index<T: CwtSample>(n_divs: &[T]) -> Option<usize>
where
    f64: AsPrimitive<T>,
{
    n_divs
        .iter()
        .map(|x| x.fract()) // n_divs % 1
        .collect::<Vec<_>>()
        .windows(2) // diff
        .enumerate()
        .find(|(_, w)| w[1] - w[0] < -0.8f64.as_())
        .map(|(i, _)| i)
}

pub(crate) fn find_max_scale<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    n: usize,
    min_cutoff: T,
    max_cutoff: T,
) -> Result<T, ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    if max_cutoff <= T::zero() || min_cutoff <= T::zero() {
        return Err(ScaletError::Generic(format!(
            "`max_cutoff` and `min_cutoff` must be positive (got {}, {})",
            max_cutoff, min_cutoff
        )));
    } else if max_cutoff <= min_cutoff {
        return Err(ScaletError::Generic(format!(
            "must have `max_cutoff > min_cutoff` (got {}, {})",
            max_cutoff, min_cutoff
        )));
    }

    let (w_peak, peak) = find_maximum(
        wavelet.clone(),
        1e-3f64.as_(),
        10000,
        T::zero(),
        1000f64.as_(),
        -1f64.as_(),
    )?;

    // we solve the inverse problem; instead of looking for spacing of xi
    // that'd land symmetrically about psih's peak, we pick such points
    // above a set ratio of peak's value and ensure they divide the line
    // from left symmetry point to zero an integer number of times

    // define all points of wavelet from cutoff to peak, left half
    let (w_cutoff, _) = find_first_occurrence(
        wavelet.clone(),
        min_cutoff * peak,
        1e-3f64.as_(),
        10000,
        T::zero(),
        w_peak,
    )?;

    let step = 1.0f64.as_() / n.as_();

    let mut w_ltp = Vec::new();
    let mut v = w_cutoff;

    while v < w_peak {
        w_ltp.push(v);
        v += step;
    }

    // consider every point on wavelet(w_ltp) (except peak) as candidate cutoff
    // point, and pick the earliest one that yields integer number of increments
    // from left point of symmetry to origin
    let all_but_last = &w_ltp[..w_ltp.len() - 1];
    let div_size = all_but_last
        .iter()
        .map(|&x| (w_peak - x) * 2f64.as_())
        .collect::<Vec<T>>(); // doubled so peak is skipped
    let n_divs = all_but_last
        .iter()
        .zip(div_size.iter())
        .map(|(&a, &b)| a / b)
        .collect::<Vec<T>>();
    // diff of modulus; first drop in n_divs is like [.98, .99, 0, .01], so at 0
    // we've hit an integer, and n_divs grows ~linearly so behavior guaranteed
    // -.8 arbitrary to be ~1 but <1
    let idx = find_wrap_index(&n_divs).ok_or(ScaletError::Generic(
        "Failed to find sufficiently-integer xi divisions; try widening (min_cutoff, max_cutoff)"
            .to_string(),
    ))?;
    // the div to base the scale on (angular bin spacing of scale*xi)
    let div_scale = div_size[idx + 1];

    // div size of scale=1 (spacing between angular bins at scale=1)
    let w_1div = T::PI / (n / 2).as_();

    let max_scale = div_scale / w_1div;
    Ok(max_scale)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MinMaxScale<T> {
    pub(crate) min: T,
    pub(crate) max: T,
}

pub(crate) fn find_min_max_scales<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    cutoff: T,
) -> Result<MinMaxScale<T>, ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let min_scale = find_min_scale(wavelet.clone(), cutoff)?;
    const M: usize = 4096;
    let max_scale = find_max_scale(wavelet.clone(), M, 0.6f64.as_(), 0.8f64.as_())?;
    Ok(MinMaxScale {
        min: min_scale,
        max: max_scale,
    })
}
