# One-Dimensional Fitting Findings

The same one-dimensional labels were fitted with Chebyshev regression, PCHIP,
Akima, global Bernstein/Bézier regression, and piecewise cubic Bézier curves.

## Results

| Product | Best method | Worst max relative error |
|---|---|---:|
| European call | Global Bernstein/Bézier degree 12 | 0.536% |
| Arithmetic Asian call after state reduction | Akima | 0.186% |

European values are globally smooth, so global regularized regression averages
Monte Carlo noise better than interpolating it. Chebyshev remains strong at
0.854%; PCHIP and Akima are around 2.5%.

The Asian adjusted-moneyness labels have more local slope variation. Akima
handles this especially well and reduces the worst error from 3.286% for the
degree-19 Chebyshev comparison to 0.186%.

An independent-seed rerun confirmed the result: Akima's worst error was 0.144%,
versus 2.577% for Chebyshev and 1.643% for PCHIP.

PCHIP is viable for both products, but it is not universally best. It is most
valuable when shape preservation or a moving kink matters, as in the American
put continuation problem.

## Bézier equivalence

A cubic Hermite segment with endpoint values `y_i`, `y_(i+1)` and endpoint
derivatives `d_i`, `d_(i+1)` is exactly the cubic Bézier curve with control
points:

```text
B0 = y_i
B1 = y_i + h d_i / 3
B2 = y_(i+1) - h d_(i+1) / 3
B3 = y_(i+1)
```

Therefore, a piecewise Bézier curve using PCHIP slopes is numerically identical
to PCHIP. Bézier is a representation, not a new smoothing rule, unless the
control points are chosen by a different fitting or regularization criterion.

## Recommendation

- Universal one-dimensional operational default: PCHIP.
- Smooth one-dimensional European-style surface: Chebyshev or Bernstein ridge.
- Reduced Asian coordinate: Akima, with Chebyshev retained as a smooth fallback.
- Early-exercise continuation with a kink: PCHIP.
- Noisy sparse labels: prefer a regularized global method over exact local
  interpolation unless the local method is validated independently.

The expanded 99-case study in `expanded_spline_study.md` supersedes this narrow
two-product ranking. Natural cubic interpolation led the balanced average by
only 0.035 percentage points, while PCHIP had zero local overshoot and won 53
of 99 head-to-head p99 comparisons. PCHIP therefore remains the universal
one-dimensional operational default.
