# Proxy Result Audit Rerun

Date: 2026-07-14

## Why This Rerun Was Needed

The old broad generic exotic-family results were too optimistic. The issue was not in the payoff evaluator itself; it was in the validation design:

- `ExoticPipelineCore.run_proxy_study` reused the same simulated path-ratio array for training and validation.
- `ExoticPipelineCore.run_family_proxy_study` also reused the same path-ratio array at shifted scale states.
- That setup measures interpolation error on a common MC noise surface. It is useful diagnostically, but it is not a true out-of-sample benchmark.

Two implementation fixes were made during this audit:

- Generic exotic studies now use independent low-discrepancy antithetic path streams for training labels and benchmark labels.
- `AmericanMain.py` now clips PCHIP continuation queries to the training spot range during backward induction. The previous standalone script allowed extrapolation and could explode.
- `BasketCliquetMain.py` now uses benchmark-grade Sobol paths for the hard `sobol_mc_proxy` fallback, reducing proxy-path sampling noise in order-statistic basket cliquets.

## Trusted Product-Specific Rerun Results

These are the best current results from scripts with exact, tree, finite-difference, or independent MC benchmarks.

| Option Type | Script | Benchmark Type | Training / Benchmark Size | Worst Max Relative Error | Notes |
|---|---|---|---|---:|---|
| European | `EuroMain.py` | Black-Scholes closed form | 121 states, 32,768 shifted Sobol MC paths/state for labels | 0.004% | Benchmark is exact. |
| European with vol/rate features | `EuroMain_vol_rate.py` | exact term-structure Black-Scholes | 6,885 train states, 3,969 test states/option | 0.565% | Best collapsed effective-curve degree 5 model. |
| American put | `AmericanMain.py` | projected finite difference | 12,390,400 training transitions, 4,000 time x 2,000 spot FD grid | 2.103% | Fixed standalone PCHIP extrapolation bug. |
| Bermudan put | `BermudanMain.py` | independent CRR tree | 8,192 paths/state | 8.300% | Tree benchmark, exercise on monthly dates. |
| Asian arithmetic call | `AsianMain.py` | independent shifted Sobol MC + geometric control | 10M target scenarios/fit, 524,288 benchmark paths/state | 1.739% | Monthly fixings over one year. |
| Cliquet | `CliquetMain.py` | independent antithetic MC + clipped-sum control | 10M target scenarios/fit, 524,288 benchmark paths/state | 3.286% | Late reset month remains hardest. |
| Barrier | `BarrierMain.py` | independent Sobol MC + Brownian-bridge continuous adjustment | 10M target scenarios/fit, 524,288 benchmark paths/state | 9.351% | Up-and-out variants are hardest. |
| SLV cliquet | `SLVCliquetMain.py` | independent Sobol MC under SLV | 10M target scenarios/fit, 524,288 benchmark paths/state | 5.998% | Month 9 is hardest. |
| Basket Asian | `BasketAsianMain.py` | independent Sobol MC | 20M target scenarios/fit, 524,288 benchmark paths/state | 5.018% | PCA/state enrichment still works. |
| Basket cliquet | `BasketCliquetMain.py` | independent Sobol MC under 3-asset SLV | 20M target scenarios/fit, 524,288 benchmark paths/state | 4.053% | Improved by using 524,288 proxy Sobol paths for hard fallback. |
| Autocallable | `AutocallableMain.py` | independent Sobol MC | 16,384 train paths/state, 65,536 benchmark paths/state | 0.123% | Akima best on this grid. |
| Random payoff | `RandomOptionMain.py` | independent Sobol MC | 16,384 train paths/state, 65,536 benchmark paths/state | 0.658% | Akima best; Chebyshev is not robust for random piecewise payoffs. |

## Broad Generic Family Reruns

These are intentionally broad 100 single + 100 basket configuration sweeps. After the independent-path fix, several families no longer pass the 12% max-error target. These results should be treated as research diagnostics, not production-ready proxies.

| Family | Script | Path Count | Single Worst Max | Basket Worst Max | Audit Status |
|---|---|---:|---:|---:|---|
| Rainbow | `RainbowMain.py` | 16,384 | 23.467% | 28.638% | REVIEW |
| YieldSeeker | `YieldSeekerMain.py` | 16,384 | 0.229% | 0.559% | PASS |
| Himalayan | `HimalayanMain.py` | 16,384 | 21.582% | 52.732% | REVIEW |
| Lookback | `LookbackMain.py` | 16,384 | 27.550% | 29.169% | REVIEW |
| Barrier family | `BarrierFamilyMain.py` | 32,768 | 43.243% | 70.370% | REVIEW |
| Binary | `BinaryMain.py` | 131,072 | 31.090% | 39.927% | REVIEW |

## Aggregate Generic Exotic Reruns

| Study | Script | Cases | Worst Family Result | Audit Status |
|---|---|---:|---:|---|
| SingleExotic | `SingleExoticMain.py` | 106 | Binary 22.094% | REVIEW |
| BasketExotic | `BasketExoticMain.py` | 120 | Binary 56.590% | REVIEW |

## Coverage Tests

`ProxyCorrectnessTests.py` passed all 11 tests.

`ProxyExpandedCoverageTests.py` passed the minimum 100-case coverage target for every product type:

| Option Type | Cases |
|---|---:|
| American | 108 |
| Asian | 216 |
| Autocallable | 125 |
| Barrier | 360 |
| Basket Asian | 120 |
| Basket cliquet | 120 |
| Bermudan | 108 |
| Cliquet | 120 |
| European | 144 |
| Random payoff | 288 |
| SLV cliquet | 125 |

## Audit Conclusion

The product-specific scripts are generally credible after this rerun because their benchmarks are exact, tree/finite-difference, or independent high-path MC. The broad generic one-dimensional family sweeps were the main source of "too good to be true" numbers. Once the common-path validation leak was removed, discontinuous and order-statistic products became much harder, especially binary, barrier-family, Himalayan, Rainbow, and Lookback variants.

The current recommendation is:

- Use the product-specific proxy methods for European, Asian, cliquet, barrier, American, Bermudan, autocallable, basket Asian, SLV cliquet, and basket cliquet.
- Do not present the broad generic family sweep as production accuracy yet.
- For discontinuous payoffs, add event-distance features, smoothing/classification layers, or direct conditional MC fallback near discontinuity surfaces.
- Keep independent benchmark streams in every future result table unless the result is explicitly labeled as a common-random diagnostic.
