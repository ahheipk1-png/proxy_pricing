# Three-underlying SLV basket cliquet experiment

The strongest fixed generic baseline is `sparse_chebyshev` trained on 2,001
low-discrepancy states. Results vary across independent state-space designs:

| Variant | Development | Validation 2 | Validation 3 | Worst |
|---|---:|---:|---:|---:|
| `basket_return` | 10.258% | 7.108% | 18.500% | 18.500% |
| `average_clipped` | 5.624% | 11.566% | 5.745% | 11.566% |
| `worst_of` | 9.700% | 9.862% | 6.774% | 9.862% |

The development-only adaptive blend is retained in the CSV for research
comparison, but it is not the recommended default because it did not generalize.
