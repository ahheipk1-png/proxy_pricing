# Barrier option proxy experiment

| Variant | PCHIP worst max | Best method | Best worst max |
|---|---:|---|---:|
| `down_out_discrete` | 0.770% | `chebyshev` | 0.748% |
| `down_out_continuous` | 0.638% | `pchip` | 0.638% |
| `up_out_discrete` | 9.351% | `pchip` | 9.351% |
| `up_out_continuous` | 9.271% | `akima` | 9.270% |
| `double_out_discrete` | 1.230% | `bernstein` | 1.167% |
| `double_out_continuous` | 1.800% | `akima` | 1.701% |
