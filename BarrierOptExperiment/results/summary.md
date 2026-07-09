# Barrier option proxy experiment

| Variant | PCHIP worst max | Best method | Best worst max |
|---|---:|---|---:|
| `down_out_discrete` | 2.253% | `bernstein` | 2.217% |
| `down_out_continuous` | 2.531% | `akima` | 2.518% |
| `up_out_discrete` | 9.096% | `pchip` | 9.096% |
| `up_out_continuous` | 8.808% | `pchip` | 8.808% |
| `double_out_discrete` | 8.387% | `bernstein` | 8.301% |
| `double_out_continuous` | 8.148% | `akima` | 8.133% |
