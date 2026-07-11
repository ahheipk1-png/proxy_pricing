# Expanded Coverage Summary

Minimum required cases per option type: `100`.

| Option type | Cases executed | Status | Example subtests |
|---|---:|---|---|
| American | 108 | PASS | K=80.0, vol=0.15, r=0.01, S=60.0<br>K=80.0, vol=0.15, r=0.01, S=100.0<br>K=80.0, vol=0.15, r=0.01, S=160.0<br>K=80.0, vol=0.15, r=0.04, S=60.0<br>K=80.0, vol=0.15, r=0.04, S=100.0 |
| Asian | 216 | PASS | K=80.0, vol=0.15, day=0, S=70.0, avg=90.0<br>K=80.0, vol=0.15, day=0, S=100.0, avg=100.0<br>K=80.0, vol=0.15, day=0, S=140.0, avg=115.0<br>K=80.0, vol=0.15, day=3, S=70.0, avg=90.0<br>K=80.0, vol=0.15, day=3, S=100.0, avg=100.0 |
| Autocallable | 125 | PASS | base, obs=0, S=45.0<br>base, obs=0, S=70.0<br>base, obs=0, S=90.0<br>base, obs=0, S=105.0<br>base, obs=0, S=140.0 |
| Barrier | 360 | PASS | down_out, month=0, vol=0.15, S=76.00<br>down_out, month=0, vol=0.15, S=90.71<br>down_out, month=0, vol=0.15, S=105.43<br>down_out, month=0, vol=0.15, S=120.14<br>down_out, month=0, vol=0.15, S=134.86 |
| Basket Asian | 120 | PASS | day=0, scale=0.7, avg=80.0<br>day=0, scale=0.7, avg=95.0<br>day=0, scale=0.7, avg=105.0<br>day=0, scale=0.7, avg=120.0<br>day=0, scale=0.7, avg=140.0 |
| Basket cliquet | 120 | PASS | basket_return, month=0, returns=[[0.1, -0.04, 0.01]]<br>basket_return, month=0, returns=[[-0.08, -0.02, 0.05]]<br>basket_return, month=0, returns=[[0.0, 0.02, 0.03]]<br>basket_return, month=3, returns=[[0.1, -0.04, 0.01]]<br>basket_return, month=3, returns=[[-0.08, -0.02, 0.05]] |
| Bermudan | 108 | PASS | K=80.0, vol=0.15, ex=0, S=60.0<br>K=80.0, vol=0.15, ex=0, S=100.0<br>K=80.0, vol=0.15, ex=0, S=160.0<br>K=80.0, vol=0.15, ex=4, S=60.0<br>K=80.0, vol=0.15, ex=4, S=100.0 |
| Cliquet | 120 | PASS | cap=0.03, day=0, accrued=-0.0500<br>cap=0.03, day=0, accrued=-0.0357<br>cap=0.03, day=0, accrued=-0.0214<br>cap=0.03, day=0, accrued=-0.0071<br>cap=0.03, day=0, accrued=0.0071 |
| European | 144 | PASS | K=80.0, vol=0.1, T=0.5, S=60.0<br>K=80.0, vol=0.1, T=0.5, S=90.0<br>K=80.0, vol=0.1, T=0.5, S=110.0<br>K=80.0, vol=0.1, T=0.5, S=160.0<br>K=80.0, vol=0.1, T=1.0, S=60.0 |
| Random payoff | 288 | PASS | base_market, random_piecewise_1, t=0.0, S=55.0<br>base_market, random_piecewise_1, t=0.0, S=100.0<br>base_market, random_piecewise_1, t=0.0, S=170.0<br>base_market, random_piecewise_1, t=0.25, S=55.0<br>base_market, random_piecewise_1, t=0.25, S=100.0 |
| SLV cliquet | 125 | PASS | month=0, S=50.0, v=0.01<br>month=0, S=50.0, v=0.03<br>month=0, S=50.0, v=0.06<br>month=0, S=50.0, v=0.1<br>month=0, S=50.0, v=0.16 |
