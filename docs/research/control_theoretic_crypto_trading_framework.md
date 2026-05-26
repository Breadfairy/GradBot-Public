# Control-Theoretic Crypto Trading Framework

## 1. Objective

Build a cryptocurrency trading system using a layered control-theory
structure.

The system has three interacting layers:

1. **Inner loop:** price-vs-moving-average PID controller
2. **Outer loop:** benchmark-relative supervisory controller
3. **Risk loop:** drawdown and exposure limiter

The inner loop generates a local trade signal from price behaviour.

The outer loop monitors whether the strategy is gaining or losing ground
relative to a moving benchmark.

The risk loop limits capital loss, excessive exposure, turnover, and unstable
behaviour.

The goal is not simply to make absolute profit. The goal is to generate
positive benchmark-relative performance while keeping downside risk bounded.

---

## 2. Core Idea

The benchmark is treated as a moving reference trajectory.

The strategy is not only judged by whether it makes money. It is judged by
whether it is outperforming the benchmark.

For example:

```text
strategy return  = +8%
benchmark return = +20%
```

The strategy made money, but it underperformed. Therefore, the supervisory
controller should detect that the strategy is losing ground and apply
adaptive correction.

The core principle is:

```text
price error decides direction
alpha error decides correction
drawdown decides survival
```

---

## 3. High-Level System Structure

```text
market data
    |
    v
price-vs-MA error calculation
    |
    v
inner PID controller
    |
    v
raw trade signal
    |
    v
outer benchmark-relative supervisor
    |
    v
risk limiter
    |
    v
final exposure / trade action
```

The inner loop answers:

```text
What is price doing relative to its trend?
```

The outer loop answers:

```text
Is the strategy gaining or losing ground relative to the benchmark?
```

The risk loop answers:

```text
Is the system still operating within acceptable risk limits?
```

---

## 4. Main Variables

### 4.1 Market Variables

Let:

$$
p_k = \text{close price at candle } k
$$

$$
m_k = \text{moving average at candle } k
$$

$$
r_k = \frac{p_k - p_{k-1}}{p_{k-1}}
$$

where:

```text
p_k = current candle close
m_k = current moving average
r_k = simple return at candle k
```

---

### 4.2 Strategy Variables

Let:

$$
V_{s,k} = \text{strategy equity at candle } k
$$

$$
V_{b,k} = \text{benchmark equity at candle } k
$$

$$
x_k = \text{strategy exposure at candle } k
$$

For spot-only crypto trading:

$$
0 \le x_k \le 1
$$

where:

```text
x_k = 0 means fully in cash
x_k = 1 means fully long
```

For long-short systems:

$$
-1 \le x_k \le 1
$$

where:

```text
x_k = -1 means fully short
x_k =  0 means flat / cash
x_k =  1 means fully long
```

For the first implementation, use the spot-only constraint:

$$
0 \le x_k \le 1
$$

---

## 5. Inner Loop: Price-vs-MA PID Controller

The inner loop generates the local trade signal.

It uses the relationship between price and moving average as the control
error.

---

### 5.1 Price-vs-MA Error

Use normalised price-vs-moving-average error:

$$
e_{p,k} = \frac{p_k - m_k}{m_k}
$$

where:

```text
e_p,k > 0 means price is above the moving average
e_p,k < 0 means price is below the moving average
```

This avoids raw price-scale dependency.

For example, a price error of:

$$
e_{p,k}=0.04
$$

means price is 4 percent above its moving average.

---

### 5.2 Proportional Term

The proportional term is the current price-vs-MA error:

$$
P_k = e_{p,k}
$$

Interpretation:

```text
The proportional term measures current distance from trend.
```

A large positive value means price is strongly above the moving average.

A large negative value means price is strongly below the moving average.

---

### 5.3 Integral Term

A normal integral can wind up badly in trading. Therefore use a leaky integral:

$$
I_k = \lambda I_{k-1} + e_{p,k}
$$

where:

$$
0 < \lambda < 1
$$

and:

```text
lambda = integralDecay
```

Interpretation:

```text
The integral term measures accumulated trend pressure.
```

If price has remained above the moving average for many candles, the integral
term increases.

If price crosses back toward the moving average, the leaky integral gradually
forgets old information.

The leaky integral helps avoid uncontrolled integral windup.

---

### 5.4 Derivative Term

The derivative term is:

$$
D_k = e_{p,k} - e_{p,k-1}
$$

Interpretation:

```text
The derivative term measures rate of separation from the moving average.
```

If:

$$
D_k > 0
$$

price is moving further above the moving average.

If:

$$
D_k < 0
$$

price is falling back toward or below the moving average.

---

### 5.5 Inner PID Signal

The raw inner-loop control signal is:

$$
u_{p,k} = K_p P_k + K_i I_k + K_d D_k
$$

Substituting the terms:

$$
u_{p,k}
=
K_p e_{p,k}
+
K_i I_k
+
K_d(e_{p,k}-e_{p,k-1})
$$

where:

```text
K_p = proportional gain
K_i = integral gain
K_d = derivative gain
```

Trading interpretation:

| Term | Trading meaning |
|---|---|
| $K_p e_{p,k}$ | current distance from trend |
| $K_i I_k$ | accumulated trend pressure |
| $K_d D_k$ | rate of separation from trend |

---

## 6. Mapping the Inner Signal to Exposure

For spot-only trading:

$$
x_{inner,k} = \operatorname{clip}(u_{p,k},0,1)
$$

For long-short trading:

$$
x_{inner,k} = \operatorname{clip}(u_{p,k},-1,1)
$$

For the first implementation, use spot-only:

$$
x_{inner,k} = \operatorname{clip}(u_{p,k},0,1)
$$

where:

```text
x_inner,k = 0 means cash
x_inner,k = 1 means fully long
```

---

## 7. Threshold-Based Alternative

Instead of mapping the raw signal directly to exposure, a threshold approach
can be used.

```text
if rawSignal > entryThreshold:
    desiredMode = "long"
elif rawSignal < exitThreshold:
    desiredMode = "cash"
else:
    desiredMode = "hold"
```

This gives more discrete behaviour and can reduce over-trading.

A threshold version may be preferable if the PID signal is noisy.

---

## 8. Outer Loop: Benchmark-Relative Supervisory Controller

The outer loop does not directly predict price.

It monitors whether the strategy is gaining or losing ground relative to a
moving benchmark.

This is the more research-oriented part of the framework.

---

### 8.1 Benchmark-Relative Alpha Error

Use log-relative equity performance:

$$
e_{\alpha,k} = \log(V_{s,k}) - \log(V_{b,k})
$$

Equivalent form:

$$
e_{\alpha,k} = \log\left(\frac{V_{s,k}}{V_{b,k}}\right)
$$

Interpretation:

```text
e_alpha,k > 0 means strategy is beating the benchmark
e_alpha,k < 0 means strategy is underperforming the benchmark
```

Using log equity is useful because it naturally handles compounded returns.

---

### 8.2 Alpha Error Derivative

The benchmark is moving, so the controller must monitor whether the strategy
is gaining or losing ground.

Define:

$$
\Delta e_{\alpha,k}
=
e_{\alpha,k} - e_{\alpha,k-1}
$$

Interpretation:

```text
Delta e_alpha,k > 0 means strategy is gaining ground
Delta e_alpha,k < 0 means benchmark is gaining ground faster
```

This is important because the strategy can still be profitable in absolute
terms while losing ground relative to the benchmark.

---

## 9. Benchmark as a Moving Reference

The benchmark is not a fixed target. It is a moving reference trajectory.

For a buy-and-hold BTC benchmark:

$$
V_{b,k} = V_{b,k-1}\frac{p_k}{p_{k-1}}
$$

or:

$$
V_{b,k} = V_{b,0}\frac{p_k}{p_0}
$$

This means the benchmark equity curve moves with the asset.

The strategy is judged against this moving curve, not against zero profit.

---

## 10. Alpha Performance Regimes

The outer loop uses both alpha error and alpha error derivative.

### 10.1 Ahead and Pulling Away

Condition:

$$
e_{\alpha,k} > 0
$$

and:

$$
\Delta e_{\alpha,k} > 0
$$

Meaning:

```text
The strategy is beating the benchmark and gaining ground.
```

Action:

```text
Continue nominal trading.
Do not automatically increase risk.
```

---

### 10.2 Ahead but Losing Ground

Condition:

$$
e_{\alpha,k} > 0
$$

and:

$$
\Delta e_{\alpha,k} < 0
$$

Meaning:

```text
The strategy is still ahead, but the benchmark is catching up.
```

Action:

```text
Monitor for deterioration.
Apply mild correction if this persists.
```

---

### 10.3 Behind but Recovering

Condition:

$$
e_{\alpha,k} < 0
$$

and:

$$
\Delta e_{\alpha,k} > 0
$$

Meaning:

```text
The strategy is behind but gaining ground.
```

Action:

```text
Continue corrective mode.
Avoid over-correcting while recovery is occurring.
```

---

### 10.4 Behind and Falling Further

Condition:

$$
e_{\alpha,k} < 0
$$

and:

$$
\Delta e_{\alpha,k} < 0
$$

Meaning:

```text
The strategy is behind and the benchmark is gaining ground faster.
```

Action:

```text
Enter stronger corrective mode.
Diagnose failure mode.
Reduce trust in the current controller.
```

---

## 11. Supervisory Control Philosophy

The outer loop should not behave like this:

```text
if strategy is winning:
    increase exposure
```

That creates positive-feedback risk.

Instead, use this philosophy:

```text
if strategy is beating the benchmark:
    continue trading as-is

if strategy is not beating the benchmark:
    take adaptive corrective measures
```

The outer loop is a supervisor, not a leverage booster.

---

## 12. Supervisory Adaptation Signal

Define a supervisory adaptation signal:

$$
a_k
=
K_{\alpha p}e_{\alpha,k}
+
K_{\alpha d}\Delta e_{\alpha,k}
$$

where:

```text
K_alpha_p = alpha proportional gain
K_alpha_d = alpha derivative gain
```

Interpretation:

```text
a_k > 0 means relative performance is acceptable or improving
a_k < 0 means relative performance is deteriorating
```

The adaptation signal should be used to modify controller behaviour, not to
blindly increase leverage.

---

## 13. Correction Factor

Define a correction factor:

$$
c_{\alpha,k}
=
\operatorname{clip}
\left(
1 + K_{\alpha g}a_k,
c_{min},
1
\right)
$$

where:

```text
K_alpha_g = alpha correction gain
c_min     = minimum allowed correction factor
```

and:

$$
0 < c_{\alpha,k} \le 1
$$

The correction factor can reduce the influence of the inner controller when
relative performance deteriorates.

Corrected exposure:

$$
x_{corrected,k}
=
x_{inner,k}c_{\alpha,k}
$$

Important:

```text
The correction factor should not exceed 1 in the first implementation.
```

This prevents the alpha loop from becoming a leverage amplifier.

---

## 14. Adaptive Parameter Updates

The supervisory controller can also update the inner controller's parameters.

Define the parameter vector:

$$
\theta_k
=
\begin{bmatrix}
K_p \\
K_i \\
K_d \\
T_{entry} \\
T_{exit} \\
N_{MA}
\end{bmatrix}
$$

where:

```text
K_p     = proportional gain
K_i     = integral gain
K_d     = derivative gain
T_entry = entry threshold
T_exit  = exit threshold
N_MA    = moving average period
```

General update law:

$$
\theta_{k+1}
=
\theta_k
+
\eta \Delta \theta_k
$$

where:

```text
eta           = learning rate
Delta theta_k = parameter adjustment vector
```

The adjustment vector should be determined by the detected failure mode.

---

## 15. Failure Mode Diagnosis

The outer loop should diagnose why the strategy is underperforming.

Different failure modes require different corrections.

---

### 15.1 Too Many Bad Trades

Symptoms:

```text
high turnover
negative alpha error
frequent entries and exits
fee drag
many small losses
```

Possible corrections:

```text
increase entry threshold
increase smoothing
reduce K_p
reduce K_d
reduce max exposure
increase minimum time between trades
```

---

### 15.2 Missing Strong Trends

Symptoms:

```text
benchmark rising quickly
strategy is in cash
alpha error derivative is negative
price is above moving average
```

Possible corrections:

```text
lower entry threshold
shorten moving average period
increase K_p
increase K_d
reduce smoothing lag
increase time-in-market during bull regimes
```

---

### 15.3 Late Exits

Symptoms:

```text
strategy gives back gains
drawdown is increasing
price error is falling
derivative term is negative
```

Possible corrections:

```text
increase K_d
raise exit sensitivity
lower exit threshold
add trailing stop
reduce integral decay
clip or reset integral term
```

---

### 15.4 Choppy Sideways Market

Symptoms:

```text
price crosses moving average repeatedly
many small losing trades
turnover is high
alpha error deteriorates
```

Possible corrections:

```text
increase smoothing
increase entry threshold
reduce K_d
reduce max exposure
pause trading
require stronger confirmation
```

---

### 15.5 Integral Windup

Symptoms:

```text
integral term is large
trend has reversed
controller remains bullish too long
exits are delayed
```

Possible corrections:

```text
reduce K_i
reduce integral decay
clip integral term
reset integral term on regime change
reset integral term when exiting position
```

---

## 16. Risk Loop

The risk loop protects capital.

The system must include a drawdown-based risk limiter.

---

### 16.1 Running Equity Peak

Define running strategy equity peak:

$$
V_{peak,k} = \max_{j \le k} V_{s,j}
$$

---

### 16.2 Drawdown

Define drawdown:

$$
D_k
=
\frac{V_{peak,k}-V_{s,k}}{V_{peak,k}}
$$

Interpretation:

```text
D_k = 0.10 means the strategy is 10 percent below its peak equity.
```

---

### 16.3 Risk Factor

Define risk factor:

$$
r_{risk,k}
=
\operatorname{clip}
\left(
1 - \frac{D_k}{D_{max}},
0,
1
\right)
$$

where:

```text
D_max = maximum allowed drawdown
```

If drawdown approaches the maximum allowed drawdown, the risk factor approaches
zero.

---

### 16.4 Final Exposure

The final exposure is:

$$
x_k
=
\operatorname{clip}
\left(
x_{inner,k}
c_{\alpha,k}
r_{risk,k},
x_{min},
x_{max}
\right)
$$

For spot-only crypto:

$$
x_{min}=0
$$

$$
x_{max}=1
$$

Therefore:

$$
x_k
=
\operatorname{clip}
\left(
x_{inner,k}
c_{\alpha,k}
r_{risk,k},
0,
1
\right)
$$

---

## 17. Complete Control Equation Set

### 17.1 Price Error

$$
e_{p,k} = \frac{p_k - m_k}{m_k}
$$

### 17.2 Integral Term

$$
I_k = \lambda I_{k-1} + e_{p,k}
$$

### 17.3 Derivative Term

$$
D_k = e_{p,k} - e_{p,k-1}
$$

### 17.4 Inner PID Signal

$$
u_{p,k}
=
K_p e_{p,k}
+
K_i I_k
+
K_d D_k
$$

### 17.5 Inner Exposure

$$
x_{inner,k}
=
\operatorname{clip}(u_{p,k},0,1)
$$

### 17.6 Alpha Error

$$
e_{\alpha,k}
=
\log(V_{s,k}) - \log(V_{b,k})
$$

### 17.7 Alpha Error Derivative

$$
\Delta e_{\alpha,k}
=
e_{\alpha,k} - e_{\alpha,k-1}
$$

### 17.8 Adaptation Signal

$$
a_k
=
K_{\alpha p}e_{\alpha,k}
+
K_{\alpha d}\Delta e_{\alpha,k}
$$

### 17.9 Correction Factor

$$
c_{\alpha,k}
=
\operatorname{clip}
\left(
1 + K_{\alpha g}a_k,
c_{min},
1
\right)
$$

### 17.10 Drawdown

$$
D_k
=
\frac{V_{peak,k}-V_{s,k}}{V_{peak,k}}
$$

### 17.11 Risk Factor

$$
r_{risk,k}
=
\operatorname{clip}
\left(
1 - \frac{D_k}{D_{max}},
0,
1
\right)
$$

### 17.12 Final Exposure

$$
x_k
=
\operatorname{clip}
\left(
x_{inner,k}
c_{\alpha,k}
r_{risk,k},
0,
1
\right)
$$

---

## 18. Trading System Modes

The controller can operate in four modes:

```text
nominalMode
watchMode
correctiveMode
defensiveMode
pausedMode
```

---

### 18.1 Nominal Mode

Condition:

$$
e_{\alpha,k} \ge 0
$$

and:

$$
\Delta e_{\alpha,k} \ge 0
$$

Meaning:

```text
Strategy is ahead and not losing ground.
```

Behaviour:

```text
trade normally
do not increase risk just because performance is good
```

---

### 18.2 Watch Mode

Condition:

$$
e_{\alpha,k} > 0
$$

and:

$$
\Delta e_{\alpha,k} < 0
$$

Meaning:

```text
Strategy is ahead but benchmark is catching up.
```

Behaviour:

```text
continue trading
monitor deterioration
do not aggressively adapt yet
```

---

### 18.3 Corrective Mode

Condition:

$$
e_{\alpha,k} < 0
$$

or persistent:

$$
\Delta e_{\alpha,k} < 0
$$

Meaning:

```text
Strategy is underperforming or losing ground.
```

Behaviour:

```text
diagnose failure mode
adjust controller parameters
reduce trust in current controller
reduce exposure if required
```

---

### 18.4 Defensive Mode

Condition:

$$
D_k > D_{warning}
$$

Meaning:

```text
Drawdown has exceeded the warning threshold.
```

Behaviour:

```text
reduce max exposure
increase entry threshold
reduce turnover
disable weak entries
```

---

### 18.5 Paused Mode

Condition:

$$
D_k > D_{max}
$$

Meaning:

```text
Maximum allowed drawdown has been breached.
```

Behaviour:

```text
set final exposure to zero
stop new trades
wait for reset condition
```

---

## 19. Benchmark Options

Possible benchmarks:

```text
buy-and-hold BTC
buy-and-hold ETH
equal-weight BTC/ETH
market-cap weighted crypto index
cash/stablecoin
custom macro benchmark
```

For the first implementation, use:

```text
buy-and-hold BTC
```

Benchmark update:

$$
V_{b,k}=V_{b,k-1}\frac{p_k}{p_{k-1}}
$$

---

## 20. Causal Backtesting Requirements

The backtest must be causal.

At candle \(k\), the controller may only use information available at or
before candle \(k\).

Valid data at candle \(k\):

```text
price_k
movingAverage_k
strategyEquity_k
benchmarkEquity_k
previous price errors
previous alpha errors
previous signals
previous exposures
```

Invalid data at candle \(k\):

```text
future price
future moving average
future benchmark value
future max/min
future drawdown
future returns
```

No future information may be used to calculate the current signal.

---

## 21. Strategy Equity Update

For spot-only trading, define candle return:

$$
r_k = \frac{p_k - p_{k-1}}{p_{k-1}}
$$

If exposure from the previous candle is \(x_{k-1}\), then before fees:

$$
V_{s,k}
=
V_{s,k-1}
\left(
1 + x_{k-1}r_k
\right)
$$

With fees, define turnover:

$$
T_k = |x_k - x_{k-1}|
$$

Fee cost:

$$
F_k = f T_k V_{s,k-1}
$$

where:

```text
f = fee rate
```

Then:

$$
V_{s,k}
=
V_{s,k-1}
\left(
1 + x_{k-1}r_k
\right)
-
F_k
$$

---

## 22. Benchmark Equity Update

For a buy-and-hold benchmark:

$$
V_{b,k}
=
V_{b,k-1}
(1+r_k)
$$

Equivalent:

$$
V_{b,k}
=
V_{b,0}
\frac{p_k}{p_0}
$$

---

## 23. Metrics to Track

Track normal trading metrics:

```text
strategy return
benchmark return
alpha return
max drawdown
turnover
number of trades
win rate
average win
average loss
fee drag
time in market
Sharpe ratio
Sortino ratio
CAGR
```

Track controller-specific metrics:

```text
priceError
priceIntegral
priceDerivative
rawSignal
innerExposure
strategyEquity
benchmarkEquity
alphaError
alphaErrorDelta
adaptSignal
correctionFactor
drawdown
riskFactor
finalExposure
currentMode
```

---

## 24. Suggested Class Structure

```text
ControlTradingSystem
    MarketState
    MovingAveragePidController
    AlphaSupervisor
    RiskManager
    BacktestEngine
    MetricsTracker
```

---

### 24.1 MarketState

Responsible for storing candle-level market data.

Fields:

```text
timestamp
open
high
low
close
volume
movingAverage
return
```

---

### 24.2 MovingAveragePidController

Responsible for the inner-loop trade signal.

Inputs:

```text
price
movingAverage
previous price error
previous integral term
```

Outputs:

```text
priceError
priceIntegral
priceDerivative
rawSignal
innerExposure
```

---

### 24.3 AlphaSupervisor

Responsible for benchmark-relative monitoring.

Inputs:

```text
strategyEquity
benchmarkEquity
previous alpha error
```

Outputs:

```text
alphaError
alphaErrorDelta
adaptSignal
correctionFactor
currentMode
parameter adjustment
```

---

### 24.4 RiskManager

Responsible for drawdown and exposure limits.

Inputs:

```text
strategyEquity
equityPeak
correctedExposure
```

Outputs:

```text
drawdown
riskFactor
finalExposure
riskMode
```

---

### 24.5 BacktestEngine

Responsible for causal stepping through historical candles.

Inputs:

```text
historical candles
controller config
initial equity
fee rate
```

Outputs:

```text
equity curve
benchmark curve
controller state history
trade log
metrics summary
```

---

### 24.6 MetricsTracker

Responsible for computing performance statistics.

Outputs:

```text
total return
benchmark return
alpha return
max drawdown
trade count
fee drag
Sharpe ratio
Sortino ratio
time in market
```

---

## 25. Suggested Data Fields

Each candle output row should contain:

```text
timestamp
close
movingAverage
return
priceError
priceIntegral
priceDerivative
rawSignal
innerExposure
strategyEquity
benchmarkEquity
alphaError
alphaErrorDelta
adaptSignal
correctionFactor
drawdown
riskFactor
finalExposure
currentMode
feePaid
turnover
```

---

## 26. Suggested Config Fields

Use camelCase naming.

Keep all tuning values inside a config object.

```text
kp
ki
kd
integralDecay
entryThreshold
exitThreshold
alphaKp
alphaKd
alphaGain
minCorrectionFactor
maxAllowedDrawdown
warningDrawdown
minExposure
maxExposure
feeRate
movingAveragePeriod
initialEquity
```

Optional later-stage config fields:

```text
maxTurnover
minCandlesBetweenTrades
maxTradesPerDay
integralClip
derivativeSmoothing
regimeLookback
adaptiveLearningRate
```

---

## 27. First Implementation Target

Build the simplest version first:

```text
single asset
spot-only
BTC benchmark
one moving average
PID inner loop
alpha supervisor
drawdown risk limiter
causal backtest
CSV output of all controller states
```

Do not add machine learning initially.

Do not optimise parameters initially.

The first goal is:

```text
prove the control structure works mechanically
```

The second goal is:

```text
compare against buy-and-hold benchmark
```

The third goal is:

```text
test robustness across different market regimes
```

---

## 28. Research Framing

This system can be described as:

```text
A layered feedback-control framework for cryptocurrency trading, where the
inner loop converts price-vs-trend error into a trade signal, and the outer
loop supervises the controller using benchmark-relative performance error.
```

The key argument is:

```text
absolute profit is not enough
```

The controller must ask:

```text
is the strategy gaining or losing ground relative to the benchmark?
```

---

## 29. Difference From Standard Indicator Trading

A normal moving-average strategy asks:

```text
is price above or below the moving average?
```

This framework asks:

```text
is price above or below the moving average?
is the strategy beating the benchmark?
is the benchmark gaining ground faster?
is drawdown still bounded?
should the controller be trusted in the current regime?
```

That makes the system a feedback-control architecture rather than just a
static indicator strategy.

---

## 30. Key Safety Rule

Do not let underperformance cause uncontrolled exposure increases.

Avoid:

```text
if underperforming:
    increase exposure
```

That is martingale-like behaviour.

Prefer:

```text
if underperforming:
    diagnose failure mode
    reduce trust
    adapt parameters
    reduce risk if required
```

The outer loop should supervise the controller, not chase losses.

---

## 31. Core Summary

The full control logic can be summarised as:

```text
inner loop = local price signal
outer loop = benchmark-relative correction
risk loop  = capital preservation
```

Or:

```text
price error decides direction
alpha error decides correction
drawdown decides survival
```

The clean final equation is:

$$
x_k
=
\operatorname{clip}
\left(
x_{inner,k}
c_{\alpha,k}
r_{risk,k},
0,
1
\right)
$$

where:

$$
x_{inner,k}
=
\operatorname{clip}
\left(
K_p e_{p,k}
+
K_i I_k
+
K_d D_k,
0,
1
\right)
$$

$$
e_{p,k} = \frac{p_k-m_k}{m_k}
$$

$$
e_{\alpha,k}
=
\log(V_{s,k})-\log(V_{b,k})
$$

$$
\Delta e_{\alpha,k}
=
e_{\alpha,k}-e_{\alpha,k-1}
$$

$$
c_{\alpha,k}
=
\operatorname{clip}
\left(
1+K_{\alpha g}
\left(
K_{\alpha p}e_{\alpha,k}
+
K_{\alpha d}\Delta e_{\alpha,k}
\right),
c_{min},
1
\right)
$$

$$
r_{risk,k}
=
\operatorname{clip}
\left(
1-\frac{D_k}{D_{max}},
0,
1
\right)
$$

This is the minimal mathematical structure for a control-theoretic,
benchmark-relative crypto trading system.

---

## 32. Codex Implementation Prompt

Use the following prompt when asking Codex to implement the first version.

```text
Implement a causal backtesting framework for a control-theoretic crypto
trading system.

The system has:
1. A moving-average PID inner controller.
2. A benchmark-relative alpha supervisor.
3. A drawdown-based risk manager.

Use camelCase naming.

Use a config object for all parameters.

The strategy is spot-only, so exposure must be clipped between 0 and 1.

At each candle, calculate:
- moving average
- priceError = (price - movingAverage) / movingAverage
- leaky integral of priceError
- derivative of priceError
- raw PID signal
- inner exposure
- strategy equity
- benchmark equity
- alphaError = log(strategyEquity) - log(benchmarkEquity)
- alphaErrorDelta
- adaptation signal
- correction factor
- drawdown
- risk factor
- final exposure

The backtest must be causal. Do not use future data.

Output a CSV containing all controller state variables for every candle.

Also output summary metrics:
- strategy return
- benchmark return
- alpha return
- max drawdown
- turnover
- fee drag
- number of trades
- time in market
```
