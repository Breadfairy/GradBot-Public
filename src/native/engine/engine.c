#include <math.h>
#include <stdlib.h>
#include "engine.h"

/* Clip a scalar into a fixed range. */
static double clipVal(double value, double low, double high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

/* Map positive z-score to a 0..1 fraction. */
static double zFracPositive(double zVal, double zMin, double zMax) {
    double zPos = zVal;
    double frac;

    if (zPos < 0.0) {
        zPos = 0.0;
    }
    if (zMax == zMin) {
        return 0.0;
    }

    frac = (zPos - zMin) / (zMax - zMin);
    return clipVal(frac, 0.0, 1.0);
}

/* Build EMA series. */
void emaLpf(const double* values, int n, int period, double* out) {
    int i;
    double alpha;
    double keep;

    if (n <= 0) {
        return;
    }
    if (period < 1) {
        period = 1;
    }

    alpha = 2.0 / ((double)period + 1.0);
    keep = 1.0 - alpha;
    out[0] = values[0];

    for (i = 1; i < n; i++) {
        out[i] = (alpha * values[i]) + (keep * out[i - 1]);
    }
}

/* Build first derivative series. */
void grad1Series(const double* values, int n, double target, double* out) {
    int i;
    double den;

    for (i = 0; i < n; i++) {
        out[i] = 0.0;
    }

    for (i = 1; i < n; i++) {
        den = values[i];
        if (den == 0.0) {
            den = 1e-12;
        }
        out[i] = ((values[i] - values[i - 1]) / den) * target;
    }
}

/* Build trend code array. */
void trendCodes(
    const double* m1,
    const double* m2,
    const double* m3,
    int n,
    int* out
) {
    int i;

    for (i = 0; i < n; i++) {
        out[i] = 0;

        if (m1[i] > m2[i] && m2[i] > m3[i]) {
            out[i] = 1;
        }
        else if (m1[i] < m2[i] && m2[i] < m3[i]) {
            out[i] = -1;
        }
        else if (m1[i] < m3[i] && m3[i] < m2[i]) {
            out[i] = -2;
        }
        else if (m1[i] > m3[i] && m3[i] > m2[i]) {
            out[i] = 2;
        }
    }
}

/* Build rolling mean and std arrays. */
void rollingMeanAndStd(
    const double* series,
    int n,
    int window,
    double* meanOut,
    double* stdOut
) {
    int i;
    double sum = 0.0;
    double sum2 = 0.0;
    double val;
    double old;
    double meanVal;
    double var;

    if (window < 1) {
        window = 1;
    }

    for (i = 0; i < n; i++) {
        meanOut[i] = NAN;
        stdOut[i] = NAN;
    }

    for (i = 0; i < n; i++) {
        val = series[i];
        sum += val;
        sum2 += val * val;

        if (i >= window) {
            old = series[i - window];
            sum -= old;
            sum2 -= old * old;
        }

        if (i >= (window - 1)) {
            meanVal = sum / (double)window;
            var = (sum2 / (double)window) - (meanVal * meanVal);
            if (var < 0.0) {
                var = 0.0;
            }
            meanOut[i] = meanVal;
            stdOut[i] = sqrt(var);
        }
    }
}

/* Build regime energy array. */
void energyCsum(
    const double* m1,
    const double* m2,
    const double* m3,
    const int* trendCode,
    int n,
    int leg,
    double* out
) {
    int i;
    int code;
    int prevReg = 0;
    int prevValid = 0;
    double running = 0.0;
    double delta;

    for (i = 0; i < n; i++) {
        code = trendCode[i];
        if (abs(code) != 1) {
            running = 0.0;
            out[i] = 0.0;
            prevReg = 0;
            prevValid = 0;
            continue;
        }

        if (leg == 12) {
            delta = fabs(m1[i] - m2[i]);
        }
        else {
            delta = fabs(m2[i] - m3[i]);
        }

        if (!prevValid || code != prevReg) {
            running = 0.0;
        }

        running += delta;
        out[i] = running;
        prevReg = code;
        prevValid = 1;
    }
}

/* Build regime spread peak ratio array. */
void spreadPeakRatioFromMas(
    const double* mA,
    const double* mB,
    const int* trendCode,
    int n,
    double* out
) {
    int i;
    int code;
    int prevReg = 0;
    int prevValid = 0;
    double peak = 0.0;
    double spread;

    for (i = 0; i < n; i++) {
        code = trendCode[i];
        if (abs(code) != 1) {
            peak = 0.0;
            out[i] = 1.0;
            prevReg = 0;
            prevValid = 0;
            continue;
        }

        spread = fabs(mA[i] - mB[i]);
        if (!prevValid || code != prevReg) {
            peak = spread;
        }
        else if (spread > peak) {
            peak = spread;
        }

        if (peak > 0.0) {
            out[i] = spread / peak;
        }
        else {
            out[i] = 1.0;
        }

        prevReg = code;
        prevValid = 1;
    }
}

/* Build macro dynamic threshold series. */
void macroDynFromMas(
    const double* m1,
    const double* m2,
    const double* m3,
    int n,
    double barsPerDay,
    double winDays,
    double zMin,
    double zMax,
    double pctMax,
    double pctMin,
    double gradWinDays,
    double gradZMin,
    double gradZMax,
    double gradMultMin,
    double gradMultMax,
    double* out
) {
    int i;
    int winBars;
    int gradBars;
    int lastStart = 0;
    int regValid = 0;
    int start = 0;
    double den;
    double spread;
    double spacing13Pct;
    double zSpacing;
    double baseMag;
    double mult;
    double dynMag;
    double warmFrac;
    double ageBars;
    double g1p2Abs;
    double zGrad;
    double gradFrac;
    double gradMult;
    double maxStep;
    double prev;
    double cur;
    double step;
    int sameSide;
    int* spreadSign;
    double* spacingArr;
    double* meanArr;
    double* stdArr;
    double* ratioArr;
    double* gradArr;
    double* gradMean;
    double* gradStd;

    if (n <= 0) {
        return;
    }

    winBars = (int)round(winDays * barsPerDay);
    if (winBars < 1) {
        winBars = 1;
    }

    gradBars = (int)round(gradWinDays * barsPerDay);
    if (gradBars < 1) {
        gradBars = 1;
    }

    spreadSign = (int*)malloc((size_t)n * sizeof(int));
    spacingArr = (double*)malloc((size_t)n * sizeof(double));
    meanArr = (double*)malloc((size_t)n * sizeof(double));
    stdArr = (double*)malloc((size_t)n * sizeof(double));
    ratioArr = (double*)malloc((size_t)n * sizeof(double));
    gradArr = (double*)malloc((size_t)n * sizeof(double));
    gradMean = (double*)malloc((size_t)n * sizeof(double));
    gradStd = (double*)malloc((size_t)n * sizeof(double));

    for (i = 0; i < n; i++) {
        spread = m1[i] - m3[i];
        if (spread > 0.0) {
            spreadSign[i] = 1;
        }
        else if (spread < 0.0) {
            spreadSign[i] = -1;
        }
        else {
            spreadSign[i] = 0;
        }

        den = fabs(m3[i]);
        if (den == 0.0) {
            den = 1e-12;
        }
        spacingArr[i] = (fabs(spread) / den) * 100.0;
    }

    rollingMeanAndStd(spacingArr, n, winBars, meanArr, stdArr);
    spreadPeakRatioFromMas(m1, m3, spreadSign, n, ratioArr);

    gradArr[0] = 0.0;
    for (i = 1; i < n; i++) {
        den = m2[i];
        if (den == 0.0) {
            den = 1e-12;
        }
        gradArr[i] = fabs(((m2[i] - m2[i - 1]) / den) * 100.0);
    }
    rollingMeanAndStd(gradArr, n, gradBars, gradMean, gradStd);

    for (i = 0; i < n; i++) {
        spacing13Pct = spacingArr[i];
        if (!isnan(meanArr[i]) && !isnan(stdArr[i]) && stdArr[i] > 1e-6) {
            zSpacing = (spacing13Pct - meanArr[i]) / stdArr[i];
        }
        else {
            zSpacing = 0.0;
        }
        zSpacing = clipVal(zSpacing, -10.0, 10.0);
        baseMag = zFracPositive(zSpacing, zMin, zMax) * pctMax;

        mult = ratioArr[i];

        regValid = abs(spreadSign[i]) == 1;
        if (regValid) {
            start = 0;
            if (i == 0) {
                start = 1;
            }
            else if (
                abs(spreadSign[i - 1]) != 1
                || spreadSign[i] != spreadSign[i - 1]
            ) {
                start = 1;
            }
            if (start) {
                lastStart = i;
            }
            ageBars = (double)(i - lastStart);
            warmFrac = clipVal(ageBars / (double)winBars, 0.0, 1.0);
        }
        else {
            warmFrac = 0.0;
        }

        if (pctMin > 0.0) {
            if (baseMag < pctMin) {
                baseMag = pctMin;
            }
            baseMag = pctMin + ((baseMag - pctMin) * warmFrac);
            dynMag = pctMin + ((baseMag - pctMin) * mult);
        }
        else {
            dynMag = (baseMag * warmFrac) * mult;
        }

        g1p2Abs = gradArr[i];
        if (
            !isnan(gradMean[i])
            && !isnan(gradStd[i])
            && gradStd[i] > 1e-6
        ) {
            zGrad = (g1p2Abs - gradMean[i]) / gradStd[i];
        }
        else {
            zGrad = 0.0;
        }
        zGrad = clipVal(zGrad, -10.0, 10.0);
        gradFrac = zFracPositive(zGrad, gradZMin, gradZMax);
        gradMult = gradMultMin + (gradFrac * (gradMultMax - gradMultMin));
        dynMag = dynMag * gradMult;

        if (pctMin > 0.0) {
            dynMag = clipVal(dynMag, pctMin, pctMax);
        }
        else {
            dynMag = clipVal(dynMag, 0.0, pctMax);
        }

        out[i] = (double)spreadSign[i] * dynMag;
    }

    maxStep = (pctMax - (pctMin > 0.0 ? pctMin : 0.0)) * 0.25;
    if (maxStep > 0.0 && n > 1) {
        for (i = 1; i < n; i++) {
            prev = out[i - 1];
            cur = out[i];
            sameSide = (
                (prev > 0.0 && cur > 0.0)
                || (prev < 0.0 && cur < 0.0)
            );
            if (sameSide) {
                step = cur - prev;
                step = clipVal(step, -maxStep, maxStep);
                out[i] = prev + step;
            }
        }
    }

    free(spreadSign);
    free(spacingArr);
    free(meanArr);
    free(stdArr);
    free(ratioArr);
    free(gradArr);
    free(gradMean);
    free(gradStd);
}
