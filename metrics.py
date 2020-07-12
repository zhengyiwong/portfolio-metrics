import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
from scipy.interpolate import interp1d
from scipy.stats import norm

# Portfolio Metrics Functions Created by Wong Zhengyi
# Useful for creating your own Tear-Sheet


def max_drawdown(returns: pd.Series):
    """
    Takes a time series of asset returns and returns the percentage drawdown
    Drawdown represents the lost from peak to through
    """
    index = 1000 * (1 + returns).cumprod()
    drawdowns = index - index.cummax() / index.cummax()
    return pd.Series(drawdowns)


def skew(returns: pd.Series):
    """
    Computes the skewness of a asset series
    """
    relative_ret = returns - returns.mean()
    sigma = returns.std(ddof=0)
    exp = (relative_ret ** 3).mean()
    return exp / sigma ** 3


def kurtosis(returns: pd.Series):
    """
    Computes the skewness of a asset series
    """
    relative_ret = returns - returns.mean()
    sigma = returns.std(ddof=0)
    exp = (relative_ret ** 4).mean()
    return exp / sigma ** 4


def jb_test(data, level=0.01):
    """
    Applies the Jarque-Bera test to test for normal distribution in a series
    Test is applied at a default of 1% level
    Returns True if hypothesis is accepted, False if otherwise
    """
    n = data.shape[0]
    s = n / 6 * (skew(data) ** 2 + 0.25 * (kurtosis(data) - 3 ** 2))
    return s > level


def semideviation(data):
    """
    Computes the semideviation of a series/dataframe
    """
    check_negative = data < 0
    return data[check_negative].std(ddof=0)


def historic_var(data, level=5):
    """
    Returns the historic value at risk at a specified level
    (100 - level) percent are  above the returned number
    """
    if isinstance(data, pd.DataFrame):
        return data.aggregate(historic_var, level=level)
    elif isinstance(data, pd.Series):
        return -np.percentile(data, level) / 100
    else:
        raise TypeError("Expected Data to be a Series or DataFrame")


def gaussian_var(data, level=0.05, cornish_fisher=False):
    """
    Returns the parametric Gaussian VaR of a Series or DataFrame at a specified level
    """
    z_score = norm.ppf(level)
    # Modify Z-Score based on observed skewness and kurtosis
    if cornish_fisher:
        s = skew(data)
        k = kurtosis(data)
        z_score = (
            z_score
            + (z_score ** 2 - 1) * s / 6
            + (z_score ** 3 - 3 * z_score) * (k - 3) / 24
            - (2 * z_score ** 3 - 5 * z_score) * (s ** 2) / 36
        )
    return -(data.mean() + z_score * data.std(ddof=0)) / 100


def historic_cvar(data, level=5):
    """
    Returns the historic value at risk at a specified level
    (100 - level) percent are  above the returned number
    """
    if isinstance(data, pd.Series):
        return -data[data <= -historic_var(data, level=level)].mean() / 100
    elif isinstance(data, pd.DataFrame):
        return data.aggregate(historic_cvar, level=level)
    else:
        raise TypeError("Expected Data to be a Series or DataFrame")


def annualize_vol(returns, periods):
    """
    Annualizes vol of a set of returns. If daily data annualize by all trading days e.g. 252
    """
    return returns.std() * np.sqrt(periods)


def annualize_ret(returns, periods):
    """
    Annualizes a set of returns
    """
    cum_ret = (1 + returns).prod()
    n_periods = returns.shape[0]
    return cum_ret ** (periods / n_periods) - 1


def sharpe_ratio(returns, rfr, periods):
    """
    Computes the annualized sharpe ratio
    """
    rf_period = (1 + rfr) ** (1 / periods) - 1
    excess_ret = returns - rf_period
    ann_ex_ret = annualize_ret(excess_ret, periods)
    ann_vol = annualize_vol(returns, periods)
    return ann_ex_ret / ann_vol


def information_ratio(returns, bmk_returns, periods):
    """
    Information / Appraisal Ratio that compares the active return of an investment
    compared to a benchmark index relative to the volatility of the active return
    """
    # Annual the returns first depending on periods
    ann_ret = ((np.prod(1 + returns)) ** (1 / len(returns))) ** periods - 1
    ann_bmk_ret = ((np.prod(1 + bmk_returns)) ** (1 / len(bmk_returns))) ** periods - 1
    ir = (ann_ret - ann_bmk_ret) / (np.sqrt(12) * np.std(returns - bmk_returns, ddof=1))
    return ir


def beta(returns, bmk_returns):
    ret = np.array(returns).reshape(-1, 1)
    bmk_ret = np.array(bmk_returns).reshape(-1, 1)
    bmk_ret_w_constant = sm.add_constant(bmk_ret)
    model = sm.OLS(ret, bmk_ret_w_constant).fit()
    return model.params[1]


def expost_te(returns, bmk_returns, periods):
    """
    Calculate the EX-POST Tracking Error. Returns a Dataframe of rolling Ex-Post Tracking Error Annualized
    """
    temp = pd.concat([pd.DataFrame(returns) - pd.DataFrame(bmk_returns)], axis=1)
    temp = temp.dropna()
    temp.columns = ["returns", "bmk_returns"]
    temp = temp.returns - temp.bmk_returns
    rolling_te = temp.rolling(periods).apply(
        lambda x: (np.std(x, ddof=1) * np.sqrt(12)), raw=True
    )
    rolling_te = rolling_te.dropna()
    return rolling_te


def exante_te(
    tilt_weights: pd.DataFrame,
    returns: pd.DataFrame,
    window_periods=60,
    annualised_periods=12,
):
    """
    Calculate the EX-ANTE Tracking Error. Returns a DataFrame of Ex-Ante Tracking Error
    tilt_weights : Weight difference between the absolute portfolio weights and benchmark weights
    returns : Historical Returns
    window_periods : Window of Tracking Error
    annualised_periods : Period to annualize returns
    """
    assert (
        tilt_weights.shape[0] <= returns.shape[0]
    ), "Weights Longer Than historical returns"
    tilt_weights.index = pd.to_datetime(tilt_weights.index)
    ret = returns[tilt_weights.columns]
    tilt_dict = tilt_weights.T.to_dict("list")
    ret.index = pd.to_datetime(ret.index)
    ret = ret.sort_index()
    date_list = tilt_weights.index.tolist()
    final_te = []
    for latest_date in date_list:
        tilt_weight = tilt_dict[latest_date]
        start_date = latest_date - pd.offsets.MonthEnd(window_periods)
        ret_period = ret[(ret.index >= start_date) & (ret.index <= latest_date)]
        cov_period = ret_period.cov()
        te = np.sqrt(annualised_periods) * np.sqrt(
            tilt_weight @ cov_period @ np.transpose(tilt_weight)
        )
        final_te.append(te)
    final_te = (
        pd.DataFrame(list(zip(date_list, final_te)), columns=["date_list", "TE"])
        .set_index("date_list")
        .rename_axis("Dates")
    )
    final_te = final_te.dropna()
    final_te = final_te.reset_index()
    return final_te


def risk_contri(weights, returns, latest_date, window_period=60, annualised_period=12):
    """
    Calculates the tracking error contribution for the period specified from each contributing asset
    Formula:
        cov = covariance matrix of returns * active weights * active weights'
        total risk = sqrt(sum of all values in cov * annualized period) * 10000    # Make basis point readable
        risk contri : (sum of cov by row) / sum of all values in cov * total risk
    Final output is a series with each row representing an asset TE contribution

    weights : DataFrame of asset weights
    returns : DataFrame of asset returns
    latest_date : Latest Date
    window_period : Tracking Error window period
    annualised_period : Period to annualize returns
    """
    output = dict()
    latest_date = pd.to_datetime(latest_date)
    start_date = latest_date - pd.DateOffset(months=window_period)
    ret = returns[(returns.index >= start_date) & (returns.index <= latest_date)]
    cov = ret.cov()
    final_cov = cov.mul(weights, axis=0)
    final_cov = final_cov.mul(weights, axis=1)
    # TOTAL RISK
    totalrisk = np.sqrt(final_cov.values.sum() * annualised_period) * 10000
    # Risk Contri
    sum_row = final_cov.sum(axis=1)
    risk_contri = sum_row / final_cov.values.sum() * totalrisk
    f_weights = pd.Series(weights, index=weights.columns)
    for i in weights.columns:
        output[i] = dict()
        output[str(i)]["asset_te"] = risk_contri[i]
        output[str(i)]["weight"] = f_weights[i]
    return pd.DataFrame.from_dict(output).fillna(0)


def up_down_ratio(returns, bmk_returns, rolling_periods, annualize_periods):
    """
    The hit rate is the % of rolling period which outperforms the benchmark.
    Returns 4 specific metrics
    1) Pos Hit Rate : % of rolling period which outperform benchmark
    2) Avg Up : Average of out-performance conditional that it outperforms
    3) Avg Down : Average of out-performance conditional that it underperforms
    4) Avg up to down : Total number of outperform / Total number of underperform
    """
    temp = pd.concat([pd.DataFrame(returns), pd.DataFrame(bmk_returns)], axis=1)
    temp = temp.dropna()
    rr = temp.rolling(rolling_periods).apply(
        lambda x: pd.DataFrame(x).add(1).prod() ** (annualize_periods / len(x)) - 1,
        raw=True,
    )
    rr = rr.dropna()
    rr.columns = ["returns", "bmk_returns"]
    poshitrate = (rr.returns > rr.bmk_returns).mean()
    avgup = ((rr.returns - rr.bmk_returns)[rr.returns > rr.bmk_returns]).mean()
    avgdown = ((rr.returns - rr.bmk_returns)[rr.returns < rr.bmk_returns]).mean()
    try:
        avguptodown = sum(rr.returns > rr.bmk_returns) / sum(
            rr.returns < rr.bmk_returns
        )
    except ZeroDivisionError:
        avguptodown = 1
    return poshitrate, avgup, avgdown, avguptodown


def expected_ddn(
    w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rfr: np.ndarray, horizon: int
):
    """
    Returns the expectation Value E[D] of maximum drawdowns of Brownian Motion for a given drift mean, variance sd,
    and runtime horizon of the Brownian Motion process.
    Based on http://www.cs.rpi.edu/~magdon/ps/journal/drawdown_journal.pdf
    w : current portfolio weight
    mu : expected return
    sigma : historical returns covariance matrix
    rfr : risk Free Rate
    horizon : runtime horizon
    """
    exp_ret = w @ (mu + rfr)
    vol = np.sqrt(w @ sigma @ w)
    eddn = maxddstats(mu=exp_ret - 0.5 * vol ** 2, sigma=vol, horizon=horizon)
    return 1 - 1 / math.exp(eddn)


def maxddstats(mu: np.ndarray, sigma: np.ndarray, horizon: int):
    """
    Helps with the calculation of Expected Value of Maximum Drawdown
    """
    # Internal Function - POSITIVE CASE: mu > 0
    def qp(_x):
        gamma = math.sqrt(math.pi / 8)
        vqn = np.reshape(
            np.array(
                [
                    0.0005,
                    0.019690,
                    0.0010,
                    0.027694,
                    0.0015,
                    0.033789,
                    0.0020,
                    0.038896,
                    0.0025,
                    0.043372,
                    0.0050,
                    0.060721,
                    0.0075,
                    0.073808,
                    0.0100,
                    0.084693,
                    0.0125,
                    0.094171,
                    0.0150,
                    0.102651,
                    0.0175,
                    0.110375,
                    0.0200,
                    0.117503,
                    0.0225,
                    0.124142,
                    0.0250,
                    0.130374,
                    0.0275,
                    0.136259,
                    0.0300,
                    0.141842,
                    0.0325,
                    0.147162,
                    0.0350,
                    0.152249,
                    0.0375,
                    0.157127,
                    0.0400,
                    0.161817,
                    0.0425,
                    0.166337,
                    0.0450,
                    0.170702,
                    0.0500,
                    0.179015,
                    0.0600,
                    0.194248,
                    0.0700,
                    0.207999,
                    0.0800,
                    0.220581,
                    0.0900,
                    0.232212,
                    0.1000,
                    0.243050,
                    0.2000,
                    0.325071,
                    0.3000,
                    0.382016,
                    0.4000,
                    0.426452,
                    0.5000,
                    0.463159,
                    1.5000,
                    0.668992,
                    2.5000,
                    0.775976,
                    3.5000,
                    0.849298,
                    4.5000,
                    0.905305,
                    10.000,
                    1.088998,
                    20.000,
                    1.253794,
                    30.000,
                    1.351794,
                    40.000,
                    1.421860,
                    50.000,
                    1.476457,
                    150.00,
                    1.747485,
                    250.00,
                    1.874323,
                    350.00,
                    1.958037,
                    450.00,
                    2.020630,
                    1000.0,
                    2.219765,
                    2000.0,
                    2.392826,
                    3000.0,
                    2.494109,
                    4000.0,
                    2.565985,
                    5000.0,
                    2.621743,
                ]
            ),
            (-1, 2),
        )
        # Interpolation
        if _x < 0:
            raise ValueError("error x < 0 in QP(x)")
        if 0 <= _x < 5000:
            ax = np.log(vqn[:, 0])
            by = vqn[:, 1]
            return interp1d(ax, by, kind="cubic", bounds_error=False)(np.log(_x))
        if _x > 5000:
            return 0.25 * np.log(_x) + 0.49088

    # Internal Function - NEGATIVE CASE: mu < 0
    def qn(_x):
        gamma = math.sqrt(math.pi / 8)
        vqn = np.reshape(
            np.array(
                [
                    0.0005,
                    0.019965,
                    0.0010,
                    0.028394,
                    0.0015,
                    0.034874,
                    0.0020,
                    0.040369,
                    0.0025,
                    0.045256,
                    0.0050,
                    0.064633,
                    0.0075,
                    0.079746,
                    0.0100,
                    0.092708,
                    0.0125,
                    0.104259,
                    0.0150,
                    0.114814,
                    0.0175,
                    0.124608,
                    0.0200,
                    0.133772,
                    0.0225,
                    0.142429,
                    0.0250,
                    0.150739,
                    0.0275,
                    0.158565,
                    0.0300,
                    0.166229,
                    0.0325,
                    0.173756,
                    0.0350,
                    0.180793,
                    0.0375,
                    0.187739,
                    0.0400,
                    0.194489,
                    0.0425,
                    0.201094,
                    0.0450,
                    0.207572,
                    0.0475,
                    0.213877,
                    0.0500,
                    0.220056,
                    0.0550,
                    0.231797,
                    0.0600,
                    0.243374,
                    0.0650,
                    0.254585,
                    0.0700,
                    0.265472,
                    0.0750,
                    0.276070,
                    0.0800,
                    0.286406,
                    0.0850,
                    0.296507,
                    0.0900,
                    0.306393,
                    0.0950,
                    0.316066,
                    0.1000,
                    0.325586,
                    0.1500,
                    0.413136,
                    0.2000,
                    0.491599,
                    0.2500,
                    0.564333,
                    0.3000,
                    0.633007,
                    0.3500,
                    0.698849,
                    0.4000,
                    0.762455,
                    0.5000,
                    0.884593,
                    1.0000,
                    1.445520,
                    1.5000,
                    1.970740,
                    2.0000,
                    2.483960,
                    2.5000,
                    2.990940,
                    3.0000,
                    3.492520,
                    3.5000,
                    3.995190,
                    4.0000,
                    4.492380,
                    4.5000,
                    4.990430,
                    5.0000,
                    5.498820,
                ]
            ),
            (-1, 2),
        )
        # Interpolation
        if _x < 0:
            raise ValueError("error x < 0 in QN(x)")
        if 0 <= _x < 0.0005:
            return gamma * math.sqrt(2 * _x)
        if 0.0005 <= _x <= 5:
            ax = vqn[:, 0]
            by = vqn[:, 1]
            return interp1d(ax, by, kind="cubic", bounds_error=False)(_x)
        if _x > 5:
            return _x + 1 / 2

    # Results
    if mu != 0:
        x = (mu ** 2) * horizon / (2 * sigma ** 2)
        if mu > 0:
            if x < 0:
                raise ValueError("x is negative 1")
            a = qp(x)
            return (2 * sigma ** 2 / mu) * a
        if mu < 0:
            if x < 0:
                raise ValueError("x is negative 2")
            a = qn(x)
            return -(2 * sigma ** 2 / mu) * a
    else:
        g = math.sqrt(math.pi / 8)
        return 2 * g * sigma * math.sqrt(horizon)
