# カーブ構築モジュール

import numpy as np
from scipy.interpolate import interp1d
from datetime import date

# 日数計算

def year_fraction(start: date, end: date, convention: str = "ACT/365") -> float:
    days = (end - start).days
    if convention == "ACT/365":
        return days / 365.0
    elif convention == "ACT/360":
        return days / 360.0
    elif convention == "30/360":
        d1, d2 = start.day, end.day
        m1, m2 = start.month, end.month
        y1, y2 = start.year, end.year
        D1 = min(d1, 30)
        D2 = d2 if d1 < 30 else min(d2, 30)
        return ((y2 - y1) * 360 + (m2 - m1) * 30 + (D2 - D1)) / 360.0
    else:
        raise ValueError(f"Unsupported convention: {convention}")

class FundingCurveBuilder:
    """
    離散的なOISポイントと（オプション）信用スプレッドポイントから
    連続的な資金調達カーブを構築します。
    """
    def __init__(self, ois_points: dict, valuation_date: date,
                 dc: str = "ACT/365", cs_points: dict = None):
        """
        ois_points: {日付: 金利（小数）}
        cs_points: {日付: 信用スプレッド（小数）} (オプション)
        valuation_date: 年分数計算の基準日
        dc: 日数計算方法
        """
        self.valuation_date = valuation_date
        self.dc = dc
        # OISカーブの準備
        self.ois_dates = sorted(ois_points.keys())
        self.ois_tenors = np.array([
            year_fraction(valuation_date, d, dc)
            for d in self.ois_dates
        ])
        self.ois_rates = np.array([ois_points[d] for d in self.ois_dates])
        self.ois_interp = None
        # 信用スプレッドカーブの準備（提供されている場合）
        self.cs_interp = None
        if cs_points:
            self.cs_dates = sorted(cs_points.keys())
            self.cs_tenors = np.array([
                year_fraction(valuation_date, d, dc)
                for d in self.cs_dates
            ])
            self.cs_rates = np.array([cs_points[d] for d in self.cs_dates])
        else:
            self.cs_tenors = None
            self.cs_rates = None

    def build(self):
        # OIS補間器の構築
        self.ois_interp = interp1d(
            self.ois_tenors, self.ois_rates,
            kind='linear', fill_value='extrapolate', assume_sorted=True
        )
        # 信用スプレッド補間器の構築（利用可能な場合）
        if self.cs_tenors is not None:
            self.cs_interp = interp1d(
                self.cs_tenors, self.cs_rates,
                kind='linear', fill_value='extrapolate', assume_sorted=True
            )
        # 合成カーブ：OISと信用スプレッドの合計
        def total_rate(t: float) -> float:
            base = float(self.ois_interp(t))
            cs = float(self.cs_interp(t)) if self.cs_interp is not None else 0.0
            return base + cs
        self.total_interp = total_rate
        return self.total_interp

class DiscountFactorService:
    """
    補間された合計金利カーブを使用して割引係数を提供します。
    """
    def __init__(self, interp, valuation_date: date, dc: str = "ACT/365"):
        self.interp = interp
        self.valuation_date = valuation_date
        self.dc = dc

    def get_df(self, target_date: date) -> float:
        t = year_fraction(self.valuation_date, target_date, self.dc)
        r = float(self.interp(t))
        return np.exp(-r * t)
