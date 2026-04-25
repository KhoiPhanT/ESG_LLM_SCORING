"""
WUI Loader — Đọc dữ liệu World Uncertainty Index cho Việt Nam.
"""
import pandas as pd
import os


class WUILoader:
    def __init__(self, file_path="inputs/WUI.xlsx"):
        self.file_path = file_path
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.file_path):
            print(f"  [WARN] Không tìm thấy WUI file: {self.file_path}")
            return pd.DataFrame()

        df = pd.read_excel(self.file_path, header=3)
        df.columns = [
            "Period", "Lending_Rate", "Uncertainty_Word_Count",
            "Total_Word_Count", "VI_Uncertainty", "Trade_Uncertainty",
            "Total_Page_Count", "World_Trade_Uncertainty",
        ]
        # Bỏ hàng metadata (Code, Currency)
        df = df[~df["Period"].isin(["Code", "CURRENCY"])].copy()
        df = df.dropna(subset=["Period"])

        # Parse year và quarter
        def parse_period(p):
            parts = str(p).split()
            if len(parts) == 2:
                return parts[1], parts[0]
            return None, None

        df[["Year", "Quarter"]] = df["Period"].apply(
            lambda x: pd.Series(parse_period(x))
        )
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

        # Convert to numeric
        for col in ["VI_Uncertainty", "World_Trade_Uncertainty"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_wui_by_year(self, year):
        """Trung bình WUI 4 quý của năm."""
        if self.data.empty:
            return 0
        year_data = self.data[self.data["Year"] == year]
        if year_data.empty:
            return 0
        return year_data["VI_Uncertainty"].mean()

    def get_world_trade_uncertainty_by_year(self, year):
        """Trung bình World Trade Uncertainty 4 quý."""
        if self.data.empty:
            return 0
        year_data = self.data[self.data["Year"] == year]
        if year_data.empty:
            return 0
        return year_data["World_Trade_Uncertainty"].mean()

    def get_all_years(self):
        """Trả về dict {year: wui_value}."""
        if self.data.empty:
            return {}
        return self.data.groupby("Year")["VI_Uncertainty"].mean().to_dict()


if __name__ == "__main__":
    loader = WUILoader()
    print("WUI theo năm:")
    for year, wui in sorted(loader.get_all_years().items()):
        print(f"  {year}: WUI = {wui:.4f}")
