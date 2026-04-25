"""
PLS-SEM — Mô hình phương trình cấu trúc bình phương nhỏ nhất riêng phần.
Phân tích nhân quả: ESG → Hiệu quả tài chính (ROA, Firm Value).
Chạy thật với thư viện semopy trên dữ liệu từ Data(4).xlsx.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class PLS_SEM_Model:
    def __init__(self, data_path="inputs/ACB/chỉ số hiệu qủa tài chính/Data (4).xlsx"):
        self.data_path = data_path
        self.data = None
        self.results = None

    def load_data(self, companies=None, min_year=2015):
        """Load dữ liệu từ Excel, filter theo companies và năm."""
        df = pd.read_excel(self.data_path, sheet_name="Final - stata")

        # Filter
        if companies:
            df = df[df["company"].isin(companies)]
        df = df[df["year"] >= min_year]

        # Giữ các cột cần thiết
        cols = ["company", "year", "esg", "roa", "lev", "size", "fv", "growth"]
        available = [c for c in cols if c in df.columns]
        self.data = df[available].dropna()
        print(f"  [PLS-SEM] Loaded {len(self.data)} observations, {self.data['company'].nunique()} companies")
        return self.data

    def run_sem(self):
        """Chạy SEM thật bằng semopy."""
        if self.data is None or len(self.data) < 10:
            print("  [PLS-SEM] Không đủ dữ liệu")
            return None

        try:
            import semopy

            # Standardize data
            df = self.data.copy()
            for col in ["esg", "roa", "lev", "size"]:
                if col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()

            # Mô hình: ESG → ROA, kiểm soát bởi LEV, SIZE
            model_desc = """
            roa ~ esg + lev + size
            """

            model = semopy.Model(model_desc)
            result = model.fit(df)

            # Lấy estimates
            estimates = model.inspect()
            print("\n  [PLS-SEM] Kết quả ước lượng:")
            print(estimates.to_string())

            # Parse key results
            path_coefficients = {}
            for _, row in estimates.iterrows():
                if row["op"] == "~":
                    key = f"{row['rval']} → {row['lval']}"
                    path_coefficients[key] = {
                        "coefficient": round(float(row["Estimate"]), 4),
                        "std_err": round(float(row["Std. Err"]), 4) if pd.notna(row.get("Std. Err")) else None,
                        "p_value": round(float(row["p-value"]), 4) if pd.notna(row.get("p-value")) else None,
                    }

            self.results = {
                "path_coefficients": path_coefficients,
                "n_observations": len(df),
                "model": "roa ~ esg + lev + size",
            }
            return self.results

        except Exception as e:
            print(f"  [PLS-SEM ERROR] {e}")
            # Fallback: OLS regression
            return self._fallback_regression()

    def _fallback_regression(self):
        """Fallback: dùng OLS regression nếu semopy lỗi."""
        from scipy import stats

        df = self.data.copy()
        if "esg" not in df.columns or "roa" not in df.columns:
            return None

        # Simple OLS: ROA = a + b*ESG + c*LEV + d*SIZE
        X_cols = [c for c in ["esg", "lev", "size"] if c in df.columns]
        y = df["roa"].values
        X = df[X_cols].values
        X = np.column_stack([np.ones(len(X)), X])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            n, k = X.shape
            mse = np.sum(residuals ** 2) / (n - k)
            se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
            t_stats = beta / se
            p_values = [2 * (1 - stats.t.cdf(abs(t), n - k)) for t in t_stats]

            r_squared = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)

            path_coefficients = {}
            labels = ["intercept"] + X_cols
            for i, label in enumerate(labels):
                if label != "intercept":
                    path_coefficients[f"{label} → roa"] = {
                        "coefficient": round(beta[i], 4),
                        "std_err": round(se[i], 4),
                        "p_value": round(p_values[i], 4),
                    }

            self.results = {
                "path_coefficients": path_coefficients,
                "r_squared": round(r_squared, 4),
                "n_observations": n,
                "model": "OLS: roa ~ esg + lev + size",
            }
            print(f"\n  [OLS Regression] R² = {r_squared:.4f}")
            for k, v in path_coefficients.items():
                sig = "***" if v["p_value"] < 0.01 else "**" if v["p_value"] < 0.05 else "*" if v["p_value"] < 0.1 else ""
                print(f"    {k}: β={v['coefficient']:.4f}, p={v['p_value']:.4f} {sig}")
            return self.results

        except Exception as e:
            print(f"  [OLS ERROR] {e}")
            return None

    def interpret(self):
        """Giải thích kết quả."""
        if not self.results:
            return "Chưa có kết quả"

        interpretations = []
        for path, info in self.results.get("path_coefficients", {}).items():
            coeff = info["coefficient"]
            p = info.get("p_value")
            sig = p is not None and p < 0.05

            if "esg" in path:
                if sig and coeff > 0:
                    interpretations.append(f"✅ ESG có tác động TÍCH CỰC đến ROA (β={coeff:.4f}, p={p:.4f})")
                elif sig and coeff < 0:
                    interpretations.append(f"⚠ ESG có tác động TIÊU CỰC đến ROA (β={coeff:.4f}, p={p:.4f})")
                else:
                    interpretations.append(f"ℹ ESG không có tác động có ý nghĩa thống kê (β={coeff:.4f}, p={p:.4f})")

        return "\n".join(interpretations) if interpretations else "Không đủ dữ liệu để phân tích"


if __name__ == "__main__":
    model = PLS_SEM_Model()
    # Demo nhỏ: 5 DN ngân hàng + công nghệ
    model.load_data(companies=["ACB", "VCB", "MBB", "FPT", "VNM"], min_year=2016)
    model.run_sem()
    print(f"\n{model.interpret()}")
