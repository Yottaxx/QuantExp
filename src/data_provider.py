import akshare as ak
import pandas as pd
import os
import glob
import numpy as np
import time
import random
import threading
import datetime
import concurrent.futures
import pickle
import warnings
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from .config import Config
from .vpn_rotator import vpn_rotator
from .alpha_lib import AlphaFactory
from pandarallel import pandarallel
import json
# 忽略 pandas 的性能警告
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 初始化并行计算 (利用多核加速 pandas apply)
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())




# =========================================================================
# Utils: 下载记录器与行业编码器
# =========================================================================

class DownloadRecorder:
    """记录下载失败的文件，支持导出供下次重试"""

    def __init__(self, log_path=None):
        self.log_path = log_path or os.path.join(Config.DATA_DIR, "download_failures.json")
        self._lock = threading.Lock()
        self.failed_tasks = {
            "price": [],
            "finance": [],
            "info": []
        }

    def log(self, category, code, reason):
        with self._lock:
            self.failed_tasks[category].append({
                "code": code,
                "reason": str(reason),
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    def save(self):
        """保存失败记录到 JSON"""
        if any(self.failed_tasks.values()):
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_tasks, f, indent=4, ensure_ascii=False)
            print(f"⚠️ 存在下载失败的任务，已记录至: {self.log_path}")
        else:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)  # 如果全部成功，清除旧日志


class IndustryEncoder:
    """持久化行业编码映射，保证 One-Hot/Embedding ID 的一致性"""

    def __init__(self, map_path=None):
        self.map_path = map_path or os.path.join(Config.DATA_DIR, "industry_map.json")
        self.mapping = self._load_mapping()
        self._lock = threading.Lock()

    def _load_mapping(self):
        if os.path.exists(self.map_path):
            with open(self.map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"Unknown": 0}  # 0 号位留给未知

    def save_mapping(self):
        with open(self.map_path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=4, ensure_ascii=False)
        print(f"✅ 行业编码表已更新: {self.map_path}")

    def encode(self, industries):
        """
        将行业列表转换为 ID，如果遇到新行业自动添加到映射表
        """
        ids = []
        is_updated = False

        # 预检查，避免频繁加锁
        current_keys = set(self.mapping.keys())
        new_industries = set(industries) - current_keys

        if new_industries:
            with self._lock:
                # 二次检查
                max_id = max(self.mapping.values()) if self.mapping else -1
                for ind in new_industries:
                    if ind not in self.mapping:
                        max_id += 1
                        self.mapping[ind] = max_id
                        is_updated = True

        if is_updated:
            self.save_mapping()

        # 转换
        return [self.mapping.get(i, 0) for i in industries]


# =========================================================================
# 1. NetworkManager: 网络与反爬对抗层
# =========================================================================
class NetworkManager:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    @staticmethod
    def setup_proxy_env():
        """配置系统代理环境变量，确保 akshare 请求走代理池"""
        proxy_url = Config.PROXY_URL
        if proxy_url:
            for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
                os.environ[k] = proxy_url

    @classmethod
    def safe_switch_vpn(cls):
        """线程安全的 VPN 切换逻辑 (冷却时间 5s)"""
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5: return
            try:
                print("🔄 [Network] 检测到反爬/封禁，正在切换 IP 线路 ...")
                vpn_rotator.switch_random()
            except Exception as e:
                print(f"⚠️ VPN 切换异常: {e}")
            cls._last_switch_time = time.time()
            time.sleep(3)  # 等待网络稳定


# =========================================================================
# 2. DataDownloader: ETL 层 (Extract, Transform, Load)
# =========================================================================

def to_em_symbol(x: str) -> str:
    s = (x or "").strip().upper()
    if s.endswith((".SZ", ".SH", ".BJ")):
        return s
    if x[0] in ("6", "9"):
        return f"{x}.SH"
    if x[0] in ("0", "3"):
        return f"{x}.SZ"
    if x[0] in ("8", "4"):
        return f"{x}.BJ"
    return f"{x}.SZ"




class DataDownloader:
    recorder = DownloadRecorder()  # 实例化记录器
    @staticmethod
    def _get_latest_trading_date():
        """获取最近一个交易日，用于判断数据是否需要更新"""
        try:
            df = ak.stock_zh_index_daily(symbol=Config.MARKET_INDEX_SYMBOL)
            return pd.to_datetime(df['date']).max().date().strftime("%Y-%m-%d")
        except:
            return datetime.date.today().strftime("%Y-%m-%d")

    @staticmethod
    def _read_parquet_safe(path):
        """
        [安全读取] 尝试读取 Parquet，如果文件损坏则删除并返回 None
        """
        if not os.path.exists(path): return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"⚠️ 发现损坏文件，已删除并准备重试: {path} ({e})")
            try:
                os.remove(path)
            except:
                pass
            return None

    @staticmethod
    def _fetch_pub_date_map(code):
        """
        获取财报公告日映射 (PIT Data Core)
        用于将财报数据对齐到其真实的发布日期，而非报告期末。
        """
        try:
            df = ak.stock_financial_abstract(symbol=code)
            if df is None or df.empty: return None
            df.columns = [c.strip() for c in df.columns]
            if '截止日期' in df.columns and '公告日期' in df.columns:
                res = df[['截止日期', '公告日期']].copy()
                res.columns = ['date', 'pub_date']
                res['date'] = pd.to_datetime(res['date'], errors='coerce')
                res['pub_date'] = pd.to_datetime(res['pub_date'], errors='coerce')
                return res.dropna()
        except:
            pass
        return None


    @staticmethod
    def _download_finance_worker(code):
        """
        下载财务数据（Eastmoney/AKShare: stock_financial_analysis_indicator_em）
        对齐 FundamentalPipeline 的稳定 schema：
          date, roe, rev_growth, profit_growth, debt_ratio, eps, bps, pub_date
        """
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir):
            os.makedirs(fund_dir)
        path = os.path.join(fund_dir, f"{code}.parquet")

        # 缓存检查: 3天内不重复下载
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) < 3 * 24 * 3600:
                return code, True, "Skipped"

        # 稳定 schema
        numeric_cols = ["roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps"]
        final_cols = ["date"] + numeric_cols + ["pub_date"]

        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.5, 1.2))  # 礼貌请求

                df = ak.stock_financial_analysis_indicator_em(symbol=to_em_symbol(code), indicator="按报告期")
                if df is None or df.empty:
                    return code, True, "Empty"

                # --- 1) report date -> date ---
                date_col = "REPORT_DATE" if "REPORT_DATE" in df.columns else (
                    "报告期" if "报告期" in df.columns else None)
                if date_col is None:
                    raise ValueError("Missing report date column (REPORT_DATE/报告期)")
                df["date"] = pd.to_datetime(df[date_col], errors="coerce")
                df.dropna(subset=["date"], inplace=True)

                # --- 2) pick fields (pipeline-style keys) ---
                pick = {
                    "roe": ("ROEJQ", "ROEKCJQ"),
                    "rev_growth": ("TOTALOPERATEREVETZ",),
                    "profit_growth": ("PARENTNETPROFITTZ",),
                    "debt_ratio": ("ZCFZL",),
                    "eps": ("EPSJB", "EPSJQ", "EPS"),
                    "bps": ("BPS",),
                }

                for k, cands in pick.items():
                    src = next((c for c in cands if c in df.columns), None)
                    df[k] = pd.to_numeric(df[src], errors="coerce") if src else np.nan

                # need debug

                # --- 3) YoY fallback (only when growth fully missing) ---
                df = df.sort_values("date")
                df = df.drop_duplicates(subset=["date"], keep="last")

                idx = pd.DatetimeIndex(df["date"])

                if df["rev_growth"].isna().all() and "TOTALOPERATEREVE" in df.columns:
                    cur = pd.to_numeric(df["TOTALOPERATEREVE"], errors="coerce").to_numpy(dtype=np.float64)
                    prev = pd.Series(cur, index=idx).reindex(idx - pd.DateOffset(years=1)).to_numpy(dtype=np.float64)
                    prev = np.where(prev == 0.0, np.nan, prev)
                    df["rev_growth"] = ((cur / prev - 1.0) * 100.0).astype(np.float32)

                if df["profit_growth"].isna().all() and "PARENTNETPROFIT" in df.columns:
                    cur = pd.to_numeric(df["PARENTNETPROFIT"], errors="coerce").to_numpy(dtype=np.float64)
                    prev = pd.Series(cur, index=idx).reindex(idx - pd.DateOffset(years=1)).to_numpy(dtype=np.float64)
                    prev = np.where(prev == 0.0, np.nan, prev)
                    df["profit_growth"] = ((cur / prev - 1.0) * 100.0).astype(np.float32)

                # --- 4) pub_date (pipeline: map > NOTICE/UPDATE > estimate) ---
                pub_df = DataDownloader._fetch_pub_date_map(code)
                if pub_df is not None and not pub_df.empty and "pub_date" in pub_df.columns:
                    df = pd.merge(df, pub_df[["date", "pub_date"]], on="date", how="left")
                elif "NOTICE_DATE" in df.columns:
                    df["pub_date"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce")
                elif "UPDATE_DATE" in df.columns:
                    df["pub_date"] = pd.to_datetime(df["UPDATE_DATE"], errors="coerce")
                else:
                    df["pub_date"] = pd.NaT

                # estimate pub_date for missing (no extra dependency, same idea as pipeline)
                miss = df["pub_date"].isna()
                if miss.any():
                    d = pd.to_datetime(df.loc[miss, "date"], errors="coerce")
                    y, m, day = d.dt.year, d.dt.month, d.dt.day
                    est = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
                    est.loc[(m == 3) & (day == 31)] = pd.to_datetime(y[(m == 3) & (day == 31)].astype(str) + "-04-30")
                    est.loc[(m == 6) & (day == 30)] = pd.to_datetime(y[(m == 6) & (day == 30)].astype(str) + "-08-31")
                    est.loc[(m == 9) & (day == 30)] = pd.to_datetime(y[(m == 9) & (day == 30)].astype(str) + "-10-31")
                    est.loc[(m == 12) & (day == 31)] = pd.to_datetime(
                        (y[(m == 12) & (day == 31)] + 1).astype(str) + "-04-30")
                    df.loc[miss, "pub_date"] = est

                # --- 5) finalize schema + types ---
                for c in numeric_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

                df = df[final_cols].copy()
                df = df.sort_values("date").set_index("date")
                df.to_parquet(path)

                return code, True, "Success"

            except Exception as e:
                err_str = str(e)
                if attempt < 4 and any(
                        k in err_str for k in ["404", "429", "502", "503", "Connection", "timed out", "NoneType"]):
                    NetworkManager.safe_switch_vpn()
                    time.sleep(2)
                    continue
                if attempt == 4:
                    print(f"⚠️ [Fail] {code} Finance: {e}")
                continue

        return code, False, "Failed"

    @staticmethod
    def _download_worker(code):
        """
        下载日频行情 (支持增量更新断点续传)
        """
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        start_date = Config.START_DATE
        old_df = None

        # --- 增量更新逻辑 ---
        if os.path.exists(path):
            try:
                old_df = pd.read_parquet(path)
                if not old_df.empty:
                    # 获取本地最新日期
                    last_date = old_df.index.max()
                    # 如果本地最新日期 >= 昨天，大概率不需要更新（这里简化判断，严谨可用 calendar）
                    if last_date.date() >= (datetime.date.today() - datetime.timedelta(days=1)):
                        return code, True, "Up-to-date"

                    # 设置新的下载起点 = 本地最后日期 + 1天
                    start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y%m%d")
            except Exception as e:
                print(f"⚠️ 文件损坏，重新下载: {code} ({e})")
                os.remove(path)
                old_df = None

        # 如果 start_date 超过了今天，说明不用更新
        if start_date > datetime.date.today().strftime("%Y%m%d"):
            return code, True, "Skipped"

        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.05, 0.2))

                # 下载增量数据
                new_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")

                if new_df is None or new_df.empty:
                    # 如果没有新数据，且本来就有老数据，视作成功
                    return code, True, "No New Data" if old_df is not None else "Empty"

                # 标准化列名
                new_df.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low',
                    '成交量': 'volume', '成交额': 'amount',
                    '换手率': 'turnover'
                }, inplace=True)
                new_df['date'] = pd.to_datetime(new_df['date'])
                new_df.set_index('date', inplace=True)

                # 类型转换
                cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']
                for col in cols:
                    if col in new_df.columns:
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').astype(np.float32)

                new_df.dropna(inplace=True)

                # 合并逻辑
                if old_df is not None:
                    # 过滤掉重叠日期 (以防万一)
                    new_df = new_df[new_df.index > old_df.index.max()]
                    if new_df.empty: return code, True, "Up-to-date"
                    final_df = pd.concat([old_df, new_df])
                else:
                    final_df = new_df

                if 'amount' in final_df.columns: final_df.drop(columns=['amount'], inplace=True)

                if not final_df.empty:
                    final_df.sort_index(inplace=True)
                    # 原子写入防止中断导致文件损坏
                    temp_path = path + ".tmp"
                    final_df.to_parquet(temp_path)
                    if os.path.exists(path): os.remove(path)
                    os.rename(temp_path, path)

                return code, True, "Success"

            except Exception as e:
                if attempt == 4:
                    # 记录失败
                    DataDownloader.recorder.log("price", code, e)
                NetworkManager.safe_switch_vpn()
                continue

        return code, False, "Failed"

    @staticmethod
    def _download_info_worker(code):
        """
        [新增] 下载个股静态信息（行业、上市日期、总市值）
        用于行业中性化 (Sector Neutralization) 和 股票池筛选
        """
        info_dir = os.path.join(Config.DATA_DIR, "info")
        if not os.path.exists(info_dir): os.makedirs(info_dir)
        path = os.path.join(info_dir, f"{code}.parquet")

        # 静态数据缓存 30 天
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) < 30 * 24 * 3600:
                return code, True, "Skipped"

        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.1, 0.3))
                df = ak.stock_individual_info_em(symbol=code)
                if df is None or df.empty: return code, True, "Empty"

                # 转置 kv 为 row
                info_dict = dict(zip(df['item'], df['value']))

                clean_data = {
                    'code': code,
                    'name': info_dict.get('股票简称', 'Unknown'),
                    'industry': info_dict.get('行业', 'Unknown'),
                    'list_date': str(info_dict.get('上市时间', '19900101')),
                    'total_mkt_cap': float(info_dict.get('总市值', 0))
                }

                res_df = pd.DataFrame([clean_data])
                res_df.to_parquet(path)
                return code, True, "Success"
            except Exception:
                NetworkManager.safe_switch_vpn()
                continue

        return code, False, "Failed"

    @staticmethod
    def _get_stock_list():
        """获取全市场股票列表 (带缓存)"""
        cache_file = os.path.join(Config.DATA_DIR, "stock_list.pkl")
        if os.path.exists(cache_file):
            # 12小时有效
            if time.time() - os.path.getmtime(cache_file) < 12 * 3600:
                print(f"⚡️ [Cache] 读取本地股票列表缓存")
                with open(cache_file, 'rb') as f: return pickle.load(f)

        print("🌐 [Network] 获取最新股票列表...")
        for attempt in range(10):
            try:
                stock_info = ak.stock_zh_a_spot_em()
                if stock_info is not None and not stock_info.empty:
                    codes = stock_info['代码'].tolist()
                    # 过滤非A股
                    codes = [c for c in codes if c.startswith(('00', '60', '30', '68'))]
                    with open(cache_file, 'wb') as f: pickle.dump(codes, f)
                    return codes
            except Exception:
                NetworkManager.safe_switch_vpn()
                time.sleep(2)

        # Fallback
        if os.path.exists(cache_file):
            print("⚠️ [Fallback] 使用旧缓存列表")
            with open(cache_file, 'rb') as f: return pickle.load(f)
        return []

    @staticmethod
    def run():
        """执行全量 ETL 任务"""
        print(">>> [ETL] 启动数据下载流水线...")
        NetworkManager.setup_proxy_env()
        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        codes = DataDownloader._get_stock_list()
        if not codes: return

        target_date_str = DataDownloader._get_latest_trading_date()

        # 1. 检查行情数据更新情况
        existing_fresh = set()
        for fname in os.listdir(Config.DATA_DIR):
            if fname.endswith(".parquet"):
                try:
                    mtime = os.path.getmtime(os.path.join(Config.DATA_DIR, fname))
                    if datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d") >= target_date_str:
                        existing_fresh.add(fname.replace(".parquet", ""))
                except:
                    pass

        todo_price = sorted(list(set(codes) - existing_fresh))
        print(f"📊 股票池: {len(codes)} | 待更新行情: {len(todo_price)}")

        # # 2. 下载行情 (Price)
        # if todo_price:
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        #         futures = {executor.submit(DataDownloader._download_worker, c): c for c in todo_price}
        #         for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo_price), desc="Price"): pass

        # 3. 下载财务 (Finance)
        print("同步财务数据...")
        # DataDownloader._download_finance_worker(codes[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(DataDownloader._download_finance_worker, c): c for c in codes}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Finance"): pass

        # 4. 下载静态信息 (Info/Industry)
        # print("同步行业静态信息...")
        # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        #     futures = {executor.submit(DataDownloader._download_info_worker, c): c for c in codes}
        #     for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Info"): pass

        print("✅ 数据同步完成。")


# =========================================================================
# 3. DataProcessor: 特征工程与数据合并层
# =========================================================================
class DataProcessor:
    @staticmethod
    def get_cache_path(mode, end_date=None):
        today_str = datetime.date.today().strftime("%Y%m%d")
        end_date_str = end_date.replace("-", "") if end_date else "latest"
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{end_date_str}_{today_str}.pkl")

    @staticmethod
    def _tag_universe(panel_df):
        """
        标记动态股票池
        过滤掉: 停牌(volume=0), 低价股(<2元),以此上市未满60天的次新股
        """
        print(">>> [Tagging] 标记 Universe...")
        panel_df = panel_df.sort_values(['code', 'date'])
        panel_df['list_days_count'] = panel_df.groupby('code')['date'].cumcount() + 1

        mask = (panel_df['volume'] > 0) & \
               (panel_df['close'] >= 2.0) & \
               (panel_df['list_days_count'] > 60)

        panel_df['is_universe'] = mask
        panel_df.drop(columns=['list_days_count'], inplace=True)
        return panel_df

    @staticmethod
    def process(mode='train', end_date=None, force_refresh=False):
        """
        构建训练数据的核心逻辑
        Steps: Read -> Merge Info -> Merge Fund(PIT) -> Time Cut -> Factor Calc -> CS Factors -> Cache
        """
        cache_path = DataProcessor.get_cache_path(mode, end_date)
        if not force_refresh and os.path.exists(cache_path):
            print(f"⚡️ [Cache Hit] {cache_path}")
            with open(cache_path, 'rb') as f: return pickle.load(f)

        print(f"\n>>> [Processing] 构建 Panel (Mode: {mode})...")

        # Paths
        price_files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        info_dir = os.path.join(Config.DATA_DIR, "info")

        # --- Step 1: 读取行情 (Parallel) ---
        def _read_price(f):
            try:
                df = pd.read_parquet(f)
                if isinstance(df.index, pd.DatetimeIndex) and 'date' not in df.columns:
                    df = df.reset_index()
                df['code'] = os.path.basename(f).replace(".parquet", "")
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_read_price, price_files), total=len(price_files), desc="Reading Price"))

        data_frames = [df for df in results if df is not None and not df.empty]
        if not data_frames: raise ValueError("无有效行情数据")
        panel_df = pd.concat(data_frames, ignore_index=True)
        del data_frames

        panel_df['code'] = panel_df['code'].astype(str)
        panel_df['date'] = pd.to_datetime(panel_df['date'])

        # --- Step 2: 严格的时间截断 (Time Travel Prevention) ---
        if end_date:
            print(f"✂️  执行时间截断: {end_date}")
            panel_df = panel_df[panel_df['date'] <= pd.to_datetime(end_date)]

        # --- Step 3: 读取并合并财务数据 (PIT Merge) ---
        fund_files = glob.glob(os.path.join(fund_dir, "*.parquet"))

        def _read_fund(f):
            try:
                df = pd.read_parquet(f)
                df['code'] = os.path.basename(f).replace(".parquet", "")
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            fund_frames = [df for df in executor.map(_read_fund, fund_files) if df is not None]

        if fund_frames:
            fund_df = pd.concat(fund_frames)
            fund_df = fund_df.reset_index().sort_values(['code', 'date'])

            # 填充发布日期
            if 'pub_date' in fund_df.columns:
                fund_df['merge_date'] = fund_df['pub_date']
                mask_na = fund_df['merge_date'].isna()
                # 默认延迟90天
                fund_df.loc[mask_na, 'merge_date'] = fund_df.loc[mask_na, 'date'] + pd.Timedelta(days=90)
            else:
                fund_df['merge_date'] = fund_df['date'] + pd.Timedelta(days=90)

            fund_df = fund_df.drop(columns=['date', 'pub_date'], errors='ignore')
            fund_df.rename(columns={'merge_date': 'date'}, inplace=True)

            # 财务数据同样需要截断
            if end_date: fund_df = fund_df[fund_df['date'] <= pd.to_datetime(end_date)]

            # Merge Asof (Point-in-Time)
            panel_df = panel_df.sort_values(['code', 'date'])
            fund_df = fund_df.sort_values(['code', 'date'])
            panel_df = pd.merge_asof(panel_df, fund_df, on='date', by='code', direction='backward')

            # 动态计算 PE/PB (使用合并后的 Price 和 EPS/BPS)
            print("计算估值指标 (PE/PB)...")
            panel_df['eps'] = panel_df['eps'].fillna(0)
            panel_df['bps'] = panel_df['bps'].fillna(0)

            panel_df['pe_ttm'] = np.where(panel_df['eps'] > 0.001, panel_df['close'] / panel_df['eps'], 0)
            panel_df['pb'] = np.where(panel_df['bps'] > 0.001, panel_df['close'] / panel_df['bps'], 0)

            # 处理 Inf
            panel_df.replace([np.inf, -np.inf], 0, inplace=True)

        if 'turnover' in panel_df.columns:
            panel_df['turnover'] = panel_df['turnover'].fillna(0).astype(np.float32)

        # --- Step 4: 合并行业信息 (Industry Merge with Persistent Mapping) ---
        print("合并行业信息并编码...")
        info_files = glob.glob(os.path.join(info_dir, "*.parquet"))

        # 初始化编码器
        ind_encoder = IndustryEncoder()  # 自动加载 data/industry_map.json

        def _read_info(f):
            try:
                return pd.read_parquet(f)
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            info_results = list(executor.map(_read_info, info_files))

        info_frames = [df for df in info_results if df is not None and not df.empty]

        if info_frames:
            info_df = pd.concat(info_frames, ignore_index=True)
            info_df['code'] = info_df['code'].astype(str)

            # 处理缺失行业
            info_df['industry'] = info_df['industry'].fillna('Unknown')

            # 这里的行业列表可能是 ['银行', '医药', '银行'...]
            # 使用编码器统一转换
            unique_industries = info_df['industry'].unique().tolist()
            # 预热编码器，确保所有行业都在 map 中
            ind_encoder.encode(unique_industries)

            # 应用映射
            # 建议使用 map 速度更快，encode 方法内部已处理了更新逻辑
            industry_map = ind_encoder.mapping
            info_df['industry_cat'] = info_df['industry'].map(industry_map).fillna(0).astype(int)

            panel_df = pd.merge(panel_df, info_df[['code', 'industry_cat']], on='code', how='left')
            # 对于 panel 中有但 info 中没有的股票，填充 Unknown (0)
            panel_df['industry_cat'] = panel_df['industry_cat'].fillna(0).astype(int)

        else:
            panel_df['industry_cat'] = 0

        # --- Step 5: 计算时序因子 (AlphaFactory) ---
        # 此时 panel_df 已经包含 OHLCV, Turnover, Industry, PE/PB
        if 'date' in panel_df.columns: panel_df = panel_df.set_index('date')
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("计算时序因子 (Parallel)...")
        panel_df = panel_df.groupby('code', group_keys=False).parallel_apply(lambda x: AlphaFactory(x).make_factors())

        # --- Step 6: 构造 Label (Target) ---
        print("构造 Labels...")
        panel_df['next_open'] = panel_df.groupby('code')['open'].shift(-1)
        panel_df['future_close'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN)
        panel_df['target'] = panel_df['future_close'] / panel_df['next_open'] - 1
        panel_df.drop(columns=['next_open', 'future_close'], inplace=True)

        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)

        # 标记 Universe
        panel_df = DataProcessor._tag_universe(panel_df)

        # --- Step 7: 截面处理与中性化 (Cross-Section) ---
        print("计算截面因子与中性化...")
        panel_df = panel_df.set_index('date')

        # 调用 AlphaFactory 的静态方法，进行行业/市场中性化
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        # --- Step 8: 保存 ---
        feature_cols = [c for c in panel_df.columns if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]

        panel_df = panel_df.reset_index()
        # 将数据转为 float32 节省空间
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

        with open(cache_path, 'wb') as f:
            pickle.dump((panel_df, feature_cols), f)

        return panel_df, feature_cols


# =========================================================================
# 4. DatasetBuilder: 数据集构建层
# =========================================================================
class DatasetBuilder:
    @staticmethod
    def build(panel_df, feature_cols):
        print(">>> [Dataset] Tensor Split (Train/Valid/Test)...")
        panel_df = panel_df.sort_values(['code', 'date']).reset_index(drop=True)

        feature_matrix = panel_df[feature_cols].values.astype(np.float32)
        # 使用 rank_label (0~1均匀分布) 作为训练目标，比 raw return 更稳定
        target_array = panel_df['rank_label'].fillna(0.5).values.astype(np.float32)

        universe_mask = panel_df['is_universe'].values
        dates = panel_df['date'].values
        codes = panel_df['code'].values

        # 快速计算每个股票的切分点
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        valid_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = Config.STRIDE

        # 生成合法的样本起始索引 (保证 seq_len 长度且属于 Universe)
        for start, end in zip(start_indices, end_indices):
            if end - start <= seq_len: continue
            for i in range(start + seq_len - 1, end, stride):
                if universe_mask[i]: valid_indices.append(i - seq_len + 1)

        valid_indices = np.array(valid_indices)

        # 按时间切分数据集
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        train_end_idx = int(n_dates * Config.TRAIN_RATIO)
        val_end_idx = int(n_dates * (Config.TRAIN_RATIO + Config.VAL_RATIO))

        train_date_limit = unique_dates[train_end_idx]
        val_start_date = unique_dates[min(train_end_idx + Config.CONTEXT_LEN, n_dates - 1)]
        val_date_limit = unique_dates[val_end_idx]
        test_start_date = unique_dates[min(val_end_idx + Config.CONTEXT_LEN, n_dates - 1)]

        sample_pred_dates = dates[valid_indices + seq_len - 1]

        idx_train = valid_indices[sample_pred_dates < train_date_limit]
        idx_valid = valid_indices[(sample_pred_dates >= val_start_date) & (sample_pred_dates < val_date_limit)]
        idx_test = valid_indices[sample_pred_dates >= test_start_date]

        print(f"Dataset Size: Train={len(idx_train)}, Valid={len(idx_valid)}, Test={len(idx_test)}")

        def create_gen(indices, shuffle=False):
            def _gen():
                if shuffle: np.random.shuffle(indices)
                for start_idx in indices:
                    yield {
                        "past_values": feature_matrix[start_idx: start_idx + seq_len],
                        "labels": target_array[start_idx + seq_len - 1]
                    }

            return _gen

        return DatasetDict({
            'train': Dataset.from_generator(create_gen(idx_train, shuffle=True)),
            'validation': Dataset.from_generator(create_gen(idx_valid, shuffle=False)),
            'test': Dataset.from_generator(create_gen(idx_test, shuffle=False))
        }), len(feature_cols)


# =========================================================================
# 5. DataProvider: 门面 (Facade)
# =========================================================================
class DataProvider:
    """
    统一入口类
    """

    @staticmethod
    def _get_cache_path(mode='train', end_date=None):
        return DataProcessor.get_cache_path(mode=mode,end_date=end_date)

    @staticmethod
    def download_data():
        """执行全量数据下载"""
        return DataDownloader.run()

    @staticmethod
    def load_and_process_panel(mode='train', end_date=None, force_refresh=False):
        """生成因子表"""
        return DataProcessor.process(mode=mode, end_date=end_date, force_refresh=force_refresh)

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        """生成张量数据集"""
        return DatasetBuilder.build(panel_df, feature_cols)


# 兼容旧接口
def get_dataset(force_refresh=False):
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features