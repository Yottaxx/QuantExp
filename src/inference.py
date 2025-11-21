import torch
import pandas as pd
import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


def run_inference(top_k=10):
    device = Config.DEVICE
    model_path = f"{Config.OUTPUT_DIR}/final_model"

    print(f"加载模型: {model_path}")
    model = PatchTSTForStock.from_pretrained(model_path).to(device)
    model.eval()

    results = []
    files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))

    print("正在扫描全市场...")
    with torch.no_grad():
        for fpath in tqdm(files):
            try:
                stock_code = os.path.basename(fpath).replace('.parquet', '')
                df = pd.read_parquet(fpath)
                if len(df) < 100: continue

                # 处理数据，获取因子列
                df_proc, factor_cols = DataProvider.process_single_stock(df)

                # 取最后一段窗口
                last_window = df_proc.iloc[-Config.CONTEXT_LEN:]
                if len(last_window) != Config.CONTEXT_LEN: continue

                # 归一化
                scaler = StandardScaler()
                input_data = scaler.fit_transform(last_window[factor_cols].values)

                tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

                # 预测
                output = model(past_values=tensor)
                score = output.logits.item()

                results.append((stock_code, score))
            except:
                continue

    results.sort(key=lambda x: x[1], reverse=True)
    top_stocks = results[:top_k]

    print(f"\n【Top {top_k} 潜力股】")
    for code, score in top_stocks:
        print(f"股票: {code} | 预测得分: {score:.4f}")

    return top_stocks