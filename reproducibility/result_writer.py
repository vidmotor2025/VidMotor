import pandas as pd
from pathlib import Path
import numpy as np

class ResultWriter:
    """Excel result appending writer"""
    def __init__(self, excel_path):  # excel_path: Path to the Excel file
        self.excel_path = Path(excel_path)

    def write_result(self, dataset, model_name, accuracy, f1, auc, balanced_acc):
        cols = {
            f'{model_name}_Accuracy': accuracy,
            f'{model_name}_Balanced-accuracy': balanced_acc,
            f'{model_name}_F1-score': f1,
            f'{model_name}_AUROC': auc
        }
        # Read existing file or create new DataFrame
        if self.excel_path.exists():
            df = pd.read_excel(self.excel_path)
        else:
            df = pd.DataFrame(columns=['Dataset'])
        # Ensure dataset column exists
        if 'Dataset' not in df.columns:
            df['Dataset'] = []

        for col in cols.keys():
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(float)

        if dataset in df['Dataset'].values:
            idx_list = df.index[df['Dataset'] == dataset].tolist()
            written = False
            for idx in idx_list:
                can_write = True
                for key, value in cols.items():
                    if key in df.columns and pd.notna(df.at[idx, key]):
                        can_write = False
                        break
                if can_write:
                    for key, value in cols.items():
                        if key not in df.columns:
                            df[key] = np.nan
                        df.at[idx, key] = value
                    written = True
                    break
            if not written:
                new_row = {'Dataset': dataset, **{k: float(v) for k, v in cols.items()}}
                df.loc[len(df)] = new_row
        else:
            new_row = {'Dataset': dataset, **cols}
            df.loc[len(df)] = new_row

        df.to_excel(self.excel_path, index=False, float_format="%.4f")
        print(f"\t\t\tTesting finshed. Results written to {self.excel_path}: {dataset} - {model_name}")


class RatioResultWriter:
    """Excel result appending writer"""
    def __init__(self, excel_path):  # excel_path: Path to the Excel file
        self.excel_path = Path(excel_path)

    def write_result(self, dataset, model_name, auc):
        cols = {
            f'{model_name}_AUROC': auc
        }
        # Read existing file or create new DataFrame
        if self.excel_path.exists():
            df = pd.read_excel(self.excel_path)
        else:
            df = pd.DataFrame(columns=['Dataset'])
        # Ensure dataset column exists
        if 'Dataset' not in df.columns:
            df['Dataset'] = []

        for col in cols.keys():
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(float)

        if dataset in df['Dataset'].values:
            idx_list = df.index[df['Dataset'] == dataset].tolist()
            written = False
            for idx in idx_list:
                can_write = True
                for key, value in cols.items():
                    if key in df.columns and pd.notna(df.at[idx, key]):
                        can_write = False
                        break
                if can_write:
                    for key, value in cols.items():
                        if key not in df.columns:
                            df[key] = np.nan
                        df.at[idx, key] = value
                    written = True
                    break
            if not written:
                new_row = {'Dataset': dataset, **{k: float(v) for k, v in cols.items()}}
                df.loc[len(df)] = new_row
        else:
            new_row = {'Dataset': dataset, **cols}
            df.loc[len(df)] = new_row

        df.to_excel(self.excel_path, index=False, float_format="%.4f")
        print(f"\tTesting finshed. Results written to {self.excel_path}: {dataset} - {model_name}")


class TimeResultWriter:
    """Excel result appending writer about runtime"""
    def __init__(self, excel_path):  # excel_path: Path to the Excel file
        self.excel_path = Path(excel_path)

    def write_result(self, dataset, model_name, runtime):
        cols = {
            f'{model_name}_Runtime': runtime
        }
        # Read existing file or create new DataFrame
        if self.excel_path.exists():
            df = pd.read_excel(self.excel_path)
        else:
            df = pd.DataFrame(columns=['Dataset'])
        # Ensure dataset column exists
        if 'Dataset' not in df.columns:
            df['Dataset'] = []

        for col in cols.keys():
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(float)

        if dataset in df['Dataset'].values:
            idx_list = df.index[df['Dataset'] == dataset].tolist()
            written = False
            for idx in idx_list:
                can_write = True
                for key, value in cols.items():
                    if key in df.columns and pd.notna(df.at[idx, key]):
                        can_write = False
                        break
                if can_write:
                    for key, value in cols.items():
                        if key not in df.columns:
                            df[key] = np.nan
                        df.at[idx, key] = value
                    written = True
                    break
            if not written:
                new_row = {'Dataset': dataset, **{k: float(v) for k, v in cols.items()}}
                df.loc[len(df)] = new_row
        else:
            new_row = {'Dataset': dataset, **cols}
            df.loc[len(df)] = new_row

        df.to_excel(self.excel_path, index=False, float_format="%.4f")
        print(f"\tTesting finshed. Results written to {self.excel_path}: {dataset} - {model_name}")
