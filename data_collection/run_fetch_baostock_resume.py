from multiprocessing import freeze_support
from pathlib import Path
import shutil
import socket
import time
import traceback

import baostock as bs
import pandas as pd
from tqdm import tqdm

from fetch_baostock_data import DataManager


def valid_pickle(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        pd.read_pickle(path)
    except Exception:
        path.unlink(missing_ok=True)
        return False
    return True


def relogin() -> None:
    try:
        bs.logout()
    except Exception:
        pass
    time.sleep(0.5)
    bs.login()


def copy_instruments(base: Path, target: Path) -> None:
    (target / "instruments").mkdir(parents=True, exist_ok=True)
    for p in (base / "instruments").glob("*.txt"):
        if p.name != "all.txt":
            shutil.copy2(p, target / "instruments" / p.name)


def main() -> None:
    socket.setdefaulttimeout(45)

    base = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    target = Path.home() / ".qlib" / "qlib_data" / "cn_data_baostock_alphagen"
    save_path = Path.cwd().parent / "data_baostock"
    k_data_path = save_path / "k_data"
    failed_path = save_path / "failed_downloads.txt"

    copy_instruments(base, target)

    dm = DataManager(
        save_path=str(save_path),
        qlib_export_path=str(target),
        qlib_base_data_path=str(base),
        adjust_date="2009-01-01",
        max_workers=2,
        max_retries=6,
        retry_wait_seconds=8.0,
    )
    dm._basic_info = pd.read_csv(save_path / "basic_info.csv", index_col=0)
    dm._adjust_factors = pd.read_csv(save_path / "adjust_factors.csv", index_col=[0, 1])

    k_data_path.mkdir(parents=True, exist_ok=True)
    codes = list(dm._basic_info.index)
    missing = [
        code for code in codes
        if not valid_pickle(k_data_path / f"{code}.pkl")
    ]
    print(f"Existing valid k_data: {len(codes) - len(missing)} / {len(codes)}")
    print(f"Missing k_data: {len(missing)}")

    failed = []
    relogin()
    for code in tqdm(missing, desc="Resume stock data"):
        try:
            dm._download_stock_data_job(code, dm._basic_info.loc[code])
        except Exception as exc:
            failed.append((code, repr(exc)))
            try:
                relogin()
            except Exception:
                pass
            continue
        time.sleep(0.2)

    try:
        bs.logout()
    except Exception:
        pass

    remaining = [
        code for code in codes
        if not valid_pickle(k_data_path / f"{code}.pkl")
    ]
    if failed or remaining:
        lines = [
            "Failed during this run:",
            *[f"{code}\t{err}" for code, err in failed],
            "",
            "Still missing after this run:",
            *remaining,
        ]
        failed_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Still missing: {len(remaining)}. See {failed_path}")
        return

    print("All stock data downloaded. Exporting CSV and dumping Qlib data.")
    for csv_path in (save_path / "export").glob("*.csv"):
        csv_path.unlink()
    for subdir in ("features", "calendars"):
        path = target / subdir
        if path.exists():
            shutil.rmtree(path)
    all_instruments = target / "instruments" / "all.txt"
    all_instruments.unlink(missing_ok=True)

    dm._save_csv()
    dm._dump_qlib_data()
    copy_instruments(base, target)
    dm._fix_constituents()
    print(f"Done: {target}")


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
