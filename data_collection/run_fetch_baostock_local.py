from multiprocessing import freeze_support
from pathlib import Path
import shutil

from fetch_baostock_data import DataManager


def main() -> None:
    base = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    target = Path.home() / ".qlib" / "qlib_data" / "cn_data_baostock_alphagen"

    (target / "instruments").mkdir(parents=True, exist_ok=True)
    for p in (base / "instruments").glob("*.txt"):
        if p.name != "all.txt":
            shutil.copy2(p, target / "instruments" / p.name)

    dm = DataManager(
        save_path=str(Path.cwd().parent / "data_baostock"),
        qlib_export_path=str(target),
        qlib_base_data_path=str(base),
        adjust_date="2009-01-01",
        max_workers=4,
        max_retries=100,
        retry_wait_seconds=10.0,
    )

    dm.fetch_and_save_data()


if __name__ == "__main__":
    freeze_support()
    main()
