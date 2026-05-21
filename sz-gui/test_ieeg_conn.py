from ieeg.auth import Session
import os

u = os.environ.get("IEEG_USERNAME")
password = os.environ.get("IEEG_PASSWORD")
dataset_id = os.environ.get("IEEG_TEST_DATASET", "EMU0892_Day05_1")
if not u or not password:
    raise SystemExit("Set IEEG_USERNAME and IEEG_PASSWORD")

with Session(u, password) as s:
    ds = s.open_dataset(dataset_id)
    labels = list(ds.get_channel_labels())
    ch0 = labels[0]
    details = ds.get_time_series_details(ch0)

    print("channels", len(labels))
    print("sample_rate", details.sample_rate)
    print("n_samples", details.number_of_samples)

    s.close_dataset(ds)
