import pandas as pd
import os

train_data_path = "data_train.csv"
val_data_path = "data_val.csv"
test_data_path = "data_test.csv"
target_file_path = "output_data/Osaka_v4_dstip_id_allinfo.txt"

output_test = "output_data/Osaka_v4_dstip_id_allinfo_test.txt"
output_train = "output_data/Osaka_v4_dstip_id_allinfo_train.txt"
output_val = "output_data/Osaka_v4_dstip_id_allinfo_val.txt"


def load_ip_set(file_path, ip_col="IP"):
    try:
        df = pd.read_csv(file_path, usecols=[ip_col], encoding="utf-8", on_bad_lines="skip")
        ip_set = set(df[ip_col].dropna().astype(str).str.strip())
        print(f"Read {len(ip_set)} unique IPs from {os.path.basename(file_path)}")
        return ip_set
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return set()


if __name__ == "__main__":
    os.makedirs("output_data", exist_ok=True)
    
    train_ips = load_ip_set(train_data_path)
    val_ips = load_ip_set(val_data_path)
    test_ips = load_ip_set(test_data_path)
    
    try:
        target_df = pd.read_csv(
            target_file_path,
            header=None,
            encoding="utf-8",
            on_bad_lines="skip",
            dtype={0: str}
        )
        target_df.columns = ["ip"] + [f"col_{i}" for i in range(1, target_df.shape[1])]
        print(f"\nRead {len(target_df)} records from target file")
    except Exception as e:
        print(f"Failed to read target file {target_file_path}: {e}")
        exit(1)
    

    target_df["dataset"] = "none"
    
    target_df.loc[target_df["ip"].isin(val_ips), "dataset"] = "val"
    target_df.loc[target_df["ip"].isin(train_ips) & (target_df["dataset"] == "none"), "dataset"] = "train"
    target_df.loc[target_df["ip"].isin(test_ips) & (target_df["dataset"] == "none"), "dataset"] = "test"
    
    test_data = target_df[target_df["dataset"] == "test"].drop("dataset", axis=1)
    test_data.to_csv(output_test, index=False, header=False, encoding="utf-8")
    print(f"Saved test set: {len(test_data)} records -> {output_test}")
    
    train_data = target_df[target_df["dataset"] == "train"].drop("dataset", axis=1)
    train_data.to_csv(output_train, index=False, header=False, encoding="utf-8")
    print(f"Saved training set: {len(train_data)} records -> {output_train}")
    
    val_data = target_df[target_df["dataset"] == "val"].drop("dataset", axis=1)
    val_data.to_csv(output_val, index=False, header=False, encoding="utf-8")
    print(f"Saved validation set: {len(val_data)} records -> {output_val}")
    
    none_data = target_df[target_df["dataset"] == "none"]
    if len(none_data) > 0:
        print(f"\nWarning: {len(none_data)} records have IPs that did not match any dataset")

print("\nSplit completed!")
