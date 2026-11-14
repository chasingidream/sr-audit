"""EuroSAT RGB parquet -> ImageFolder 转换器（供 P2-1 下游验证用）。

源：HuggingFace `blanchon/EuroSAT_RGB` 的 3 个 parquet shard（train/test/validation），
经 hf-mirror.com 下载（见 experiments/RELEASE_MANIFEST 之外的本地 data/eurosat_rgb/raw/）。

每个 parquet 含列：image(struct{bytes,path}) / label(int) / filename(str)。
类别名直接从 filename 前缀取（如 "AnnualCrop_1050.tif" -> "AnnualCrop"），
与 label 整数映射无关，最稳健。

输出：<out_root>/<ClassName>/<split>_<idx>.png  —— 标准 ImageFolder 布局，
可直接喂 `p2_downstream_verify.load_labeled_eo`。

用法：
  python code/tools/convert_eurosat_parquet_to_imagefolder.py \
      --parquet_dir ~/.cache/eurosat_rgb/raw \
      --out data/eurosat_rgb/ImageFolder \
      [--limit_per_class 0]
"""
import argparse
import io
import os
import sys

from PIL import Image


def _image_bytes(cell):
    """从 parquet 的 image 单元格取出编码后的图像字节。"""
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is not None:
            return b
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    # pyarrow 可能返回嵌套结构，尝试递归取 bytes 字段
    if hasattr(cell, "get"):
        b = cell.get("bytes")
        if b is not None:
            return b
    raise TypeError("无法解析 image 单元格类型: %r" % type(cell))


def convert(parquet_dir, out_root, limit_per_class=0):
    import pyarrow.parquet as pq  # 懒加载，确保运行环境装了 pyarrow
    splits = {
        "train": "train.parquet",
        "test": "test.parquet",
        "validation": "valid.parquet",
    }
    per_class_count = {}
    total = 0
    for split, fname in splits.items():
        path = os.path.join(parquet_dir, fname)
        if not os.path.exists(path):
            print("[warn] 缺少 %s，跳过" % path)
            continue
        print("== 读取 %s ==" % path)
        table = pq.read_table(path)
        cols = table.column_names
        img_col = "image" if "image" in cols else cols[0]
        fn_col = "filename" if "filename" in cols else None
        n = table.num_rows
        print("   行数=%d 列=%s" % (n, cols))
        for i in range(n):
            img_cell = table.column(img_col)[i].as_py()
            raw = _image_bytes(img_cell)
            if fn_col is not None:
                filename = table.column(fn_col)[i].as_py() or ""
                cls = os.path.basename(filename).split("_")[0].split(".")[0]
            else:
                cls = "class_%d" % int(table.column("label")[i].as_py())
            if not cls:
                cls = "class_%d" % int(table.column("label")[i].as_py())
            # 限流：每类最多 limit_per_class（0=不限）
            c = per_class_count.get(cls, 0)
            if limit_per_class and c >= limit_per_class:
                continue
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                print("[warn] 解码失败 row %d split %s: %s" % (i, split, e))
                continue
            dst_dir = os.path.join(out_root, cls)
            os.makedirs(dst_dir, exist_ok=True)
            im.save(os.path.join(dst_dir, "%s_%06d.png" % (split, i)))
            per_class_count[cls] = c + 1
            total += 1
    print("== 完成 == 共写出 %d 张；各类：" % total)
    for k in sorted(per_class_count):
        print("   %-22s %d" % (k, per_class_count[k]))
    return per_class_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True, help="含 train/test/valid .parquet 的目录")
    ap.add_argument("--out", required=True, help="ImageFolder 输出根目录")
    ap.add_argument("--limit_per_class", type=int, default=0, help="每类最多写几张(0=不限)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert(args.parquet_dir, args.out, args.limit_per_class)


if __name__ == "__main__":
    main()
