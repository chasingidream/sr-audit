#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR 图像超分推理服务 · Python 客户端调用示例
==========================================
把一张低分辨率图片发给 SR 推理服务，取回超分后的图片并保存。

运行前先启动服务：
    python code/service/sr_infer_service.py --port 8766

本示例只用标准库 urllib（无需安装 requests）。
"""
import base64
import json
import sys
import urllib.request
from io import BytesIO

BASE = "http://127.0.0.1:8766"


def sr_image(image_bytes, compare=True):
    """调用 POST /sr，返回 (png_bytes, meta)。支持 multipart 或 JSON。"""
    # 方式一：multipart 表单（最简单）
    import io
    boundary = "----srclientboundary"
    body = (
        ("--%s\r\n" % boundary).encode()
        + b'Content-Disposition: form-data; name="image"; filename="in.png"\r\n'
        + b"Content-Type: image/png\r\n\r\n"
        + image_bytes
        + ("\r\n--%s--\r\n" % boundary).encode()
    )
    req = urllib.request.Request(
        BASE + "/sr?compare=%s" % ("1" if compare else "0"),
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        png = resp.read()
        meta = {}
        xmeta = resp.headers.get("X-SR-Meta")
        if xmeta:
            import base64 as _b64
            meta = json.loads(_b64.b64decode(xmeta).decode("utf-8"))
    return png, meta


def main():
    # 若命令行给了图片路径就用它，否则生成一张合成测试图（避免依赖外部文件）
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            img_bytes = f.read()
    else:
        from PIL import Image
        import numpy as np
        rng = np.random.default_rng(0)
        arr = (np.clip(rng.normal(0.5, 0.25, (3, 64, 64)), 0, 1) * 255).astype("uint8")
        arr = np.transpose(arr, (1, 2, 0))
        buf = BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()

    png, meta = sr_image(img_bytes, compare=True)
    out = "code/service/_sr_out.png"
    with open(out, "wb") as f:
        f.write(png)
    print("返回元信息:", meta)
    print("已保存超分结果:", out, "(%d bytes)" % len(png))
    print("提示：用任意图片查看器打开即可看到 [bicubic | 模型SR] 并排对比。")


if __name__ == "__main__":
    main()

# 方式二（等价）：用 base64 JSON 调用（适合无文件表单的场景）
#   payload = json.dumps({"image": base64.b64encode(img_bytes).decode(), "compare": True})
#   req = urllib.request.Request(BASE + "/sr", data=payload.encode(),
#       headers={"Content-Type": "application/json"}, method="POST")
