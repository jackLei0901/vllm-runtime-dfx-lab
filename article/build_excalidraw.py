from __future__ import annotations

import json
import zlib
from pathlib import Path


ASSETS = Path(__file__).with_name("assets")
NOW = 1787760000000


def base(element_id: str, element_type: str, x: float, y: float, w: float, h: float):
    return {
        "id": element_id,
        "type": element_type,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": zlib.crc32(element_id.encode("utf-8")),
        "version": 1,
        "versionNonce": zlib.crc32((element_id + "v").encode("utf-8")),
        "isDeleted": False,
        "boundElements": [],
        "updated": NOW,
        "link": None,
        "locked": False,
    }


def rect(element_id: str, x: float, y: float, w: float, h: float, color: str):
    item = base(element_id, "rectangle", x, y, w, h)
    item["backgroundColor"] = color
    item["roundness"] = {"type": 3}
    return item


def text(element_id: str, x: float, y: float, value: str, size: int = 20, width: float = 220):
    lines = value.splitlines() or [""]
    height = len(lines) * size * 1.25
    item = base(element_id, "text", x, y, width, height)
    item.update(
        {
            "fontSize": size,
            "fontFamily": 2,
            "text": value,
            "originalText": value,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": None,
            "lineHeight": 1.25,
            "baseline": int(size * 0.9),
            "autoResize": True,
        }
    )
    return item


def arrow(element_id: str, x: float, y: float, dx: float, dy: float):
    item = base(element_id, "arrow", x, y, abs(dx), abs(dy))
    item.update(
        {
            "points": [[0, 0], [dx, dy]],
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "elbowed": False,
        }
    )
    return item


def box(elements: list[dict], prefix: str, x: float, y: float, w: float, h: float, label: str, color: str):
    elements.append(rect(prefix + "r", x, y, w, h, color))
    elements.append(text(prefix + "t", x + 12, y + 22, label, 18, w - 24))


def save(name: str, elements: list[dict]):
    ASSETS.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    (ASSETS / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def runtime_to_dfx():
    e: list[dict] = [text("title", 160, 30, "图 D0  从正常执行路径进入 Runtime DFX", 26, 900)]
    labels = [
        ("请求入口", "#d0ebff"),
        ("Scheduler", "#d3f9d8"),
        ("Model Runner", "#fff3bf"),
        ("GPU Kernel", "#ffe8cc"),
        ("输出 token", "#e5dbff"),
    ]
    for i, (label, color) in enumerate(labels):
        x = 70 + i * 220
        box(e, f"b{i}", x, 150, 170, 90, label, color)
        if i < len(labels) - 1:
            e.append(arrow(f"a{i}", x + 170, 195, 50, 0))
    box(e, "sym", 220, 340, 240, 115, "线上现象\n慢、卡住、OOM、进程死亡", "#ffc9c9")
    box(e, "ev", 700, 340, 260, 115, "诊断证据\n状态历史、故障域、退出语义", "#c5f6fa")
    e.append(arrow("down", 545, 240, -120, 100))
    e.append(arrow("right", 460, 398, 240, 0))
    e.append(text("q", 455, 480, "不只问“在哪报错”，还要回答“故障前发生了什么”", 20, 440))
    save("figD0-runtime-to-dfx.excalidraw", e)


def dfx_loop():
    e: list[dict] = [text("title", 190, 25, "图 D1  Runtime DFX 的五步闭环", 26, 820)]
    labels = [
        ("Detect\n发现异常", "#ffc9c9"),
        ("Capture\n保留现场", "#d0ebff"),
        ("Diagnose\n缩小故障域", "#fff3bf"),
        ("Recover\n验证恢复", "#d3f9d8"),
        ("Verify\n固化回归", "#e5dbff"),
    ]
    positions = [(90, 160), (350, 100), (650, 160), (570, 390), (230, 390)]
    for i, ((label, color), (x, y)) in enumerate(zip(labels, positions)):
        box(e, f"b{i}", x, y, 210, 100, label, color)
    arrows = [
        ("a0", 300, 190, 50, -35),
        ("a1", 560, 150, 90, 35),
        ("a2", 730, 260, -55, 130),
        ("a3", 570, 440, -130, 0),
        ("a4", 230, 420, -60, -160),
    ]
    e.extend(arrow(*item) for item in arrows)
    e.append(text("center", 390, 285, "证据必须可复现\n结论必须有边界", 22, 340))
    save("figD1-dfx-loop.excalidraw", e)


def flight_recorder():
    e: list[dict] = [text("title", 170, 25, "图 D2  外部 flight recorder 的采集与触发", 26, 880)]
    sources = ["/health", "/metrics", "/proc / PID", "nvidia-smi"]
    for i, label in enumerate(sources):
        box(e, f"s{i}", 60, 115 + i * 105, 190, 70, label, "#d0ebff")
        e.append(arrow(f"sa{i}", 250, 150 + i * 105, 130, 0))
    box(e, "ring", 380, 185, 270, 210, "有界环形历史\n最近 N 个样本\n\n固定内存上限\n只记录低基数字段", "#fff3bf")
    e.append(arrow("rt", 650, 290, 130, 0))
    box(e, "trigger", 780, 135, 280, 130, "触发条件\nprocess exit / health lost\nKV pressure / preemption storm", "#ffc9c9")
    box(e, "artifact", 780, 345, 280, 130, "Incident artifact\n环境 + 时间线 + 注入事件\n原子 JSON + Markdown 摘要", "#d3f9d8")
    e.append(arrow("ta", 920, 265, 0, 80))
    e.append(text("privacy", 380, 445, "默认不采集 prompt / token IDs / schema；写入前再次脱敏", 18, 380))
    save("figD2-flight-recorder.excalidraw", e)


def propagation():
    e: list[dict] = [text("title", 160, 25, "图 D3  TP worker 故障到 supervisor 的传播边界", 26, 900)]
    labels = [
        ("VllmWorker-0\nSIGKILL", "#ffc9c9"),
        ("EngineCore\n检测 worker 死亡", "#ffe8cc"),
        ("EngineClient\nEngineDeadError", "#fff3bf"),
        ("HTTP Server\nteardown", "#d0ebff"),
        ("顶层进程\nexit 1", "#d3f9d8"),
        ("Supervisor\n触发重启", "#e5dbff"),
    ]
    for i, (label, color) in enumerate(labels):
        x = 45 + i * 195
        box(e, f"b{i}", x, 160, 155, 110, label, color)
        if i < len(labels) - 1:
            e.append(arrow(f"a{i}", x + 155, 215, 40, 0))
    e.append(text("normal", 235, 350, "正常 SIGTERM：signal event 先被记录 → cleanup → exit 0", 20, 430))
    e.append(text("boundary", 690, 350, "DP supervisor 是另一条父进程生命周期\n不能由 TP=2 结果直接外推", 20, 390))
    save("figD3-failure-propagation.excalidraw", e)


if __name__ == "__main__":
    runtime_to_dfx()
    dfx_loop()
    flight_recorder()
    propagation()
    print(f"wrote diagrams to {ASSETS}")
