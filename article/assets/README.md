# Diagram sources

本目录中的 `.excalidraw` 是系列四的可编辑图源：

- `figD0-runtime-to-dfx.excalidraw`
- `figD1-dfx-loop.excalidraw`
- `figD2-flight-recorder.excalidraw`
- `figD3-failure-propagation.excalidraw`

在 Excalidraw 中调整后，分别导出同名 PNG 到本目录，正文中的图片引用即可生效。需要恢复初始版本时运行：

```bash
python ../build_excalidraw.py
```

该命令会覆盖四个 `.excalidraw`，人工调整后不要再次运行，除非确认要恢复生成版。

