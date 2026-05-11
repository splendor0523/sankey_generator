# Sankey Generator

这是一个基于 Streamlit、Pandas 和 Matplotlib 的桑基图生成工具。  
程序可以从 Excel 表格读取数据，生成多列桑基图，并支持导出 PNG 和 PDF。

## 功能特点

- 从 Excel 文件导入数据
- 支持横向 / 竖向桑基图
- 支持节点颜色、连线颜色设置
- 支持节点标签字体、字号、颜色、加粗、斜体、下划线
- 支持中英文字体分离显示
- 支持节点标签自动避让
- 支持节点图例，图例可放在左侧、右侧或下侧
- 支持表头和图标题
- 支持节点描边、连线描边
- 支持 PNG / PDF 导出
- 支持 JSON 配置导入导出，方便复刻图形参数

## 项目结构

```text
sankey_generator/
├─ sankey_app.py
├─ s_engine.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ sankey_core/
   ├─ __init__.py
   ├─ colors.py
   ├─ graph.py
   ├─ layout.py
   └─ render.py
```

## 安装环境

建议使用 Python 3.9 或更高版本。

先进入项目文件夹：

```powershell
cd "D:\python file\sankey_generator"
```

安装依赖：

```powershell
pip install -r requirements.txt
```

## 启动程序

运行：

```powershell
streamlit run sankey_app.py
```

启动后，程序会自动在浏览器中打开。  
如果没有自动打开，可以在浏览器中访问终端里显示的本地地址，通常类似：

```text
http://localhost:8501
```

## Excel 数据说明

程序默认按成对列读取节点信息：

```text
第1列：第1层节点名称
第2列：第1层节点颜色
第3列：第2层节点名称
第4列：第2层节点颜色
……
```

后续权重列、配置列等请按照程序界面中的设置进行选择。

如果启用了“首行表头”功能，则 Excel 第 1 行会被解析为每列表头信息，不参与桑基图节点和连线计算。

## JSON 配置

程序支持导出和导入 JSON 配置文件。

JSON 配置可以保存当前图形的大部分参数，例如：

- 画布尺寸
- 节点样式
- 连线样式
- 字体设置
- 标签设置
- 图例设置
- 表头设置
- 标题设置
- 导出参数

这样在 Excel 数据不变的情况下，可以通过导入 JSON 配置复刻之前的图形效果。

## 不建议上传的文件

请不要把以下文件上传到 GitHub：

```text
客户真实 Excel 数据
导出的 PNG 图片
导出的 PDF 文件
本地虚拟环境 venv/
缓存文件 __pycache__/
.env
.streamlit/secrets.toml
```

这些文件已经建议写入 `.gitignore`。

## 依赖列表

主要依赖包括：

```text
streamlit
pandas
numpy
matplotlib
openpyxl
fonttools
```

完整依赖见：

```text
requirements.txt
```

## 开发说明

核心代码大致分为：

- `sankey_app.py`：Streamlit 前端界面和参数入口
- `s_engine.py`：辅助逻辑
- `sankey_core/graph.py`：图数据构建和配置结构
- `sankey_core/layout.py`：节点和连线布局计算
- `sankey_core/render.py`：Matplotlib 绘图和导出
- `sankey_core/colors.py`：颜色处理逻辑

## License

如果需要开源，可以后续添加 License 文件。
