## 目录结构

- `Untitled2.ipynb`: 主流程（参数配置、训练入口、可视化）
- `pvlib/config.py`: 超参数与随机种子
- `pvlib/data.py`: 数据加载、增强、DataLoader
- `pvlib/models.py`: 模型定义与工厂函数 `get_model`
- `pvlib/train.py`: 训练/验证/曲线/预测可视化
- `pvlib/compression.py`: 剪枝、量化、压缩函数

## 安装依赖

```bash
pip install -r requirements_modular.txt
```

## 常见问题

- 若 `from pvlib...` 报错，通常是当前工作目录不在 `project.ipynb` 同级目录。
- 可在 Notebook 开头加入：

```python
import os
os.chdir(r"plantvillage dataset")
```

- 若 `inception_resnet_v2` 报错，安装 `timm`：

```bash
pip install timm
```
