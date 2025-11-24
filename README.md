# PiPrime + HighNine 非自回归重排序系统（完全独立版本）

这是一个**完全独立**的文件夹，包含运行`test_piprime_highnine_100.py`所需的**所有**文件，无需任何外部依赖。

## 📁 完整文件结构

```
noneautorerank/
├── README.md                           # 本文件
├── QUICK_START.md                      # 快速开始指南
├── test_piprime_highnine_100.py       # 主测试脚本
├── piprime_with_mass_check.py         # PiPrime生成器（带质量检查）
├── piprime_reranker.py                # PiPrime重排序器
├── piprime_efficient_reranker.py      # 高效重排序器
├── test_beam_search_accuracy.py       # Beam Search测试
├── piprime_mass_calculator.py         # 质量计算模块
├── model_massive.ckpt                  # PiPrime模型文件（约1.5GB）
├── PrimeNovo/                          # PiPrime完整代码
│   ├── __main__.py
│   ├── config.yaml
│   ├── masses.py
│   ├── PrimeNovo.py
│   ├── utils.py
│   ├── utils2.py
│   ├── version.py
│   ├── components/                     # 模型组件
│   │   ├── __init__.py
│   │   ├── encoders.py
│   │   ├── feedforward.py
│   │   ├── mixins.py
│   │   └── transformers.py
│   └── denovo/                         # De novo测序模块
│       ├── ctc_beam_search.py
│       ├── ctc_beam_search_top_k.py
│       ├── ctc_decoder_base.py
│       ├── db_dataloader.py
│       ├── db_dataset.py
│       ├── db_index.py
│       ├── evaluate.py
│       ├── mass_con.py
│       ├── model.py
│       ├── model_runner.py
│       └── parser2.py
├── testdata/                           # 测试数据
│   └── high_nine_validation_1000_converted.mgf
└── test_data/                          # 索引数据
    └── high_nine/
        └── high_nine_database.mgf.efficient_index.pkl
```

## 🎯 功能说明

### 核心功能
- **非自回归生成**: 使用PiPrime的非自回归模型生成候选peptide
- **质量过滤**: 在Beam Search过程中进行precursor mass过滤
- **高效重排序**: 使用预建索引进行快速Database匹配
- **Prosit Fallback**: 对于Database中没有的peptide，使用Prosit预测
- **完全独立**: 所有代码、模型、数据都在此文件夹中

### 工作流程
1. **PiPrime Encoder**: 将query spectrum编码为embedding向量
2. **PiPrime Decoder**: 生成log probability matrix
3. **Beam Search**: 在probability matrix上进行beam search，生成1000条候选peptide
4. **质量过滤**: 过滤掉不符合precursor mass的候选
5. **重排序**: 
   - 优先使用Database中的参考谱图计算相似度
   - 如果Database中没有，使用Prosit预测谱图
6. **Top-1选择**: 选择相似度最高的peptide作为最终结果

## 📋 依赖要求

### Python包（需要安装）
```bash
pip install torch numpy pandas pyteomics requests pyyaml scikit-learn spectrum_utils
```

### 无需外部文件
✅ 所有代码、模型、数据都已包含在此文件夹中
✅ 无需配置路径
✅ 开箱即用

## 🚀 使用方法

### 1. 安装Python依赖
```bash
pip install torch numpy pandas pyteomics requests pyyaml scikit-learn spectrum_utils
```

### 2. 直接运行
```bash
cd C:\Users\research\Desktop\noneautorerank
python test_piprime_highnine_100.py
```

就这么简单！

### 3. 输出说明
脚本会创建一个带时间戳的输出目录，例如：
```
piprime_highnine_results_20231124_083000/
├── spectrum_0000.txt    # 第1个谱图的详细结果
├── spectrum_0001.txt    # 第2个谱图的详细结果
├── ...
└── spectrum_0999.txt    # 第1000个谱图的详细结果
```

每个文件包含：
- Spectrum信息（precursor m/z, charge, ground truth）
- Beam Search统计（总候选数、通过质量检查的数量）
- 所有候选peptide（按similarity排序）
- Ground Truth分析（是否在候选中、排名）
- Reranking结果（Top-1 peptide）

## 📊 预期结果

### 控制台输出示例
```
================================================================================
PiPrime + HighNine 集成测试
================================================================================
MGF文件: testdata/high_nine_validation_1000_converted.mgf
PiPrime模型: model_massive.ckpt
索引文件: test_data/high_nine/high_nine_database.mgf.efficient_index.pkl
输出目录: piprime_highnine_results_20231124_083000
测试规模: 1000个谱图
================================================================================

开始处理谱图...

已处理: 10/1000
  当前准确率: 6/10 = 60.00%
  平均候选数: 234.5

...

================================================================================
最终结果
================================================================================
总谱图数: 1000
正确预测: 613
准确率: 61.30%

平均候选数: 245.3

来源统计:
  Database: 523 (52.3%)
  Prosit: 431 (43.1%)
  Failed: 46 (4.6%)

详细结果保存在: piprime_highnine_results_20231124_083000/
================================================================================
```

## 🔧 高级配置

如果需要修改参数，编辑`test_piprime_highnine_100.py`：

### 修改测试规模
```python
# 第425行
for idx, spec in enumerate(reader):
    if idx >= 1000:  # 改为你想要的数量
        break
```

### 修改Beam Search参数
```python
# 第81-86行
self.piprime_predictor = PiPrimeWithMassCheck(
    self.piprime_model,
    precursor_mass_tol=50,      # 质量容差（ppm）
    isotope_error_range=(0, 1), # 同位素误差范围
    beam_width=100              # Beam宽度
)
```

### 修改候选数量
```python
# 第163行
candidates_raw = beam_search_with_mass_pruning(
    self.piprime_predictor,
    log_prob_matrix,
    precursor_mz,
    precursor_charge,
    target_count=1000,  # 目标候选数量
    top_n=10            # 每步Top-N
)
```

## 📊 性能指标

### 准确率
- **Top-1准确率**: 约61.30%（在1000个谱图上）
- **来源分布**:
  - Database匹配: ~52%
  - Prosit预测: ~43%
  - 失败: ~5%

### 速度
- **单个谱图处理时间**: 约2-5秒
- **1000个谱图总时间**: 约40-80分钟（取决于GPU性能）

### 候选peptide统计
- **平均候选数**: 约245条
- **通过质量检查**: 约60-70%
- **Database匹配率**: 约50-60%

## 🔍 关键技术细节

### 1. 质量计算
使用PiPrime的质量计算方法（`piprime_mass_calculator.py`）：
- 支持修饰：`C+57.021`, `M+15.995`, `N+0.984`等
- 质量包含H2O（18.010565 Da）
- Precursor mass公式：`M = (m/z - H) * z`

### 2. Beam Search策略
- **动态beam width**: 第1步10条，第2步100条，第3步及之后1000条
- **质量剪枝**: 超过precursor mass的路径被剪枝
- **CTC规约**: 去除重复的token

### 3. 重排序策略
- **Database优先**: 如果peptide在Database中，使用Top-3相似度的平均值
- **Prosit Fallback**: 如果Database中没有，使用Prosit_2025_intensity_MultiFrag预测
- **缓存优化**: 相同peptide只计算一次similarity

### 4. 修饰支持
- **PiPrime格式**: `C+57.021`, `M+15.995`
- **UNIMOD格式**: `C[UNIMOD:4]`, `M[UNIMOD:35]`
- **自动转换**: 在Database查找时自动转换格式

## ⚠️ 注意事项

1. **GPU内存**: 建议至少8GB GPU内存
2. **磁盘空间**: 模型文件约1.5GB，确保有足够空间
3. **网络连接**: Prosit预测需要访问Koina服务器（https://koina.wilhelmlab.org）
4. **修饰限制**: Prosit不支持修饰，会使用去除修饰后的序列进行预测

## 🐛 常见问题

### Q1: GPU内存不足
**A**: 修改`test_piprime_highnine_100.py`第407行：
```python
processor = PiPrimeHighNineProcessor(piprime_model, index_file, output_dir, device='cpu')
```

### Q2: Prosit预测失败
**A**: 检查网络连接，确保可以访问Koina服务器。如果网络不稳定，可以在第206行设置`use_prosit=False`。

### Q3: 找不到模块
**A**: 确保所有文件都在`noneautorerank`文件夹中，并且从该文件夹运行脚本。

### Q4: 模型加载失败
**A**: 确认`model_massive.ckpt`文件完整（约1.5GB）。如果文件损坏，需要重新复制。

## 📝 文件说明

### 核心脚本
- **test_piprime_highnine_100.py**: 主测试脚本，集成所有功能
- **piprime_with_mass_check.py**: PiPrime生成器，实现质量过滤的Beam Search
- **piprime_efficient_reranker.py**: 高效重排序器，使用预建索引和缓存
- **piprime_mass_calculator.py**: 质量计算模块，使用PiPrime的质量字典

### PrimeNovo代码
- **PrimeNovo/**: PiPrime的完整实现
  - **components/**: 模型组件（encoder, decoder等）
  - **denovo/**: De novo测序核心算法

### 数据文件
- **model_massive.ckpt**: 预训练的PiPrime模型
- **testdata/**: 1000个测试谱图
- **test_data/**: 预建的Database索引

## 📚 技术原理

### PiPrime vs Casanovo
- **Casanovo**: 自回归模型，逐个生成氨基酸
- **PiPrime**: 非自回归模型，一次性生成probability matrix
- **优势**: PiPrime更快，可以并行生成多条候选

### 非自回归的实现
1. Encoder将spectrum编码为向量
2. Decoder生成`[seq_len, vocab_size]`的probability matrix
3. 在matrix上进行Beam Search，而不是逐步生成
4. 使用CTC（Connectionist Temporal Classification）处理重复token

## 📧 支持

如有问题，请检查：
1. Python版本（建议3.8+）
2. PyTorch版本（建议1.10+）
3. GPU驱动（如果使用GPU）
4. 网络连接（Prosit需要）

---

**版本**: 2.0.0（完全独立版本）
**最后更新**: 2024-11-24
**文件夹大小**: 约2GB（包含模型和数据）