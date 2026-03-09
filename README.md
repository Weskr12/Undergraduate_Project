# 專題程式使用說明

這個 repo 目前分成兩個主題：

- `collision_avoidance/`：影像辨識防撞主程式
- `traffic_sign/`：路上標誌辨識相關程式

目前主要使用的是：

- `collision_avoidance/monitorVehicle.py`

## 專案結構

```text
Undergraduate_Project/
  collision_avoidance/
    monitorVehicle.py
  traffic_sign/
    test_my_model.py
    test_SAHI.py
  dataset/
  model/
    weights/
  checkpoints/
  output/
    videos/
    json/
    calibration/
  third_party/
    ml_depth_pro/
      src/
        depth_pro/
```

## 需要先下載的東西

執行 `collision_avoidance/monitorVehicle.py` 前，至少要準備這些檔案：

1. YOLO 權重

- 放置位置：`model/weights/best.pt`

2. Depth Pro 權重

- 放置位置：`checkpoints/depth_pro.pt`

3. 測試影片

- 放置位置：`dataset/`
- 例如：`dataset/test1.MP4`

說明：

- `third_party/ml_depth_pro/src/depth_pro/` 已經放了目前程式需要的最小 Depth Pro source
- 所以組員不需要再另外 clone 官方 `ml-depth-pro` repo
- 但 `depth_pro.pt` 權重檔仍然要自己下載後放進 `checkpoints/`

## 建議安裝環境

建議使用：

- Python 3.10 或 3.11
- NVIDIA GPU 可加速，但不是必要

先安裝 PyTorch：

- 如果你有 NVIDIA GPU，請到 PyTorch 官方網站選對 CUDA 版本安裝
- 如果你只想先在 CPU 跑，也可以直接安裝 CPU 版

再安裝其餘套件：

```powershell
pip install ultralytics opencv-python numpy pillow transformers accelerate
```

如果你的 PyTorch 還沒裝：

```powershell
pip install torch torchvision torchaudio
```

## 最簡單的執行方式

1. 把影片放到 `dataset/`
2. 打開 `collision_avoidance/monitorVehicle.py`
3. 修改 `DEFAULT_VIDEO_PATH`，指向你要跑的影片
4. 把檔案最底部的：

```python
RUN_FULL_PIPELINE = False
```

改成：

```python
RUN_FULL_PIPELINE = True
```

5. 在 repo 根目錄執行：

```powershell
python collision_avoidance/monitorVehicle.py
```

## 輸出結果會在哪裡

主要輸出會在：

- `output/videos/`：標註後影片
- `output/json/`：即時 JSON 與 JSONL

如果之後要用非 `depth_pro` 的 backend，才會用到：

- `output/calibration/depth_calibration.json`

## 目前程式的注意事項

1. 目前主程式還不是正式 CLI 工具，所以最簡單的方式仍然是直接修改 `DEFAULT_VIDEO_PATH`
2. `depth_pro.pt` 和測試影片都沒有放進 repo，組員需要自行補到指定資料夾
3. `output/` 目前只保留空資料夾骨架，不會把產生結果 push 上去
4. 如果用 GPU 跑，速度會明顯比 CPU 快

## 組員最少要做的事

如果只是要直接測試：

1. clone 這個 repo
2. 安裝 Python 套件
3. 把 `best.pt` 放到 `model/weights/`
4. 把 `depth_pro.pt` 放到 `checkpoints/`
5. 把影片放到 `dataset/`
6. 修改 `DEFAULT_VIDEO_PATH`
7. 執行 `python collision_avoidance/monitorVehicle.py`
