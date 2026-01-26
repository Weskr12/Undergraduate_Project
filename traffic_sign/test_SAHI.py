import cv2
import os
import numpy as np
# 引入 SAHI 相關模組
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

# ================= 設定區 =================


MODEL_PATH = r'C:/Users/User/Desktop/CW/Project/traffic_sign/models/model_20260124/Traffic_Sign_Project_1080p_v1/single_class_run/weights/best.pt'

FILE_PATH = r'C:/Users/User/Desktop/CW/Project/main/dataset/sign3.mp4'
CONF_THRESHOLD = 0.25 # SAHI 對小物件比較敏感，可以稍微調高一點點，或者維持 0.2

# SAHI 切片設定 (關鍵參數)
SLICE_H = 480   # 切片大小，設為您模型訓練的大小
SLICE_W = 266
OVERLAP_RATIO = 0.5  # 重疊率，避免物件剛好被切一半
# =========================================
    
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 錯誤：找不到模型檔案 -> {MODEL_PATH}")
        return

    print(f"✅ 正在載入模型並設定 SAHI：{MODEL_PATH} ...")
    
    # 1. 初始化 SAHI 模型 (包裝您的 YOLO 模型)
    # model_type='yolov8' 是通用的，也適用於 YOLOv11 (因為都是 Ultralytics 框架)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device="cuda:0",  # 如果有顯卡一定要開，不然會跑不動！沒有顯卡改 "cpu"
    )

    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到影片：{FILE_PATH}")
        return

    print(f"🎥 正在啟動影片推論 (SAHI 模式)... 按 'q' 離開")
    cap = cv2.VideoCapture(FILE_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("影片讀取結束。")
            break

        # ================= SAHI 推論核心 =================
        # 原本是 results = model(frame)
        # 現在改成 get_sliced_prediction
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=SLICE_H,
            slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            perform_standard_pred=True,
            verbose=0 # 設為 0 不印出詳細 log，加快速度
            
        )

        # ================= 繪圖與顯示 =================
        # SAHI 自帶的視覺化工具
        # result.object_prediction_list 包含了所有偵測到的物件
        visualization_result = visualize_object_predictions(
            frame,
            object_prediction_list=result.object_prediction_list,
            rect_th=2, # 框框粗細
            text_size=0.5, # 文字大小
            text_th=1 # 文字粗細
            
        )
        
        # 取得繪製好的圖片
        annotated_frame = visualization_result["image"]

        # 注意：SAHI 預設處理可能是 RGB，但 OpenCV 是 BGR
        # 如果發現顏色怪怪的 (例如藍色變紅色)，請取消下面這行的註解：
        #annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("YOLO11 + SAHI Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()