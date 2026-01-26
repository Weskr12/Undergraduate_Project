import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# ================= 設定區 (請修改這裡) =================
# 獲取目前腳本所在的目錄（相對路徑的基準點）
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # 上一層目錄是 Project

# 1. 你的模型路徑 (根據你剛才訓練成功的路徑)

#MODEL_PATH = SCRIPT_DIR / 'models' / 'Traffic_Sign_Project_1080p_v2' / 'single_class_run' / 'weights' / 'best.pt'

MODEL_PATH = SCRIPT_DIR / 'models' / 'model_best' / 'best_modelv1' / 'weights' / 'best.pt'


MODEL_PATH2 = SCRIPT_DIR / 'models' / 'model_20260124' / 'Traffic_Sign_Project' / 'single_class_run4' / 'weights' / 'best.pt'
# 2. 測試模式選擇
# 'image'  -> 測試單張圖片
# 'video'  -> 測試影片檔
# 'webcam' -> 使用鏡頭即時測試
MODE = 'video' 

# 3. 檔案路徑 (當 MODE 為 'image' 或 'video' 時需設定)
# 建議去你的 dataset/test/images 資料夾隨便複製一張圖片路徑過來試試



FILE_PATH = PROJECT_ROOT / 'main' / 'dataset' / 'sign1.mp4'
#FILE_PATH = r'C:/Users/User/Desktop/CW/Project/main/dataset/daytime_front_dashcam.mp4'
# 如果是測影片，改成影片路徑，例如：r'my_road_test.mp4'

# 4. 信心度門檻 (0.1 ~ 1.0)
# 如果traffic_sign很遠抓不到，把這個調低 (例如 0.25)
CONF_THRESHOLD = 0.2
# ======================================================                                

def main():
    # 1. 檢查模型是否存在

    if not MODEL_PATH.exists():
        print(f"❌ 錯誤：找不到模型檔案 -> {MODEL_PATH}")
        print("請確認路徑是否正確，或確認訓練是否真的完成了。")
        return

    print(f"✅ 正在載入模型：{MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))
    model2 = YOLO(str(MODEL_PATH2))

    # 2. 執行不同模式
    if MODE == 'image':
        if not FILE_PATH.exists():
            print(f"❌ 找不到測試圖片：{FILE_PATH}")
            return
            
        print(f"📷 正在測試圖片：{FILE_PATH}")
        # 進行預測
        results = model.predict(source=str(FILE_PATH), conf=CONF_THRESHOLD, save=True)
        
        # 顯示結果圖片
        for result in results:
            result.show()  # 這會直接彈出視窗顯示結果
            print(f"💾 結果已儲存至：{result.save_dir}")

    elif MODE == 'video' or MODE == 'webcam':
        source = '0' if MODE == 'webcam' else str(FILE_PATH)
        print(f"🎥 正在啟動{'鏡頭' if MODE == 'webcam' else '影片'}推論... (按 'q' 離開)")
        
        # 使用 OpenCV 進行自定義顯示迴圈
        cap = cv2.VideoCapture(int(source) if source == '0' else source)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("影片讀取結束或無法讀取鏡頭。")
                break

            # YOLO 推論
            results = model(frame, conf=CONF_THRESHOLD,imgsz=1280)
            
            # 將結果畫在畫面上
            annotated_frame = results[0].plot()
            
            # 顯示畫面
            cv2.imshow("model_1", annotated_frame)



            results2 = model2(frame, conf=CONF_THRESHOLD,imgsz=1280)
            annotated_frame2 = results2[0].plot()
            cv2.imshow("model_2", annotated_frame2)    
            
            
            # 按 'q' 鍵退出
        
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

                
            cv2.waitKey(10)

        

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()