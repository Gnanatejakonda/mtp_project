from ultralytics import YOLO
import os

def evaluate_models():
    # 1. Updated with the exact paths where your fixed models finished
    models_to_test = {
        "YOLOv26_Nano": "F:/backup_drones_dataset/drone_project/drone_project/yolo_versions_check/runs/detect/drone_project_yolo26n_fixed-2/weights/best.pt",
        "YOLOv26_Small": "F:/backup_drones_dataset/drone_project/drone_project/yolo_versions_check/runs/detect/train/weights/best.pt"
    }

    # Use the absolute path to your fixed dataset to be perfectly safe
    yaml_path = '/home/gnanateja/Downloads/dorne_data_set/labeled_indrones_dataset_2_fixed/data.yaml'

    for model_name, weight_path in models_to_test.items():
        print(f"\n==================================================")
        print(f"🚀 Starting Evaluation for: {model_name}")
        print(f"==================================================")
        
        if not os.path.exists(weight_path):
            print(f"❌ Error: Could not find weights at {weight_path}. Skipping.")
            continue

        model = YOLO(weight_path)

        # 2. Run the validation process
        metrics = model.val(
            data=yaml_path,
            split='val',     
            imgsz=1280,      # CRITICAL: Ensures it grades on the high-def grid!
            conf=0.25,       
            iou=0.6,         
            plots=True       
        )

        # 3. Print the core metrics in Percentages
        print(f"\n--- 📊 Final Results for {model_name} ---")
        # We multiply by 100 and format with .2f to get 2 decimal places (e.g., 86.79%)
        print(f"Mean Average Precision (mAP50-95): {metrics.box.map * 100:.2f}%")
        print(f"mAP at 50% IoU (mAP50):            {metrics.box.map50 * 100:.2f}%")
        print(f"Precision:                         {metrics.box.mp * 100:.2f}%")
        print(f"Recall:                            {metrics.box.mr * 100:.2f}%")
        
        print(f"\n📁 Full visual report saved to: {metrics.save_dir}")

if __name__ == "__main__":
    evaluate_models()