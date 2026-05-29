from ultralytics import YOLO
import torch

def train_drone_suite():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 RESUMING Training starting on device: {device}")

    print(f"\n{'='*50}")
    print(f"  RESUMING: yolo26s (Small Model)")
    print(f"{'='*50}\n")
    
    # 1. Load the exact save-state from 4:40 AM
    model = YOLO('/home/gnanateja/drone_project/yolo_versions_check/runs/detect/drone_project_yolo26s_fixed-2/weights/last.pt') 

    # 2. Tell YOLO to pick up exactly where it left off
    model.train(resume=True)
        
    print(f"✅ Finished training YOLO Small!")

if __name__ == '__main__':
    train_drone_suite()