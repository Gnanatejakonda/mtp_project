from ultralytics import YOLO

# Load your PREVIOUSLY trained Nano model instead of the generic COCO one
model = YOLO('/home/gnanateja/drone_project/yolo_versions_check/runs/detect/drone_project_yolo26n/weights/best.pt')
print("🚀 Starting MAX PERFORMANCE training for YOLO Nano...")

# 2. Execute Training
results = model.train(
    data='/home/gnanateja/Downloads/dorne_data_set/labeled_indrones_dataset_2_fixed/data.yaml',
    epochs=100,           
    name='drone_project_yolo26n_fixed',
    device=0,             # Target your main GPU
    batch=-1,             # MAX GPU VRAM: AutoBatch
    cache=True,           # MAX SYSTEM RAM: Cache images in memory
    workers=16,           # MAX CPU: High-speed data loading
    amp=True              # Hardware acceleration for RTX architecture
)