from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 model (choose 'yolov8n', 'yolov8s', 'yolov8m', etc.)
model = YOLO('yolov9t.yaml')  # 'n' = nano, for faster, smaller model

# 2. Train on your custom dataset
model.train(
    data='./data/data.yaml',  # path to your dataset config file
    epochs=250,               # number of training epochs
    imgsz=640,                # image resolution
    batch=16,                 # batch size
    name='yolov9t',   # experiment name
    patience=10,              # early stopping if no improvement
    project='runs/train',     # output directory
    task='detect',
    workers=0,
)

# 3. Validate the model (optional, runs automatically after training)
metrics = model.val()

# 4. Inference (run on a test image or folder)
results = model.predict(
    source='./data/test/images',    # path to image folder or image
    save=True,                # save predictions
    conf=0.5                  # confidence threshold
)
