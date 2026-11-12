from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("./YOLO-TS_TT100K.yaml")  # or model = YOLO("./best.pt")

    # Train the model
    model.train(data="./TT100K-2016.yaml", epochs=200, batch=48, imgsz=640, device='0,1,2,3')

    # Evaluate model performance on the validation set
    metrics = model.val(data="./TT100K-2016.yaml", imgsz=640, batch=1, device='0')

    # Export the model to ONNX format
    path = model.export(format="engine", device='0', half=True, opset=12)