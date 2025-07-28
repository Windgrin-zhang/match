from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


if __name__ == '__main__':
# Use the model
    model.train(data="A_cla.yaml", epochs=50,batch = 16)  # train the model
    metrics = model.val(data="A_cla.yaml")   # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format