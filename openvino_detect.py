import cv2
import numpy as np
import os

from openvino.runtime import Core, Layout

# Follows the defaults in ultralytics repo
BOX_THRESH = 0.25
CLASS_THRESH = 0.25
NMS_THRESH = 0.45
NMS_SCORE_THRESH = BOX_THRESH * CLASS_THRESH

def into2owh(x):
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # x origin
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # y origin
    return x

def resize_to_frame(imraw):
    major_dim = np.max(imraw.shape)
    scale = 320 / major_dim
    outscale = 1 / scale
    imraw = cv2.resize(imraw, None, fx=scale, fy=scale)
    img = np.zeros((320, 320, 3), dtype=imraw.dtype)
    img[: imraw.shape[0], : imraw.shape[1], :] = imraw
    return img, outscale

def process_yolo_output_tensor(tensor):
    tensor = tensor.squeeze()

    best_score = np.max(tensor[:, 5:], axis=1)
    # Where the best score >= the class thresh
    valid = best_score >= CLASS_THRESH
    tensor = tensor[valid]

    class_ids = np.argmax(tensor[:, 5:], axis=1)
    boxes = into2owh(tensor[:, :4])
    confidences = tensor[:, 4:5].squeeze() * best_score[valid]

    nms_res = cv2.dnn.NMSBoxes(boxes, confidences, NMS_SCORE_THRESH, NMS_THRESH)

    return (
        nms_res,
        boxes,
        confidences,
        class_ids,
    )

class YoloOpenVinoDetector:
    def __init__(self, openvino_dir, backend="AUTO"):
        model = None
        weights = None
        meta = None
        mapping = None
        files = os.listdir(openvino_dir)
        for x in files:
            if x.endswith(".xml"):
                model = "{}/{}".format(openvino_dir, x)
            elif x.endswith(".bin"):
                weights = "{}/{}".format(openvino_dir, x)
            elif x.endswith(".yaml"):
                meta = "{}/{}".format(openvino_dir, x)
            elif x.endswith(".mapping"):
                mapping = "{}/{}".format(openvino_dir, x)

        self.scale = 1.0

        self.ie = Core()
        self.network = self.ie.read_model(model=model, weights=weights)

        if self.network.get_parameters()[0].get_layout().empty:
            self.network.get_parameters()[0].set_layout(Layout("NCHW"))

        self.executable_network = self.ie.compile_model(
            self.network, device_name=backend
        )

    def detect(self, im):  # img is a np array
        im, self.scale = resize_to_frame(im)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        y = list(self.executable_network([blob]).values())

        (nms_res, boxes, confidences, class_ids) = process_yolo_output_tensor(y[0])

        res = []
        for idx in nms_res:
            conf = confidences[idx]
            classnm = class_ids[idx]
            x, y, w, h = np.clip(boxes[idx], 0, 320).astype(np.uint32)
            d = (x, y, x + w, y + h)  # xyxy format
            corners = np.array(((d[0], d[1]), (d[0], d[3]), (d[2], d[3]), (d[2], d[1])))

            res.append(
                {
                    "type": "yolov5",
                    "id": classnm,
                    "color": (0, 255, 0),
                    "corners": corners * self.scale,
                    "confidence": conf,
                }
            )
        return res

# To convert the model into openvino format either...
#
# Export first to onnx in the yolov5 repo
# > python export.py --weights /path/to/your/weights.pt --include onnx --opset 12
# Then pass the onnx file to mo (included in openvino-dev pip package)
# > mo --input_model /path/to/your/weights.onnx --data_type FP16
#
# (untested) Or convert directly to openvino. Might fail if host lacks a gpu?
# > python export.py --weights /path/to/your/weights.pt --include openvino --device 0 --half
import sys
if __name__ == '__main__':
    model_dir = ""
    im = None
    if len(sys.argv) >= 3:
        model_dir = sys.argv[1]
        im = cv2.imread(sys.argv[2])

        # CV2 uses BGR natively, but YOLO wants RGB.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        print("Usage: python openvino_detect.py /path/to/your/weights/ /path/to/some/image.jpg")
        print("weights should be a folder containing an openvino model.")
        print("Weights must have at least a .bin and .xml file")
        exit(1)

    # Do this only once
    det = YoloOpenVinoDetector(model_dir)

    # Call this in a loop
    res = det.detect(im)
    print(res)
