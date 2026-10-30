"""PPE detection inference with a YOLO .pt checkpoint or an exported ONNX model.

Usage:
    uv run main.py                                  # best.pt (YOLO26m), live playback
    uv run main.py --model ppe_yolov8m_export/ppe_yolov8m_best.onnx
    uv run main.py --source video/test-video.mp4 --save
    uv run main.py --source path/to/image.jpg --show
    uv run main.py --source 0                       # webcam

Press q or Esc to quit the live window.
"""

import argparse
import ast
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path("best.pt")
COLORS = [(0, 200, 255), (255, 120, 0), (0, 255, 0), (255, 0, 255),
          (0, 0, 255), (0, 128, 255), (255, 255, 0), (128, 0, 255)]  # BGR
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLODetector:
    """PyTorch .pt checkpoint (e.g. YOLO26m) via Ultralytics."""

    def __init__(self, model_path, conf=0.30, iou=0.7):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.names = [self.model.names[i] for i in sorted(self.model.names)]
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.conf = conf
        self.iou = iou

    def __call__(self, img):
        r = self.model.predict(img, conf=self.conf, iou=self.iou, device=self.device,
                               verbose=False)[0]
        boxes = r.boxes
        return list(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy().tolist(),
                        boxes.cls.cpu().numpy().astype(int).tolist()))


class ONNXDetector:
    """Exported YOLOv8 ONNX model (raw (1, 4+nc, 8400) output, NMS done here)."""

    def __init__(self, model_path, conf=0.30, iou=0.7):
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.device = self.session.get_providers()[0]
        names = self.session.get_modelmeta().custom_metadata_map.get("names")
        self.names = list(ast.literal_eval(names).values()) if names else []
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.h, self.w = self.session.get_inputs()[0].shape
        self.conf = conf
        self.iou = iou

    def letterbox(self, img):
        """Resize keeping aspect ratio and pad to model input size."""
        h0, w0 = img.shape[:2]
        r = min(self.h / h0, self.w / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        dw, dh = (self.w - nw) / 2, (self.h - nh) / 2
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (left, top)

    def preprocess(self, img):
        padded, r, pad = self.letterbox(img)
        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
        return x, r, pad

    def postprocess(self, output, r, pad, orig_shape):
        preds = output[0].T  # (8400, 4 + nc)
        scores = preds[:, 4:]
        class_ids = scores.argmax(1)
        confs = scores[np.arange(len(scores)), class_ids]
        keep = confs > self.conf
        preds, class_ids, confs = preds[keep], class_ids[keep], confs[keep]
        if not len(preds):
            return []

        # cx, cy, w, h (letterboxed) -> x1, y1, x2, y2 (original image)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        boxes -= [pad[0], pad[1], pad[0], pad[1]]
        boxes /= r
        h0, w0 = orig_shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        # Class-aware NMS: offset boxes per class so they never overlap across classes
        offset = class_ids[:, None] * 7680.0
        nms_boxes = boxes + offset
        xywh = np.column_stack([nms_boxes[:, :2], nms_boxes[:, 2:] - nms_boxes[:, :2]])
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), self.conf, self.iou)
        idx = np.array(idx).flatten()
        return [(boxes[i], float(confs[i]), int(class_ids[i])) for i in idx]

    def __call__(self, img):
        x, r, pad = self.preprocess(img)
        output = self.session.run(None, {self.input_name: x})[0]
        return self.postprocess(output, r, pad, img.shape)


def draw(img, detections, names):
    for box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[cls % len(COLORS)]
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_image(detector, path, out_dir, show):
    img = cv2.imread(str(path))
    dets = detector(img)
    out = out_dir / path.name
    cv2.imwrite(str(out), draw(img, dets, detector.names))
    print(f"{path.name}: {len(dets)} detections -> {out}")
    if show:
        cv2.imshow("PPE", img)
        cv2.waitKey(0)


def run_video(detector, source, out_dir, save):
    """Real-time playback: video advances at its native speed; if inference is
    slower than the video frame rate, frames are dropped to stay in sync."""
    is_cam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save:
        out = out_dir / ("webcam.mp4" if is_cam else Path(source).stem + "_pred.mp4")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cv2.namedWindow("PPE", cv2.WINDOW_NORMAL)
    frame_idx, shown, dropped = 0, 0, 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t = time.perf_counter()
        dets = detector(frame)
        infer_ms = (time.perf_counter() - t) * 1000
        draw(frame, dets, detector.names)
        cv2.putText(frame, f"infer {infer_ms:.0f} ms | {1000 / infer_ms:.1f} FPS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if writer:
            writer.write(frame)
        cv2.imshow("PPE", frame)
        shown += 1

        if is_cam:
            wait = 1  # webcam is already live
        else:
            # Where playback should be right now vs. where we are
            target_idx = int((time.perf_counter() - t0) * fps)
            if target_idx > frame_idx:  # behind: skip frames to catch up
                for _ in range(target_idx - frame_idx):
                    if not cap.grab():
                        break
                dropped += target_idx - frame_idx
                frame_idx = target_idx
                wait = 1
            else:  # ahead: wait until next frame is due
                wait = max(1, int((frame_idx / fps - (time.perf_counter() - t0)) * 1000))
        if cv2.waitKey(wait) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t0
    print(f"Showed {shown} frames in {elapsed:.1f}s ({shown / elapsed:.1f} FPS), "
          f"dropped {dropped} to stay real-time")


def main():
    parser = argparse.ArgumentParser(description="PPE detection (YOLO .pt or ONNX)")
    parser.add_argument("--source", default="video/test-video2.mp4",
                        help="image, folder of images, video file, or webcam index")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.455)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--out", default="output")
    parser.add_argument("--show", action="store_true", help="display image results in a window")
    parser.add_argument("--save", action="store_true", help="also save annotated video")
    args = parser.parse_args()

    Detector = ONNXDetector if args.model.endswith(".onnx") else YOLODetector
    detector = Detector(args.model, args.conf, args.iou)
    print(f"Loaded {args.model} on {detector.device} | classes: {detector.names}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                run_image(detector, p, out_dir, args.show)
    elif src.suffix.lower() in IMG_EXTS:
        run_image(detector, src, out_dir, args.show)
    else:
        run_video(detector, args.source, out_dir, args.save)


if __name__ == "__main__":
    main()
