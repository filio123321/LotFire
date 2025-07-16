from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
import cv2
from flask_socketio import SocketIO, emit
import numpy as np
import io
import os
import requests


def create_app():
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")

    # load YOLO once
    weights = os.getenv("YOLO_WEIGHTS", "best.pt")
    model = YOLO(weights)

    @app.route('/detect/image', methods=['POST'])
    def detect_image():
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        try:
            file = request.files['image']
            pil_img = Image.open(file).convert('RGB')
            # Get optional thresholds from form data
            conf = float(request.form.get('conf', 0.25))
            iou = float(request.form.get('iou', 0.45))
            imgsz = int(request.form.get('imgsz', 640))

            # Run inference
            results = model.predict(
                source=pil_img,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )

            # Annotate first result
            for r in results:
                im_array = r.plot()
                annotated = Image.fromarray(im_array[..., ::-1])
                break

            # Return image
            buf = io.BytesIO()
            annotated.save(buf, format='JPEG')
            buf.seek(0)
            return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/detect/url', methods=['POST'])
    def detect_url():
        data = request.get_json() or {}
        url = data.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            conf = float(data.get('conf', 0.25))
            iou = float(data.get('iou', 0.45))
            imgsz = int(data.get('imgsz', 640))

            results = model.predict(
                source=pil_img,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )

            for r in results:
                im_array = r.plot()
                annotated = Image.fromarray(im_array[..., ::-1])
                break

            buf = io.BytesIO()
            annotated.save(buf, format='JPEG')
            buf.seek(0)
            return send_file(buf, mimetype='image/jpeg')
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/detect/video', methods=['POST'])
    def detect_video():
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        file = request.files['video']
        tmp_in = '/tmp/input_video'
        file.save(tmp_in)

        cap = cv2.VideoCapture(tmp_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        frame_interval = int(fps)
        frame_idx = 0
        results_list = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    # Convert to PIL image
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)

                    # Run inference
                    detections = []
                    res = model.predict(
                        source=pil_img,
                        conf=0.25,
                        iou=0.45,
                        imgsz=640,
                    )
                    for r in res:
                        boxes = r.boxes
                        for box, conf_val, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                            x1, y1, x2, y2 = box.tolist()
                            detections.append({
                                'box': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                                'conf': round(float(conf_val), 3),
                                'class': model.names[int(cls)]
                            })
                    timestamp = frame_idx / fps
                    results_list.append(
                        {'time_sec': round(timestamp, 2), 'detections': detections})

                frame_idx += 1

            cap.release()
            os.remove(tmp_in)
            return jsonify({'frames': results_list})
        except Exception as e:
            cap.release()
            if os.path.exists(tmp_in):
                os.remove(tmp_in)
            return jsonify({'error': str(e)}), 500

    @socketio.on('frame')
    def handle_frame(frame_bytes: bytes, params: dict):
        """
        frame_bytes: the JPEG bytes you sent
        params:       { 'conf': number, 'iou': number, 'imgsz': number }
        """
        try:
            # decode the JPEG bytes into an OpenCV BGR frame
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            # run YOLO on it
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            results = model.predict(
                source=pil_img,
                conf=float(params.get('conf', 0.25)),
                iou=float(params.get('iou', 0.45)),
                imgsz=int(params.get('imgsz', 640))
            )
            im_array = results[0].plot()  # this is RGB numpy
            annotated = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

            # re‐encode to JPEG
            ok, jpeg = cv2.imencode('.jpg', annotated)
            if not ok:
                return

            emit('annotated_frame', jpeg.tobytes())
        except Exception as e:
            emit('error', {'message': str(e)})

    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    # use eventlet for concurrency
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)
