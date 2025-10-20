#!/usr/bin/env python3
"""
YOLO Server for TouchDesigner
Real-time object detection via Syphon (video) and OSC (detection data)
Supports YOLOv8, v9, v10, and v11

Architecture:
- Receives video from TouchDesigner via Syphon
- Runs YOLO inference on each frame
- Sends annotated video back via Syphon
- Sends detection data (bboxes, classes, scores) via OSC
"""

import cv2
import numpy as np
import syphon
from syphon.utils.numpy import copy_image_to_mtl_texture, copy_mtl_texture_to_image
from syphon.utils.raw import create_mtl_texture
import Metal
import time
from pythonosc import udp_client
from ultralytics import YOLO
import torch

print("=" * 70)
print("YOLO Server → Syphon + OSC")
print("=" * 70)

# Configuration
WIDTH, HEIGHT = 1920, 1080
SYPHON_INPUT_NAME = "TD Video Out"  # Receive from TouchDesigner
SYPHON_OUTPUT_NAME = "YOLO Detections"  # Send to TouchDesigner
OSC_SEND_IP = "127.0.0.1"
OSC_SEND_PORT = 7000  # Send detections to TD

# YOLO Configuration
YOLO_MODEL = "yolov8n.pt"  # nano, s, m, l, x (n=fastest, x=most accurate)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\n[Config] Video Input: Syphon '{SYPHON_INPUT_NAME}'")
print(f"[Config] Detection Output: Syphon '{SYPHON_OUTPUT_NAME}'")
print(f"[Config] OSC Send: {OSC_SEND_IP}:{OSC_SEND_PORT}")
print(f"[Config] YOLO Model: {YOLO_MODEL}")
print(f"[Config] Device: {DEVICE}")
print(f"[Config] Confidence: {CONFIDENCE_THRESHOLD}, IOU: {IOU_THRESHOLD}")

# Load YOLO model
print(f"\n[YOLO] Loading model {YOLO_MODEL}...")
print(f"[YOLO] First run will download the model...")
model = YOLO(YOLO_MODEL)
model.to(DEVICE)
print(f"[YOLO] ✓ Model loaded on {DEVICE}")

# Create Syphon client for input
print(f"\n[Syphon] Looking for input server '{SYPHON_INPUT_NAME}'...")
directory = syphon.SyphonServerDirectory()
matching = directory.servers_matching_name(name=SYPHON_INPUT_NAME)

if not matching:
    print(f"\n[ERROR] Syphon server '{SYPHON_INPUT_NAME}' not found!")
    print("\nTo fix:")
    print("1. Open TouchDesigner")
    print("2. Create a Syphon Out TOP")
    print(f"3. Set Server Name to: '{SYPHON_INPUT_NAME}'")
    print("4. Connect a video source")
    exit(1)

server_desc = matching[0]
print(f"[Syphon] Found server: '{server_desc.name}' from '{server_desc.app_name}'")
client = syphon.SyphonMetalClient(server_desc)
print(f"[Syphon] ✓ Input client created")

# Create Syphon server for output
print(f"[Syphon] Creating output server '{SYPHON_OUTPUT_NAME}'...")
server = syphon.SyphonMetalServer(SYPHON_OUTPUT_NAME)
print(f"[Syphon] ✓ Output server created")

# Create OSC client for sending detections
print(f"\n[OSC] Creating send client {OSC_SEND_IP}:{OSC_SEND_PORT}...")
osc_client = udp_client.SimpleUDPClient(OSC_SEND_IP, OSC_SEND_PORT)
print(f"[OSC] ✓ Send client created")

# Create Metal device and texture for output
print(f"\n[Metal] Creating output texture ({WIDTH}x{HEIGHT})...")
metal_device = Metal.MTLCreateSystemDefaultDevice()
output_texture = create_mtl_texture(metal_device, WIDTH, HEIGHT)
print(f"[Metal] ✓ Output texture created")

print("\n[Ready] Waiting for Syphon input...")
print("=" * 70)
print("TouchDesigner Setup:")
print(f"  1. Syphon Out TOP - Server: '{SYPHON_INPUT_NAME}'")
print(f"  2. Syphon In TOP - Server: '{SYPHON_OUTPUT_NAME}'")
print(f"  3. OSC In CHOP - Port: {OSC_SEND_PORT}")
print("=" * 70)
print("Press Ctrl+C to quit\n")

frame_count = 0
detection_count = 0
last_frame = None
last_annotated = None

try:
    while True:
        # Get frame from Syphon
        if client.has_new_frame:
            syphon_frame = client.new_frame_image

            if syphon_frame is not None:
                # Convert Metal texture to numpy
                frame_bgra = copy_mtl_texture_to_image(syphon_frame)

                # Convert to BGR for OpenCV/YOLO
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

                # Flip vertically (Syphon coordinate system)
                frame_bgr = cv2.flip(frame_bgr, 0)

                last_frame = frame_bgr.copy()

                # Run YOLO inference
                results = model(
                    frame_bgr,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )

                # Get annotated frame with bounding boxes
                # YOLO plot() returns RGB format (not BGR like OpenCV)
                annotated_frame = results[0].plot()

                # Send detection data via OSC
                detections = results[0].boxes
                num_detections = len(detections)

                # Send number of detections
                osc_client.send_message("/yolo/count", num_detections)

                # Send individual detection data
                for i, box in enumerate(detections):
                    # Get bounding box coordinates (normalized 0-1)
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = xyxy / [WIDTH, HEIGHT, WIDTH, HEIGHT]

                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = model.names[cls]

                    # Send OSC messages for this detection
                    osc_client.send_message(f"/yolo/{i}/class", class_name)
                    osc_client.send_message(f"/yolo/{i}/confidence", conf)
                    osc_client.send_message(f"/yolo/{i}/bbox", [x1, y1, x2, y2])

                # Flip back for Syphon output
                annotated_frame = cv2.flip(annotated_frame, 0)

                # Convert to BGRA for Syphon
                # Note: YOLO plot() returns RGB format (not BGR like OpenCV)
                annotated_bgra = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGRA)

                # Copy to Metal texture and publish
                copy_image_to_mtl_texture(annotated_bgra, output_texture)
                server.publish_frame_texture(output_texture)

                last_annotated = annotated_bgra.copy()

                frame_count += 1
                detection_count = num_detections

                # Status update
                if frame_count % 100 == 0:
                    print(f"[Status] Frames: {frame_count}, Detections: {detection_count}")

        else:
            # No new frame, republish last frame if we have one
            if last_annotated is not None:
                copy_image_to_mtl_texture(last_annotated, output_texture)
                server.publish_frame_texture(output_texture)

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n\nShutting down...")

print(f"\nTotal frames processed: {frame_count}")
print("Done!")
