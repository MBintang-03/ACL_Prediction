# api.py
import os

import json
import tempfile
import traceback
import shutil
import subprocess
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from Code_deploy import process_video

app = Flask(__name__)

# Allow requests from any origin (Vercel, localhost, etc.)
CORS(app, resources={r"/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "Soccer Tracker API",
            "version": "1.0.0",
        }
    ), 200


# ---------------------------------------------------------------------------
# Main processing endpoint
# ---------------------------------------------------------------------------
@app.route("/process-video", methods=["POST"])
def process_video_route():
    tmpdir = None

    try:
        # ---------------------- 1) Validate input ---------------------------
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video_file = request.files["video"]

        raw_bboxes = request.form.get("bboxes")
        if not raw_bboxes:
            return jsonify({"error": "Missing 'bboxes' in form-data"}), 400

        try:
            data = json.loads(raw_bboxes)
            main_bbox = data["main_bbox"]
            other_bboxes = data.get("other_bboxes", [])
        except Exception as e:
            return jsonify(
                {
                    "error": "Invalid 'bboxes' JSON",
                    "details": str(e),
                }
            ), 400

        if not isinstance(main_bbox, list) or len(main_bbox) != 4:
            return jsonify({"error": "main_bbox must be a list [x, y, w, h]"}), 400

        # ---------------------- 2) Temp paths -------------------------------
        tmpdir = tempfile.mkdtemp(prefix="soccer_api_")
        input_path = os.path.join(tmpdir, "input.mp4")
        txt_output_path = os.path.join(tmpdir, "labels.txt")

        video_file.save(input_path)

        # ---------------------- 3) Run processing ---------------------------
        # process_video should return a path to the processed video
        output_video_path, _ = process_video(
            input_path,
            txt_output_path,
            tuple(main_bbox),
            [tuple(b) for b in other_bboxes],
        )

        if not os.path.exists(output_video_path):
            return jsonify({"error": "Processed video not found"}), 500

        size = os.path.getsize(output_video_path)
        print(f"[DEBUG] Raw output video: {output_video_path}, size: {size} bytes")

        if size == 0:
            return jsonify({"error": "Processed video is empty"}), 500

        # ---------------------- 4) Ensure MP4 for browser -------------------
        final_path = output_video_path

        # If not MP4, try to convert with ffmpeg
        if not output_video_path.lower().endswith(".mp4"):
            mp4_path = os.path.join(tmpdir, "output.mp4")
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    output_video_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    mp4_path,
                ]
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"[DEBUG] ffmpeg stdout: {result.stdout.decode(errors='ignore')}")
                print(f"[DEBUG] ffmpeg stderr: {result.stderr.decode(errors='ignore')}")

                final_path = mp4_path
                print(f"[DEBUG] MP4 output: {final_path}")
            except subprocess.CalledProcessError as e:
                # If conversion fails, we just serve the original file
                print("[ERROR] FFmpeg conversion failed")
                print(e.stderr.decode(errors="ignore"))
                final_path = output_video_path

        # ---------------------- 5) Read file into memory --------------------
        with open(final_path, "rb") as f:
            video_data = f.read()

        print(f"[DEBUG] Sending {len(video_data)} bytes to client")

        # ---------------------- 6) Cleanup temp files -----------------------
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            if final_path != output_video_path and os.path.exists(final_path):
                os.remove(final_path)
            if os.path.exists(txt_output_path):
                os.remove(txt_output_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"[WARN] Cleanup error: {cleanup_err}")

        # ---------------------- 7) Return response --------------------------
        return send_file(
            BytesIO(video_data),
            mimetype="video/mp4",
            as_attachment=False,
            download_name="processed.mp4",
        )

    except Exception as e:
        # Log full traceback on server
        traceback.print_exc()

        # Best-effort cleanup
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

        return jsonify(
            {
                "error": "Internal server error",
                "details": str(e),
            }
        ), 500


if __name__ == "__main__":
    # For local testing only
    app.run(host="0.0.0.0", port=port, debug=True)
