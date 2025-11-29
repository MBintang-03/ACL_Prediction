# api.py
import os
import tempfile
import json
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO

from Code_deploy import process_video

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Soccer Tracker API",
        "version": "1.0.0"
    }), 200


@app.route("/process-video", methods=["POST"])
def process_video_route():
    tmpdir = None
    try:
        # 1) Check and read uploaded video
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video_file = request.files["video"]

        # 2) Read bbox JSON from form-data "bboxes"
        raw_bboxes = request.form.get("bboxes")
        if not raw_bboxes:
            return jsonify({"error": "Missing 'bboxes' in form-data"}), 400

        try:
            data = json.loads(raw_bboxes)
            main_bbox = data["main_bbox"]
            other_bboxes = data.get("other_bboxes", [])
        except Exception as e:
            return jsonify({"error": "Invalid bboxes JSON", "details": str(e)}), 400

        if not main_bbox or len(main_bbox) != 4:
            return jsonify({"error": "main_bbox must be [x,y,w,h]"}), 400

        # 3) Save input and prepare output paths
        tmpdir = tempfile.mkdtemp(prefix="soccer_api_")
        input_path = os.path.join(tmpdir, "input.mp4")
        txt_output_path = os.path.join(tmpdir, "labels.txt")

        video_file.save(input_path)

        # 4) Call your processing function (returns AVI)
        output_avi_path, _ = process_video(
            input_path,
            txt_output_path,
            tuple(main_bbox),
            [tuple(b) for b in other_bboxes],
        )

        # 5) Verify AVI exists
        if not os.path.exists(output_avi_path):
            return jsonify({"error": "Processed video not found"}), 500

        size = os.path.getsize(output_avi_path)
        print(f"[DEBUG] AVI output: {output_avi_path}, size: {size} bytes")

        if size == 0:
            return jsonify({"error": "Processed video is empty"}), 500

        # 6) Convert AVI to MP4 using ffmpeg for browser compatibility
        output_mp4_path = os.path.join(tmpdir, "output.mp4")
        
        try:
            # Use ffmpeg to convert AVI to MP4 (H.264)
            subprocess.run([
                'ffmpeg', '-y',  # -y to overwrite
                '-i', output_avi_path,
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Compatibility
                output_mp4_path
            ], check=True, capture_output=True)
            
            print(f"[DEBUG] MP4 output: {output_mp4_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg conversion failed: {e.stderr.decode()}")
            # Fallback: try to serve AVI directly
            output_mp4_path = output_avi_path

        # 7) Read video into memory
        with open(output_mp4_path, 'rb') as f:
            video_data = f.read()
        
        print(f"[DEBUG] Sending {len(video_data)} bytes")

        # 8) Clean up temp files
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_avi_path):
                os.remove(output_avi_path)
            if os.path.exists(output_mp4_path) and output_mp4_path != output_avi_path:
                os.remove(output_mp4_path)
            if os.path.exists(txt_output_path):
                os.remove(txt_output_path)
            os.rmdir(tmpdir)
        except Exception as e:
            print(f"[WARN] Cleanup error: {e}")

        # 9) Return video from memory with MP4 MIME type
        return send_file(
            BytesIO(video_data),
            mimetype="video/mp4",
            as_attachment=False,
            download_name="processed.mp4",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if tmpdir and os.path.exists(tmpdir):
            try:
                import shutil
                shutil.rmtree(tmpdir)
            except:
                pass
        
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
