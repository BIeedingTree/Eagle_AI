from flask import Flask, render_template, send_from_directory, request, redirect, url_for
import os
import json
import exifread
import io
import zipfile
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

IMAGE_FOLDER = os.path.join('static', 'images')
UPLOAD_FOLDER = os.path.join('static', 'input')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_metadata(filepath):
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        lat_ref = tags.get("GPS GPSLatitudeRef")
        lat = tags.get("GPS GPSLatitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        lon = tags.get("GPS GPSLongitude")
        dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")

        def dms_to_dd(dms, ref):
            d = float(dms.values[0].num) / float(dms.values[0].den)
            m = float(dms.values[1].num) / float(dms.values[1].den)
            s = float(dms.values[2].num) / float(dms.values[2].den)
            dd = d + m / 60 + s / 3600
            return dd if ref in ["N", "E"] else -dd

        if lat and lon and lat_ref and lon_ref:
            latitude = dms_to_dd(lat, str(lat_ref))
            longitude = dms_to_dd(lon, str(lon_ref))
        else:
            latitude = "N/A"
            longitude = "N/A"

        timestamp = str(dt) if dt else "N/A"
        return latitude, longitude, timestamp

    except Exception:
        return "N/A", "N/A", "N/A"

@app.route('/')
def index():
    images = []
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(IMAGE_FOLDER, filename)
            lat, lon, timestamp = extract_metadata(path)
            images.append({
                'filename': filename,
                'lat': lat,
                'lon': lon,
                'timestamp': timestamp
            })
    image_count = len(images)
    return render_template('index.html', images=images, image_count=image_count)

@app.route('/images/<filename>')
def image_file(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    files = request.files.getlist('file')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

    return redirect(url_for('index'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/download-zip')
def download_zip():
    zip_buffer = io.BytesIO()
    images = []
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(IMAGE_FOLDER, filename)
            lat, lon, timestamp = extract_metadata(path)
            
            if lat == "N/A" or lon == "N/A":
                continue
            images.append({
                'path': path,
                'filename': filename,
                'lat': lat,
                'lon': lon,
                'timestamp': timestamp
            })

    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        gps_info = '\n'.join([
            f"{img['filename']} | {img['lat']}, {img['lon']} | {img['timestamp']}" for img in images
        ])
        zipf.writestr("coordinates.txt", gps_info)
        for img in images:
            zipf.write(img['path'], arcname=img['filename'])

    zip_buffer.seek(0)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return (
        zip_buffer.read(),
        200,
        {
            'Content-Type': 'application/zip',
            'Content-Disposition': f'attachment; filename=drone_data_{date_str}.zip'
        }
    )

@app.route('/ar')
def ar_view():
    images = []
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.jpg','jpeg','.png')):
            path = os.path.join(IMAGE_FOLDER, filename)
            lat, lon, timestamp = extract_metadata(path)
            # nur Bilder mit gültigen Koordinaten
            if lat == "N/A" or lon == "N/A":
                continue
            images.append({
                'filename': filename,
                'lat': lat,
                'lon': lon,
                'timestamp': timestamp
            })
    # hier übergibst du die Liste an dein Template
    return render_template('ar.html', images=images)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

