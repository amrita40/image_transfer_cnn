from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import subprocess
import shutil

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER_CONTENT = 'content_dir'
UPLOAD_FOLDER_STYLE = 'style_dir'
STYLIZED_FOLDER = 'test_results/comm/stylized_results'


app.config['UPLOAD_FOLDER_CONTENT'] = UPLOAD_FOLDER_CONTENT
app.config['UPLOAD_FOLDER_STYLE'] = UPLOAD_FOLDER_STYLE
app.config['STYLIZED_FOLDER'] = STYLIZED_FOLDER


def save_image(file, folder, filename):
    file_path = os.path.join(app.config[folder], filename)
    file.save(file_path)
# 
def run_style_transfer(content_dir, style_dir):
    command = [
        'python3', 'main.py',
        '--type', 'test',
        '--batch_size', '1',
        '--comment', 'comm',
        '--content_dir', content_dir,
        '--style_dir', style_dir,
        '--num_workers', '1'
    ]
    subprocess.run(command)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        content_image = request.files['content_image']
        style_image = request.files['style_image']

        if content_image and style_image:
            save_image(content_image, 'UPLOAD_FOLDER_CONTENT', 'ci.jpg')
            save_image(style_image, 'UPLOAD_FOLDER_STYLE', 'si.jpg')

            run_style_transfer(UPLOAD_FOLDER_CONTENT, UPLOAD_FOLDER_STYLE)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

    return jsonify({'status': 'success', 'message': 'Images uploaded successfully'})

@app.route('/stylized')
def stylized():
    stylized_image_path = os.path.join(app.config['STYLIZED_FOLDER'], 'single_art_stylized.png')
    static_image_path = os.path.join(app.root_path, 'static', 'single_art_stylized.png')  # Construct path within static folder

    try:
        shutil.copyfile(stylized_image_path, static_image_path)  # Copy the image
    except Exception as e:
        print(f"Error copying image: {e}")
    print(static_image_path)
    return render_template('stylized.html', static_image_path=static_image_path)
    # print(stylized_image_path)
    # print("\n")
    # print(os.path.dirname(stylized_image_path))
    # print("\n")
    # print(os.path.basename(stylized_image_path))
    # send_from_directory(os.path.dirname(stylized_image_path), os.path.basename(stylized_image_path))
    return render_template('stylized.html', stylized_image_path=stylized_image_path)

@app.route('/camera')
def camera():
    return render_template('camera.html')


