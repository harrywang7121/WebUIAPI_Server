## Fully operational @2024.3.30

import webuiapi
from PIL import Image
from flask import Flask, request, send_file
import requests


app = Flask(__name__)

def generate_image(img_path):
    api = webuiapi.WebUIApi()
    prompt = "a beautiful girl sticking tongue out"
    negative_prompt = "ugly"
    seed = 179053
    steps = 75
    img = Image.open(img_path)
    unit1 = webuiapi.ControlNetUnit(input_image=img,
                                    module='depth',
                                    model='control_v11f1p_sd15_depth [cfd03158]')
    result = api.txt2img(prompt=prompt,
                         negative_prompt=negative_prompt,
                         seed=seed,
                         steps=steps,
                         controlnet_units=[unit1])
    generated_img = result.image
    generated_img_path = 'generated_img.png'
    generated_img.save(generated_img_path)
    return generated_img_path


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = './receive_from_unity.png'
        file.save(filename)
        generated_img_path = generate_image(filename)
        generated_img = Image.open(generated_img_path).convert("RGB")
        generated_img.save(generated_img_path, 'PNG')

        with open(generated_img_path, 'rb') as f:
            response = requests.post('http://127.0.0.1:5001/process-image', files={'file': f})
            if response.status_code == 200:
                with open('final_output.obj', 'wb') as out:
                    out.write(response.content)
            else:
                print("Error:", response.text)

        return send_file(generated_img_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)