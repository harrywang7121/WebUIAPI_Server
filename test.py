import webuiapi
from PIL import Image
from flask import Flask, request, send_file
from gradio_client import Client
import shutil
import time
import os

app = Flask(__name__)

def SD_generate_image(img_path):
    api = webuiapi.WebUIApi()
    prompt = "a beautiful girl sticking tongue out"
    negative_prompt = "ugly"
    seed = 17053
    steps = 10
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
    generated_img_path = 'SD_generated_img.png'
    generated_img.save(generated_img_path)
    return generated_img_path

def move_obj_file(src, dst):
    new_path = os.path.join(dst, os.path.basename(src))
    try:
        shutil.move(src, new_path)
        print(f"OBJ file moved to {new_path}.")
        return new_path
    except Exception as e:
        print(f"Failed to move the OBJ file: {e}")
        return None

def SR_generate(image_path):
    preprocess_client = Client("stabilityai/TripoSR")
    preprocess_result = preprocess_client.predict(
        image_path,
        True,  # bool in 'Remove Background' Checkbox component
        0.85,  # float in 'Foreground Ratio' Slider component
        api_name="/preprocess"
    )
    print("Preprocess Result:", preprocess_result)

    generate_client = Client("stabilityai/TripoSR")
    generate_result = generate_client.predict(
        preprocess_result,
        256,
        api_name="/generate"
    )
    print("Generate Result:", generate_result)

    glb_file_path = generate_result[1]
    save_path = "output"  # 指定保存路径
    final_path = move_obj_file(glb_file_path, save_path)

    return final_path


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        print("Received request from Unity.")
        start_time = time.time()

        filename = './test.png'
        file.save(filename)
        generated_img_path = SD_generate_image(filename)
        model_path = SR_generate(generated_img_path)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total processing time: {total_time} seconds.")

        return send_file(model_path, as_attachment=True, download_name='model.glb')

if __name__ == '__main__':
    app.run(debug=True, port=5000)