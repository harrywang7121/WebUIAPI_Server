## Fully operational @2024.4.2

from flask import Flask
from gradio_client import Client
import shutil

app = Flask(__name__)

def move_obj_file(src, dst):
    try:
        shutil.move(src, dst)
        print(f"OBJ file moved to {dst}.")
    except Exception as e:
        print(f"Failed to move the OBJ file: {e}")

def preprocess_and_generate():
    preprocess_client = Client("stabilityai/TripoSR")
    preprocess_result = preprocess_client.predict(
        "generated_img.png",
        True,  # bool in 'Remove Background' Checkbox component
        0.85,  # float in 'Foreground Ratio' Slider component
        api_name="/preprocess"
    )
    print("Preprocess Result:", preprocess_result)

    generate_client = Client("stabilityai/TripoSR")
    generate_result = generate_client.predict(
        preprocess_result,
        128,
        api_name="/generate"
    )

    print("Generate Result:", generate_result)

    # 假设generate_result[0]是OBJ文件的路径
    obj_file_path = generate_result[0]
    save_path = "output"
    move_obj_file(obj_file_path, save_path)
    return generate_result


@app.route('/do', methods=['GET', 'POST'])
def do():
    result = preprocess_and_generate()
    return "Generate process completed."


if __name__ == '__main__':
    app.run(debug=True, port=5000)
