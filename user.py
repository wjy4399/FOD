from gradio_client import Client, handle_file
import os
def process_images_in_folder(client, folder_path):
    # 获取文件夹中的所有图片文件
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in image_files:
        try:
            result = client.predict(
                image=handle_file(image_file),
                api_name="/predict"
            )
            print(f"Results for {image_file}:")
            print(result)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
if __name__ == "__main__":
    client = Client("https://05f4372e8de7e0cb92.gradio.live/")
    folder_path = r'D:\project\FOD\FOD\yolov5\datasets\test\images'
    process_images_in_folder(client, folder_path)