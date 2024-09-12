import cv2
from ultralytics import YOLO
from io import BytesIO
import pytesseract
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

# Token bot Telegram
TELEGRAM_BOT_TOKEN = '6946770589:AAFpzUX3dgYKF6MMjHuaT6VAvlmXpv859CM'
CHAT_ID = '1185669068'
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

# Path video file atau URL stream
video_path = 'rtsp://admin:admin@192.168.31.204:8554/Streaming/Channels/101'
#output_video_path = r'C:\Users\maula\Downloads\Compressed\yolov8\output\motor.mp4'

# Membuka video atau gambar
cap = cv2.VideoCapture(video_path)

# Mengatur ukuran buffer menjadi 3 frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# Tentukan ukuran frame output yang diinginkan
#output_width = 1280
#output_height = 720
output_width = 1920
output_height = 1080

# Model YOLOv8
model = YOLO(r"C:\Users\maula\Downloads\Compressed\yolov8\dataset\runs1000\detect\train\weights\best1000.pt")

# menyimpan video
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk video
#fps = 30  # Frame per second dari video output
#out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

async def kirim_tele(image, label):
    #await asyncio.sleep(delay)  # Menambahkan delay sebelum mengirim pesan
    _, image_encoded = cv2.imencode('.jpg', image)
    image_bytes = image_encoded.tobytes()
    bio = BytesIO(image_bytes)
    data = aiohttp.FormData()
    data.add_field('chat_id', CHAT_ID)
    data.add_field('caption', label)
    data.add_field('photo', bio, filename='image.jpg')

    async with aiohttp.ClientSession() as session:
        async with session.post(TELEGRAM_API_URL, data=data) as response:
            if response.status != 200:
                print(f"Error sending photo: {response.status}")

def plat(plat_image):
    MAX_OCR_LENGTH = 9
    results = model(plat_image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            if f'{model.names[cls]}' == 'PlatNomor' and conf > 0.3:
                crop_image = plat_image[y1:y2, x1:x2]

                skala = 5.0
                c_h = int(crop_image.shape[0] * skala)
                c_w = int(crop_image.shape[1] * skala)
                pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
                ri = cv2.resize(crop_image, (c_w, c_h), interpolation=cv2.INTER_AREA)
                img_gray = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
                _, threshold = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
                #edges = cv2.Canny(threshold, 180, 210)

                ocr_result = pytesseract.image_to_string(threshold)
                ocr_result = ocr_result.upper()

                first_line = ocr_result.split('\n')[0]
                result = ''.join([char for char in first_line if char.isalnum()])

                #cv2.imshow("Crop Plat", crop_image)
                #cv2.imshow("Threshold", threshold)
                #cv2.imshow("edges", edges)
                #if cv2.waitKey(500000):
                #    break

                if result:
                    return 'Plat nomor ' + result

    return None

def Pengendara(image):
    results = model(image)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            if f'{model.names[cls]}' == 'NoHelm' and conf > 0.3:
                nomor_plat = plat(image)
                if nomor_plat:
                    tasks.append(loop.create_task(kirim_tele(image, 'Pengendara tidak memakai helm, ' + nomor_plat)))
                else:
                    tasks.append(loop.create_task(kirim_tele(image, "Plat Nomor Tidak Terdeteksi")))
    
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

def Color(frame, warna, tebal, label, x1, x2, y1, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), warna, tebal)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna, tebal)

def process_frame(frame):
    resized_frame = cv2.resize(frame, (output_width, output_height))
    results = model(resized_frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            if f'{model.names[cls]}' == 'Pengendara' and conf > 0.3:
                crop_image = resized_frame[y1:y2, x1:x2]
                thread = threading.Thread(target=Pengendara, args=(crop_image,))
                thread.start()

            if f'{model.names[cls]}' == 'Helm':
                Color(resized_frame, (0, 255, 0), 2, f'{model.names[cls]} {conf:.2f}', x1, x2, y1, y2)

            if f'{model.names[cls]}' == 'NoHelm':
                Color(resized_frame, (255, 0, 255), 2, f'{model.names[cls]} {conf:.2f}', x1, x2, y1, y2)

            if f'{model.names[cls]}' == 'Pengendara':
                Color(resized_frame, (255, 0, 0), 2, f'{model.names[cls]} {conf:.2f}', x1, x2, y1, y2)

            if f'{model.names[cls]}' == 'PlatNomor':
                Color(resized_frame, (255, 0, 255), 2, f'{model.names[cls]} {conf:.2f}', x1, x2, y1, y2)

    return resized_frame

def main():

    with ThreadPoolExecutor(max_workers=4) as executor:
        if not cap.isOpened():
            print("Error: Tidak dapat membuka video atau gambar.")
        else:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Tidak dapat membaca frame dari stream.")
                    break

                if frame is None or frame.size == 0:
                    print("Error: Frame kosong.")
                    continue

                future = executor.submit(process_frame, frame)
                resized_frame = future.result()

                cv2.imshow('Output', resized_frame)
                #out.write(resized_frame) # menyimpan video

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            #out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()