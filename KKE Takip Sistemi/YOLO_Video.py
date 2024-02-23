from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x

    cap = cv2.VideoCapture(video_capture)
    frame_genislik = int(cap.get(5))
    frame_yukseklik = int(cap.get(5))

    model=YOLO("YOLO-Weights/ppe.pt")
    sinif_isimleri =['baretsiz',
                   'beyaz-baret',
                   'iş-ayakkabısı',
                   'kırmızı-baret',
                   'kırmızı-yelek',
                   'mavi-baret',
                   'mavi-yelek',
                   'normal-ayakkabı',
                   'sarı-baret',
                   'sarı-yelek',
                   'turuncu-baret',
                   'turuncu-yelek',
                   'yeleksiz']
    while True:
        success, img = cap.read()
        sonuclar = model(img, stream=True)
        for r in sonuclar:
            kutular = r.boxes
            for kutu in kutular:
                x1,y1,x2,y2 = kutu.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf = math.ceil((kutu.conf[0]*100))/100
                index = int(kutu.cls[0])
                sinif_adi = sinif_isimleri[index]
                label = f'{sinif_adi}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if sinif_adi == 'baretsiz':
                    renk = (0, 204, 255)
                elif sinif_adi == "beyaz-baret":
                    renk = (222, 82, 175)
                elif sinif_adi == "iş-ayakkabısı":
                    renk = (128, 0, 128)
                elif sinif_adi == "kırmızı-baret":
                    renk = (124, 252, 0)
                elif sinif_adi == "kırmızı-yelek":
                    renk = (255, 102, 102)
                elif sinif_adi == "mavi-baret":
                    renk = (255, 255, 102)
                elif sinif_adi == "mavi-yelek":
                    renk = (255, 165, 0)
                elif sinif_adi == "normal-ayakkabı":
                    renk = (0, 0, 128)
                elif sinif_adi == "sarı-baret":
                    renk = (255, 255, 240)
                elif sinif_adi == "sarı-yelek":
                    renk = (255, 182, 193)
                elif sinif_adi == "turuncu-baret":
                    renk = (128, 128, 128)
                elif sinif_adi == "turuncu-yelek":
                    renk = (222, 122, 175)
                elif sinif_adi == "yeleksiz":
                    renk = (0, 149, 255)
                else:
                    renk = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), renk,3)
                    cv2.rectangle(img, (x1,y1), c2, renk, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1, lineType=cv2.LINE_AA)

        yield img

cv2.destroyAllWindows()