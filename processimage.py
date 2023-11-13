from ultralytics import YOLO
import os
import cv2
import time

model = YOLO(os.path.expanduser('./best_W_3.pt')) #white
model1 = YOLO(os.path.expanduser('./last_20_21_O.pt')) #over
model3 = YOLO(os.path.expanduser('./last_13_MWO.pt'))#paste
clip_limit = 78  # Set your desired clip limit (78 in this example)
tile_size = 20


def predict(iimage):
    img = cv2.imread(f"uploads/{iimage}")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create an instance of the CLAHE (Contrast Limited Adaptive Histogram Equalization) class
    clahe = cv2.createCLAHE(clipLimit=clip_limit / 10.0, tileGridSize=(tile_size, tile_size))
    width, height, channels = img.shape

    print('Image width:', width)
    print('Image height:', height)
    # Apply AHE to the image
    equalized_image = clahe.apply(image)

    bgr_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (2048, 2048))
    bgr_image = cv2.resize(bgr_image, (2048, 2048))

    results = model(bgr_image, imgsz=(2048, 2048),conf=0.5)

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        for box in boxes:
            # print(box.xywh)
            if (int(box.cls) == 0):
                x = int(box.xywh[0][0])
                y = int(box.xywh[0][1])
                w = int(box.xywh[0][2])
                h = int(box.xywh[0][3])
                i = int(box.cls)
                cv2.rectangle(img, (x - int(w / 2), y - int(h / 2)), ((x + int(w / 2)), (y + int(h / 2))), (0, 0, 255),
                              thickness=4)
                cv2.putText(img, "whitner", ((x - int(w / 2), y - int(h / 2) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2)

############# overwriting

    bgr_image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    bgr_image2 = cv2.resize(bgr_image2, (2048, 2048))
    # model.predict(bgr_image, save=True, show=True, imgsz=(2048, 2048))
    results = model1(bgr_image2, imgsz=(2048, 2048),conf=0.5)

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        for box in boxes:
            # print(box.xywh)
            if (int(box.cls) == 0):
                x = int(box.xywh[0][0])
                y = int(box.xywh[0][1])
                w = int(box.xywh[0][2])
                h = int(box.xywh[0][3])
                i = int(box.cls)
                cv2.rectangle(img, (x - int(w / 2), y - int(h / 2)), ((x + int(w / 2)), (y + int(h / 2))), (0, 255,255 ),
                              thickness=4)
                cv2.putText(img, "overwriting", ((x - int(w / 2), y - int(h / 2) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2)
##################### paste
    clahe2 = cv2.createCLAHE(clipLimit=0 / 10.0, tileGridSize=(3, 3))

    # Apply AHE to the image
    equalized_image2 = clahe2.apply(image)
    bgr_image3 = cv2.cvtColor(equalized_image2, cv2.COLOR_GRAY2BGR)



    results = model3(bgr_image3, imgsz=(2048, 2048), conf=0.5)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        for box in boxes:
            # print(box.xywh)
            if (int(box.cls) == 0):
                x = int(box.xywh[0][0])
                y = int(box.xywh[0][1])
                w = int(box.xywh[0][2])
                h = int(box.xywh[0][3])
                i = int(box.cls)
                a = (2048 / height) * (x - (w / 2))

                b = (2048 / width) * (y - (h / 2))
                c = (2048 / height) * (x + (w / 2))

                d = (2048 / width) * (y + (h / 2))

                cv2.rectangle(img, (int(a), int(b)),
                              (int(c), int(d)), (0, 0, 255), thickness=4)
                cv2.putText(img, "discrepency", (int(a), int(b) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    newfilename = f"static/{iimage}"
    print(f"{newfilename}")
    cv2.imwrite(newfilename, img)
    return newfilename








