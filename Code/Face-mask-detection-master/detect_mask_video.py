# import packages nya
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from pygame import mixer





def detect_and_predict_mask(frame, faceNet, maskNet):
	
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# lewati gumpalan melalui jaringan dan mendapatkan deteksi wajah
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# inisialisasi daftar wajah, lokasi yang sesuai,
	# dan daftar prediksi dari jaringan masker wajah
	faces = []
	locs = []
	preds = []

	# deteksi loop nya
	for i in range(0, detections.shape[2]):
		# ekstrak kepercayaan (yaitu, probabilitas) yang terkait dengan
		# pendeteksian
		confidence = detections[0, 0, i, 2]

		# menyaring deteksi yang lemah dengan memastikan kepercayaannya
		# lebih besar dari kepercayaan minimum
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# ekstrak ROI wajah, ubah dari saluran BGR ke RGB
			# memesan, mengubah ukurannya menjadi 224x224, dan memprosesnya terlebih dahulu
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# tambahkan wajah dan kotak pembatas ke masing-masing
			# daftar
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# hanya membuat prediksi jika setidaknya satu wajah terdeteksi
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# kembalikan 2-tupel dari lokasi wajah dan yang sesuai
	# lokasi
	return (locs, preds)

# memuat model detektor wajah serial, dan masukan path nya beserta sound dari disk
prototxtPath = r"C:\Mask Detections\Code\Face-mask-detection-master\face_detector\deploy.prototxt"
weightsPath = r"C:\Mask Detections\Code\Face-mask-detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
mixer.init()
sound = mixer.Sound(r"C:\Mask Detections\Code\Face-mask-detection-master\alarm.wav")
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("C:\Mask Detections\Code\Face-mask-detection-master\mask_detector.model")

# inisialisasi video stream nya
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop di atas bingkai dari aliran video
while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# mendeteksi wajah dalam bingkai dan menentukan apakah mereka mengenakan they
	# masker wajah atau NOT
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop di atas lokasi wajah yang terdeteksi dan yang sesuai
	# lokasi
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

	
		label = "Mask" if mask > withoutMask else "No Mask !!!" 
		sound.play()== "Mask" if mask < withoutMask else "No Mask !!!"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# tampilkan bingkai keluaran
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
