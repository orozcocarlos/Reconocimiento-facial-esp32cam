import cv2
import urllib.request
import numpy as np
 
personName = 'Carlos'
dataPath = 'D:/VCS/gitclonaciones/esp32face/data' #Cambia a la ruta donde hayas almacenado Data
personPath = dataPath + '/' + personName

f_cas= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
url='http://192.168.1.141/cam-hi.jpg' #ip asignada al arduino
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
cv2.namedWindow("Live Transmission", cv2.WINDOW_AUTOSIZE)
count = 0
while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    face=f_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),roi_gray)
        count = count + 1
  
 
 
    cv2.imshow("live transmission",img)
    key=cv2.waitKey(5)
    if key==ord('q') or count >= 300:
        break
 
cv2.destroyAllWindows()
