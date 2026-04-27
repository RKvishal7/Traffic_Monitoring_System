# traffic monitoring system
import cv2 as cv
from  ultralytics import YOLO
model = YOLO("yolov8n.pt")
vid = cv.VideoCapture("traffic.mp4") # add your own video path 
line_y = 300
count = 0
tracked_ids = set()
while True:
    ret,frame = vid.read()
    if not ret:
        print("video error")
        break

    frame = cv.resize(frame,(640,480))

    result = model.track(frame,persist=True)

    if result[0] is not None:
        boxes = result[0].boxes.xyxy.cpu().numpy()
        ids = result[0].boxes.id.cpu().numpy()

        for box , object_id in zip(boxes,ids):
            x1,y1,x2,y2 = map(int,box)

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            cv.rectangle(frame,(x2,y2),(x1,y1),(0,255,0),2)
            cv.putText(frame,f"ID {int(object_id)}",(x1,y1-10),cv.FONT_HERSHEY_TRIPLEX,0,255,0,2)

            if cy > line_y and object_id not in tracked_ids:
                count+= 1
                tracked_ids.add(object_id)

    cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv.putText(frame,f"count: {count}",(20,30),cv.FONT_HERSHEY_TRIPLEX,0,255,3)

    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
