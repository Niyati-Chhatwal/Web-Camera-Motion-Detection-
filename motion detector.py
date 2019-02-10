import cv2, time,numpy, pandas
from datetime import datetime

first_frame=None
compare_frame = None
sum_frame = 0.0
avg_frame = 0.0
status_list=[None, None]
times=[]
p = 1
f = 0
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)

while True:
    
    
    if( f < 25):
    
        check, frame = video.read()
        status=0
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(21,21),0)
        sum_frame = sum_frame + gray
        cv2.putText(frame,str(datetime.now()), (5,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if (first_frame is None):
            first_frame= gray
            continue
    
    
        delta_frame=cv2.absdiff(first_frame,gray)
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=10)
        f = f+1

        (_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 12000:
                continue
            status=1

            (x, y, w, h)=cv2.boundingRect(contour)
            cv2.putText(frame,"Motion Detected", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

    else:
        f = 0
        avg_frame = sum_frame / 25
        avg_frame = numpy.uint8(avg_frame)
        first_frame = avg_frame
        sum_frame = 0.0

    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
        cv2.waitKey(90)
        cv2.imwrite("Motion - " + str(p) +".jpg", frame)
        p = p+1

        
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows
