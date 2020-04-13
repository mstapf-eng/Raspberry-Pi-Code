import cv2
import pandas as pd

data = pd.read_csv("data/output_0.csv")
size = len(data['image'])

for i in range(0, size-1):
    img = cv2.imread(data['image'][i])
    print(data['servo'][i], data['motor'][i])
    img = cv2.resize(img, (640, 320), interpolation = cv2.INTER_AREA)
    end_x = int((data['servo'][i]-0.15)*1000 + 15)
    img = cv2.line(img, (150, 300), (end_x, 200), (0, 255, 0), 9)
    servo = str(data['servo'][i])
    img = cv2.putText(img, servo, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
