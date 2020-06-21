import  cv2 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('model.h5')


imgpath = 'images/face.jpg'

image = cv2.imread(imgpath)

h , w , _ = image.shape

image_resized = cv2.resize(image , (300,300))

points = model.predict(image_resized.reshape((1,300,300,-1)))

image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

points  = points[0]


X = np.array(points[:9])/300 * w
y = np.array(points[9:])/300 * h


plt.plot([X[2],X[0]],[y[2],y[0]],color='red' , linewidth=2)
plt.plot([X[2],X[1]],[y[2],y[1]],color='red' , linewidth=2)
plt.plot([X[0],X[1]],[y[0],y[1]],color='red' , linewidth=2)
plt.plot([X[0],X[5]],[y[0],y[5]],color='red' , linewidth=2)
plt.plot([X[1],X[6]],[y[1],y[6]],color='red' , linewidth=2)
plt.plot([X[3],X[5]],[y[3],y[5]],color='red' , linewidth=2)
plt.plot([X[6],X[8]],[y[6],y[8]],color='red' , linewidth=2)
plt.plot([X[0],X[3]],[y[0],y[3]],color='red' , linewidth=2)
plt.plot([X[1],X[8]],[y[1],y[8]],color='red' , linewidth=2)
plt.plot([X[3],X[4]],[y[3],y[4]],color='red' , linewidth=2)
plt.plot([X[4],X[5]],[y[4],y[5]],color='red' , linewidth=2)
plt.plot([X[5],X[6]],[y[5],y[6]],color='red' , linewidth=2)
plt.plot([X[6],X[7]],[y[6],y[7]],color='red' , linewidth=2)
plt.plot([X[7],X[8]],[y[7],y[8]],color='red' , linewidth=2)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()

