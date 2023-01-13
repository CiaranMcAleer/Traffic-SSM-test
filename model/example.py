
# TF for image segmentation model

import tensorflow
import numpy
from PIL import Image
import matplotlib.pyplot as plt

model = tensorflow.saved_model.load('./')
classes = [ "Void" ,  "Sidewalk" ,  "TrafficLight" ,  "Bicyclist" ,  "Car" ,  "Pedestrian" ,  "SUVPickupTruck" ,  "Wall" ,  "Building" ,  "LaneMkgsDriv" ,  "Road" ,  "Tree" ,  "Misc_Text" ,  "Sky" ,  "Column_Pole" ,  "CartLuggagePram" ,  "OtherMoving" ,  "VegetationMisc" ,  "Truck_Bus" ,  "Fence" ,  "ParkingBlock" ,  "RoadShoulder" ,  "SignSymbol" ,  "TrafficCone" ,  "Bridge" ,  "Archway" ,  "Animal" ,  "MotorcycleScooter" ,  "LaneMkgsNonDriv" ,  "Child" ,  "Tunnel" , ]

img = Image.open("image.jpg").convert('RGB')
img = img.resize((256, 256), Image.ANTIALIAS)
inp_numpy = numpy.array(img)[None]


inp = tensorflow.constant(inp_numpy, dtype='float32')

segmentation_output = model(inp)[0].numpy().argmax(-1)

plt.imshow(segmentation_output)
plt.show()