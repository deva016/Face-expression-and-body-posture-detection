import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.src.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator

train_data=r'D:\project\pyhon\MCP Project\Face expression detection\Dataset\train'
test_data=r'D:\project\pyhon\MCP Project\Face expression detection\Dataset\validation'

train_data_dir=ImageDataGenerator(rescale=1./255)
test_data_dir=ImageDataGenerator(rescale=1./255)

train_generator=train_data_dir.flow_from_directory(train_data,target_size=(48,48),batch_size=8,color_mode="grayscale",class_mode='categorical',shuffle=True)
test_generator=test_data_dir.flow_from_directory(test_data,target_size=(48,48),batch_size=8,color_mode="grayscale",class_mode='categorical',shuffle=True,)

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

model.fit(train_generator,steps_per_epoch=28821//8,epochs=46,validation_data=test_generator,validation_steps=7066 //8,shuffle=False)
model.save('expressiondata1.h5')
