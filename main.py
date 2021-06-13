# Datos
from keras.preprocessing.image import ImageDataGenerator

# Red neuronal
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import keras

# Plots
from matplotlib import pyplot as plt


# DATA SOURCE --------------------------------------------------

batch_size = 20

# 3 clases 700 muestras por clase
train_data_dir = 'data/dataset/'
validation_data_dir = 'data/validacion/'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
        )

validation_datagen = ImageDataGenerator(
        rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

# MODEL --------------------------------------------------

model1 = Sequential()

model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))

model1.add(Dense(3, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.001),
              metrics=['accuracy'])

# TRAINING --------------------------------------------------

epochs = 600

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=60, restore_best_weights=True)

historial1 = model1.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks = [es]
)

plt.plot(historial1.history['accuracy'], label='accuracy')
plt.plot(historial1.history['val_accuracy'], label='validation accuracy')

plt.title('Entrenamiento animales Oregon')
plt.xlabel('Épocas')
plt.legend(loc="lower right")

plt.show()

'''
# Matriz de confusion
y_pred = (model1.predict(validation_generator) > 0.5).astype("int32")
con_mat = tf.math.confusion_matrix(labels=[0,1,2], predictions=y_pred).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                     index = [0, 1, 2],
                     columns = [0, 1, 2])

'''