import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.utils import random_occlusion

def attention_block(x, filters):
    """
    Squeeze-and-Excitation (SE) attention block.
    Enhances channel-wise feature responses.
    """
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, filters))(avg_pool)
    dense1 = Dense(filters // 8, activation='relu')(avg_pool)
    dense2 = Dense(filters, activation='sigmoid')(dense1)
    out = Multiply()([x, dense2])
    return out

def build_gender_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build the gender classification model using MobileNetV2 with an attention block.
    """
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    filters = int(x.shape[-1])
    x = attention_block(x, filters)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def train_model():
    """
    Train the gender classification model with enhanced data augmentation.
    Saves the best model to models/best_model.h5.
    """
    train_dir = os.path.join('dataset', 'Hackathon Test')
    val_dir = os.path.join('dataset', 'Validation')
    classes = ['Male', 'Female']

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        preprocessing_function=random_occlusion
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=classes
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=classes
    )

    model = build_gender_model(input_shape=(224, 224, 3), num_classes=2)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(os.path.join('models', 'best_model.h5'),
                                 monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[checkpoint, early_stop]
    )

if __name__ == '__main__':
    train_model()