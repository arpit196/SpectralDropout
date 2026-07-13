import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from src.models import VGG16_with_leaky_relu_and_spectral_dropout
from src.utils import load_images, lr_scheduler

def train_model(model, train_images, train_labels, test_images, test_labels, path, epochs=200, init_lr=0.1):
    # Setup learning rate schedule callback dynamically
    reduce_lr = LearningRateScheduler(lambda epoch: lr_scheduler(epoch))

    checkpoint = ModelCheckpoint(
        filepath=f'./weights/{path}.weights.h5', # Saved into weights directory
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Setup data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(train_images)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=init_lr, momentum=0.9, nesterov=True),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    return model.fit(
        datagen.flow(train_images, train_labels, batch_size=100),
        epochs=epochs,
        validation_data=(test_images, test_labels),
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    train_img, train_lbl, test_img, test_lbl = load_images(cfr100=False)

    # 2. Initialize Model
    print("Initializing Spectral VGG model...")
    model = VGG16_with_leaky_relu_and_spectral_dropout(rate=0.2)

    # 3. Start Training
    print("Starting training loop...")
    history = train_model(model, train_img, train_lbl, test_img, test_lbl, path='spectral_dropout_model_test')