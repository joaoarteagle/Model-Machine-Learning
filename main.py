
from keras import layers, models
from keras.api.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), #converte a imagem em um vetor
    layers.Dense(128, activation='relu'), #camada densa com 128 neuronios
    layers.Dropout(0.2), #evita overfitting
    layers.Dense(10, activation='softmax') #camada de saida com 10 neuronios
])


model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\n\nprecisão no conjunto de teste: {test_acc}\n\n")

