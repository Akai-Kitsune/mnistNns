#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"

#define STEPS 50
#define BATCH_SIZE 100

// Returns a random value between 0 and 1
#define RAND_INT() (rand() % 101)

const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

float predict(mnist_image_t * image, neural_network_t * network){
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset

    // Calculate the activations for each image using the neural network
    neural_network_hypothesis(image, network, activations);

    // Set predict to the index of the greatest activation
    for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
        if (max_activation < activations[j]) {
            max_activation = activations[j];
            predict = j;
        }
    }
    

    // Return predict
    return predict;
}

void print_image(uint8_t *image) {
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            if (image[row * 28 + col] < 64) {
                printf(" "); // Символ пробела, если яркость пикселя меньше 128
            } else if(image[row * 28 + col] < 128){
                printf("*"); // Символ звездочки, если яркость пикселя больше или равна 128
            } else if(image[row * 28 + col] < 256){
                printf("#"); // Символ звездочки, если яркость пикселя больше или равна 128
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches;

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    int k = 1;
    while(1){
        scanf("%d", &k);
        if(k == 0){
            break;
        }
        else if(k==1){
            int test_sample = RAND_INT();
            mnist_image_t test_image = train_dataset->images[test_sample];
            uint8_t test_label = train_dataset->labels[test_sample];

            print_image(test_image.pixels);
            printf("это число [%d]\n", test_label);
        }
        else if(k==2){
            int test_sample = RAND_INT();
            mnist_image_t test_image = train_dataset->images[test_sample];
            uint8_t test_label = train_dataset->labels[test_sample];

            print_image(test_image.pixels);
            printf("это число [%d]\n", test_label);
            printf("прогноз модели [%f]\n", predict(&test_image, &network));
        }
        else if(k==3){
            // Calculate how many batches (so we know when to wrap around)
            batches = train_dataset->size / BATCH_SIZE;

            for (i = 0; i < STEPS; i++) {
                // Initialise a new batch
                mnist_batch(train_dataset, &batch, 100, i % batches);

                // Run one step of gradient descent and calculate the loss
                loss = neural_network_training_step(&batch, &network, 0.5);

                // Calculate the accuracy using the whole test dataset
                accuracy = calculate_accuracy(test_dataset, &network);

                printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss / batch.size, accuracy);
            }
        }
        else{
            break;
        }


    }

    // // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
