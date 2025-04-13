#include <stdio.h>
#include <stddef.h>

int main() {
    size_t array_size = 5;
    int numbers[array_size];

    // Initialize the array
    for (size_t i = 0; i < array_size; i++) {
        numbers[i] = (int)(i + 1);
    }

    // Print the array elements
    printf("Array elements:\n");
    for (size_t i = 0; i < array_size; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");

    // Demonstrate size_t usage
    printf("Size of the array: %zu\n", array_size);
    printf("Size of each element: %zu bytes\n", sizeof(numbers[0]));
    printf("Total size of the array: %zu bytes\n", sizeof(numbers));

    return 0;
}