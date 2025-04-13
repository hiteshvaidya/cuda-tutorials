#include <stdio.h>

int main() {
    int numbers[] = {10, 20, 30, 40, 50};
    int *ptr = numbers; // Pointer to the first element of the array

    printf("Array elements using pointer arithmetic:\n");
    for (int i = 0; i < 5; i++) {
        printf("Element %d: %d\n", i, *(ptr + i));
    }

    printf("\nArray elements using array indexing:\n");
    for (int i = 0; i < 5; i++) {
        printf("Element %d: %d\n", i, ptr[i]);
    }

    printf("\nAddresses of array elements:\n");
    for (int i = 0; i < 5; i++) {
        printf("Address of numbers[%d]: %p\n", i, (void *)(ptr + i));
    }

    return 0;
}