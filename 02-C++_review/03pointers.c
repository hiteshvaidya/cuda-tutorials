#include <stdio.h>
#include <stdlib.h> // For malloc and free

int main(){
    void *p = NULL; // Initialize a pointer to NULL
    
    p = malloc(sizeof(int)); // Allocate memory for an integer
    if (p == NULL) {
        printf("Memory allocation failed.\n");
        return 1; // Exit if memory allocation fails
    }
    printf("value of p: %p\n", (int*)p);
    free(p); // Free the allocated memory
    return 0;
}