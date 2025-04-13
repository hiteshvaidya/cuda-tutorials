#include <stdio.h>
#include <stdlib.h> // For malloc and free

int main(){
    int *p = NULL; // Initialize a pointer to NULL
    int a = 10;
    
    // Check if the pointer is NULL before dereferencing
    if (p != NULL) {
        printf("Value pointed to by p: %d\n", *p);
    } else {
        printf("Pointer is NULL, cannot dereference.\n");
    }
    
    p = &a; // Now point to a valid address
    
    // Now it's safe to dereference
    printf("Value pointed to by p: %d\n", *p);

    return 0;
}