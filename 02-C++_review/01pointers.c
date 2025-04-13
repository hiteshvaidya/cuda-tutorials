#include <stdio.h>

void pointer1(){
    int a = 10;
    int *p = &a; // Pointer to an integer
    printf("Value of a: %d\n", a);
    printf("Address of a: %p\n", (void*)&a);
    printf("Value of p: %p\n", (void*)p);
    printf("Value pointed to by p: %d\n", *p);

    *p = 20; // Change the value of a using the pointer
    printf("New value of a: %d\n", a);
}

void pointer2pointer(){
    int value = 42;
    int* ptr1 = &value;
    int** ptr2 = &ptr1;
    int*** ptr3 = &ptr2;
    
    
    printf("Value: %d\n", ***ptr3);  // Output: 42
}

int main(){
    pointer1();
    pointer2pointer();
    return 0;
}