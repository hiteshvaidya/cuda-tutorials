#include <stdio.h>

// Define a macro to check the operating system
#if defined(_WIN32) || defined(_WIN64)
    #define OS "Windows"
#elif defined(__linux__)
    #define OS "Linux"
#elif defined(__APPLE__) || defined(__MACH__)
    #define OS "MacOS"
#else
    #define OS "Unknown OS"
#endif

// Define a macro for debugging
#define DEBUG 1

// Define macros for algebraic calculations
#ifndef ENABLE_ADDITION
    #define ENABLE_ADDITION 1
#endif

#ifndef ENABLE_SUBTRACTION
    #define ENABLE_SUBTRACTION 1
#endif

#ifndef ENABLE_MULTIPLICATION
    #define ENABLE_MULTIPLICATION 1
#endif

#ifndef ENABLE_DIVISION
    #define ENABLE_DIVISION 1
#endif

int main() {
    printf("Operating System: %s\n", OS);

    #if DEBUG
        printf("Debugging is enabled.\n");
    #else
        printf("Debugging is disabled.\n");
    #endif

    int a = 10, b = 5;

    #if ENABLE_ADDITION
        printf("Addition: %d + %d = %d\n", a, b, a + b);
    #endif

    #if ENABLE_SUBTRACTION
        printf("Subtraction: %d - %d = %d\n", a, b, a - b);
    #endif

    #if ENABLE_MULTIPLICATION
        printf("Multiplication: %d * %d = %d\n", a, b, a * b);
    #endif

    #if ENABLE_DIVISION
        if (b != 0) {
            printf("Division: %d / %d = %d\n", a, b, a / b);
        } else {
            printf("Division by zero is not allowed.\n");
        }
    #endif

    return 0;
}