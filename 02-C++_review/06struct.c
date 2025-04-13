#include <stdio.h>
#include <string.h>

// Define a struct to represent a person
struct Person {
    char name[50];
    int age;
    float height;
};

typedef struct {
    float x;
    float y;
} Point;

int main() {
    // Create and initialize a struct variable
    struct Person person1;
    Point p = {1.0, 2.0};

    // Assign values to the struct fields
    strcpy(person1.name, "Hitesh");
    person1.age = 25;
    person1.height = 5.9;

    // Print the struct fields
    printf("Name: %s\n", person1.name);
    printf("Age: %d\n", person1.age);
    printf("Height: %.1f\n", person1.height);

    return 0;
}