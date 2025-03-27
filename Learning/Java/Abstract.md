```java
abstract int getDefaultPort()
```

An **abstract class** in Java is a class that cannot be instantiated directly. It is designed to be subclassed, meaning other classes must extend it and provide implementations for its abstract methods (if any). An abstract class can contain both **abstract methods** (methods without a body) and **concrete methods** (methods with an implementation).

### Key Features of an Abstract Class:

1. **Cannot Be Instantiated**: You cannot create an object of an abstract class directly. It is meant to serve as a blueprint for other classes.

2. **Abstract Methods**: An abstract class can contain **abstract methods**—methods that are declared but not defined. These methods must be implemented by subclasses.

3. **Concrete Methods**: An abstract class can also contain **concrete methods**—methods with a body (implementation). These methods can be used directly by subclasses.

**Example:**
```java
// Abstract class
abstract class Animal {
    // Abstract method (no body)
    abstract void sound();

    // Concrete method
    void breathe() {
        System.out.println("Breathing...");
    }
}

// Concrete class that extends the abstract class
class Dog extends Animal {
    // Providing implementation for the abstract method
    void sound() {
        System.out.println("Bark");
    }
}

class Main {
    public static void main(String[] args) {
        // Cannot instantiate Animal directly
        // Animal a = new Animal(); // Error

        // Can instantiate Dog because it provides an implementation for the abstract method
        Dog dog = new Dog();
        dog.sound();  // Outputs: Bark
        dog.breathe(); // Outputs: Breathing...
    }
}
```