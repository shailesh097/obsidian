In Java, the `final` keyword is used to define constants, prevent method overriding, and prevent class inheritance. It is a modifier used to restrict the user in certain ways. It can be applied to variables, methods, and classes and its behavior varies depending on where it is used.

### 1. **Final Variables**:

- **Constant Variables**: When a variable is declared as `final`, its value cannot be changed once it is initialized.
- **Primitive Types**: The value of the variable cannot be changed.
- **Reference Types (Objects)**: The reference (memory address) cannot be changed to point to a different object, but the object itself can still be modified if it is mutable.

**Example with a primitive type:**
```java
final int MAX_SPEED = 120; MAX_SPEED = 130;  // Error: cannot assign a value to a final variable`
```

**Example with a reference type (object):**
```java
final StringBuilder sb = new StringBuilder("Hello");
sb.append(" World"); // This is allowed because the object is mutable 
sb = new StringBuilder("New Object"); // Error: cannot assign a new object to a final reference
```
### 2. **Final Methods**:

- A `final` method cannot be overridden by subclasses. This is useful when you want to prevent a subclass from changing the behavior of a method.

**Example:**
```java
class Animal {
    final void makeSound() {
        System.out.println("Animal sound");
    }
}

class Dog extends Animal {
    // Error: cannot override the final method from Animal
    void makeSound() {
        System.out.println("Bark");
    }
}
```


### 3. **Final Classes**:

- A `final` class cannot be subclassed. This is useful when you want to prevent inheritance of a class, ensuring that its behavior cannot be modified by subclasses.

**Example:**
```java
final class Vehicle { 
// Class implementation
} 

class Car extends Vehicle { 
// Error: cannot subclass a final class 
}
```

### 4. **Final Parameters**:

- When a parameter is declared as `final`, it cannot be modified within the method. This is often used to ensure that a method does not alter the values passed to it.

**Example:**
```java
public void printMessage(final String message) {
    message = "New Message";  // Error: cannot assign a value to a final parameter
    System.out.println(message);
}

```

