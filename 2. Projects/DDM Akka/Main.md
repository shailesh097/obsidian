- `Command.applyOn(args);`: Processes command-line arguments to configure the application. This likely involves parsing options such as roles, ports, or other configurations.

- `SystemConfiguration config = SystemConfigurationSingleton.get();`: Retrieves a singleton instance of `SystemConfiguration`, which contains application-specific settings like the role of the node (e.g., Master or Worker), actor system name, and Akka-specific configuration.
`

```Java
final ActorSystem<Guardian.Message> guardian = ActorSystem.create(
    Guardian.create(), 
    config.getActorSystemName(), 
    config.toAkkaConfig()
);
```

- **`ActorSystem<Guardian.Message>`**: Initializes the Akka actor system with:
    - The `Guardian` actor as the root actor.
    - A name retrieved from the configuration (`config.getActorSystemName()`).
    - Akka-specific settings (`config.toAkkaConfig()`).

- `guardian.tell(new Guardian.StartMessage());`: Sends a `StartMessage` to the `Guardian` actor to trigger its behavior. This is where the Master node begins its work.
