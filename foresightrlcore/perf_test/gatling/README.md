# Vendored Gatling Project (`foresight-perf-test`)

This is a minimal Gatling load-test project. It targets a local `vllm serve`
process on `http://localhost:8080` and is invoked by
`foresightrlcore.perf_test.runner` inside a SageMaker training container.

Everything here is provider-neutral: no credentials, no served-model names, no
prompt content. The system message and prompts are supplied at runtime through
the SageMaker data channel, never vendored.

## Layout

```
gatling/
├── build.gradle                                  # cleaned up: no app plugin, no test deps
├── settings.gradle
├── gradlew, gradlew.bat, gradle/wrapper/         # convenience for local devs
└── src/gatling/
    ├── java/org/foresightrl/perf/
    │   ├── ThrottledTpsCsvSimulation.java        # throttled-closed (tps*100), CSV data — DEFAULT
    │   ├── ThrottledTpsJsonlSimulation.java      # same injection profile, JSONL data
    │   ├── JsonlChatCompletionsSimulation.java   # open OR closed (honors -DinjectionModel), JSONL data
    │   ├── ClosedConcurrencySimulation.java      # unthrottled-closed (fixed N), JSONL data — per-instance sizing
    │   ├── ChatRequestTemplates.java             # request body builders (CSV path)
    │   ├── ChatCompletionResponseProcessor.java  # response parser + token stats
    │   ├── Constants.java                        # URL + role constants (secret-free)
    │   └── model/{ChatRequest,ConversationMessage}.java
    └── resources/
        ├── gatling.conf
        └── logback.xml
        # NOTE: prompts.csv, system-message.txt, chat-config.json, prompts.jsonl are NOT vendored.
        # They are user-supplied via the SageMaker data channel (the `gatling.data_s3_path`
        # config key) and copied here at runtime by the perf-test runner.
```

See `docs/launching-performance-test.md` → "Available simulations" for a full
catalog: which injection model each class uses (throttled-closed vs
unthrottled-closed vs open), its data source, and how to choose between them.

## Running locally (for development only)

The container entrypoint installs Gradle 8.10 and runs the simulation like this:

```bash
gradle gatlingRun \
  --simulation=org.foresightrl.perf.ThrottledTpsCsvSimulation \
  -DsimulationTps=4 \
  -DsimulationDuration=600
```

The Python runner sets these JVM properties for every simulation:
`-DsimulationTps`, `-DsimulationDuration`, `-DconcurrentUsers`,
`-DinjectionModel`, `-DsimulationWorkspace`, `-Dgatling.http.requestTimeout`.
Each class uses the subset it needs:

- **`ThrottledTpsCsvSimulation`, `ThrottledTpsJsonlSimulation`** (throttled-closed): use `-DsimulationTps` (pool is `tps*100`).
- **`ClosedConcurrencySimulation`** (unthrottled-closed): use `-DconcurrentUsers=N` (TPS ignored).
- **`JsonlChatCompletionsSimulation`**: branches on `-DinjectionModel` (`open` → arrival rate `tps`; `closed` → pool `concurrentUsers`).

## Adding a new simulation

1. Drop a new `.java` file under `src/gatling/java/org/foresightrl/perf/`.
2. Set `gatling.simulation_class` in your YAML config to its FQCN.
3. No Python or `build.gradle` changes are required — Gatling auto-discovers
   simulation classes on the classpath.

## Shipping simulations from an extension package

Simulations that are specific to one deployment — a served-model name, bespoke
sampling parameters, or a class name existing configs already reference —
belong in an extension package rather than here. Lay them out as a Java package
tree under `<package>/perf_test/gatling_java/`, for example:

```
foresightrl_intuit/perf_test/gatling_java/com/example/perf/MySimulation.java
```

List the package in `container.extra_packages` and the runner copies that tree
into `src/gatling/java/` before Gradle runs, so both compile as one source set
and extension classes can import `org.foresightrl.perf.*`. Nothing extra to
configure: the runner reads the same `extra_packages` list the plugin loader
uses.
