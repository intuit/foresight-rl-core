package org.foresightrl.perf;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.foresightrl.perf.model.ChatRequest;
import org.foresightrl.perf.model.ConversationMessage;
import io.gatling.javaapi.core.ChainBuilder;
import io.gatling.javaapi.core.FeederBuilder;
import io.gatling.javaapi.core.ScenarioBuilder;
import io.gatling.javaapi.core.Session;
import io.gatling.javaapi.core.Simulation;
import io.gatling.javaapi.http.HttpProtocolBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static io.gatling.javaapi.core.CoreDsl.*;
import static io.gatling.javaapi.http.HttpDsl.http;
import static io.gatling.javaapi.http.HttpDsl.status;

/**
 * TPS-throttled chat-completion load test driven from a JSONL file of
 * pre-built OpenAI-compatible chat messages.
 *
 * Each line of {@code prompts.jsonl} (on the classpath / in
 * src/gatling/resources/) must be a single JSON object with at least:
 *
 * <pre>
 *   {"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
 * </pre>
 *
 * The {@code prompt} field is forwarded verbatim as the {@code messages}
 * array in the outgoing OpenAI chat-completion request, so the system
 * prompt -- and any multi-turn context -- comes from the data itself; no
 * separate {@code system-message.txt} is needed.
 *
 * Other top-level fields (e.g. eval metadata) are ignored.
 *
 * Driven by JVM system properties:
 *   -DsimulationTps=N         requests-per-second target
 *   -DsimulationDuration=SEC  steady-state duration in seconds
 */
public class JsonlChatCompletionsSimulation extends Simulation {
    private static final Logger log = LoggerFactory.getLogger(JsonlChatCompletionsSimulation.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final String JSONL_RESOURCE = "prompts.jsonl";

    private final ChatCompletionResponseProcessor responseProcessor = new ChatCompletionResponseProcessor();

    private final FeederBuilder<Object> jsonlFeeder = loadJsonlFeeder().random();

    HttpProtocolBuilder httpProtocol = http.baseUrl(Constants.BASE_URL_LOCAL)
            .acceptHeader("application/json")
            .contentTypeHeader("application/json");

    private final Function<Session, String> requestBodyBuilder = session -> {
        try {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawMessages =
                    (List<Map<String, Object>>) session.get("prompt");
            if (rawMessages == null) {
                log.error("Session is missing 'prompt' messages array");
                return "{}";
            }

            List<ConversationMessage> messages = new ArrayList<>(rawMessages.size());
            for (Map<String, Object> raw : rawMessages) {
                ConversationMessage msg = new ConversationMessage();
                Object role = raw.get("role");
                Object content = raw.get("content");
                msg.setRole(role == null ? "" : role.toString());
                msg.setContent(content == null ? "" : content.toString());
                messages.add(msg);
            }

            // Use ChatRequest defaults from chat-config.json on the classpath
            // (model name, temperature, max_tokens, ...) but override messages
            // from the JSONL line.
            ChatRequest req = new ChatRequest();
            req.setMessages(messages);
            return objectMapper.writeValueAsString(req);
        } catch (Exception ex) {
            log.error("Error building JSONL request body: {}", ex.getMessage(), ex);
            return "{}";
        }
    };

    ChainBuilder invocations =
            feed(jsonlFeeder).exec(
                            http("Invocations")
                                    .post(Constants.MODEL_URL_LOCAL)
                                    .body(StringBody(requestBodyBuilder))
                                    .check(status().is(200))
                                    .check(bodyString().saveAs("responseBody"),
                                            status().saveAs("responseStatus")))
                    .exec(session -> {
                        if (!session.contains("responseStatus")) {
                            return session;
                        }
                        String responseBody = session.getString("responseBody");
                        Integer responseStatus = session.getInt("responseStatus");
                        responseProcessor.processResponse(null, responseBody, responseStatus, null, null);
                        return session;
                    });

    ScenarioBuilder users = scenario("Chat Completions JSONL").exec(invocations);

    {
        Integer simulationTps = Integer.getInteger("simulationTps");
        Integer simulationDuration = Integer.getInteger("simulationDuration");
        log.info("Simulation TPS: " + simulationTps);
        log.info("Simulation Duration: " + simulationDuration);

        if (simulationTps == null || simulationDuration == null) {
            throw new IllegalStateException(
                    "JsonlChatCompletionsSimulation requires -DsimulationTps and -DsimulationDuration JVM properties");
        }

        // Injection profile selector. 'open' (default) decouples the request
        // arrival rate from server response time -- new requests arrive at a
        // fixed `simulationTps` regardless of how fast prior requests finish, so
        // the server's admission queue is never flooded beyond the steady-state
        // arrival rate and reported latency reflects real per-request cost
        // rather than client-side queue wait. 'closed' keeps a fixed standing
        // pool of virtual users throttled to TPS (legacy behavior).
        String injectionModel = System.getProperty("injectionModel", "open");
        log.info("Injection Model: " + injectionModel);

        if ("closed".equalsIgnoreCase(injectionModel)) {
            // Concurrency ceiling for the closed injection model. The runner
            // passes -DconcurrentUsers derived from the per-request latency
            // budget (concurrent_users = tps * max_latency_seconds, Little's
            // Law). We fall back to the legacy simulationTps*100 only if the
            // prop is absent (older runners that don't pass it).
            Integer concurrentUsers = Integer.getInteger("concurrentUsers");
            if (concurrentUsers == null) {
                concurrentUsers = simulationTps * 100;
                log.warn("-DconcurrentUsers not set; falling back to legacy simulationTps*100={}",
                        concurrentUsers);
            }
            log.info("Concurrent Users: " + concurrentUsers);

            setUp(
                    users.injectClosed(
                                    rampConcurrentUsers(1).to(concurrentUsers).during(30),
                                    constantConcurrentUsers(concurrentUsers).during(simulationDuration))
                            .throttle(
                                    jumpToRps(simulationTps), holdFor(simulationDuration + 30)
                            )
            ).protocols(httpProtocol);
        } else {
            // Open model: fixed arrival rate. Ramp arrivals up over 30s to avoid
            // a cold-start spike, then hold at `simulationTps` new users/sec for
            // the steady-state duration. Each "user" fires exactly one request
            // (the scenario is a single invocation chain), so users/sec == RPS.
            setUp(
                    users.injectOpen(
                            rampUsersPerSec(1).to(simulationTps).during(30),
                            constantUsersPerSec(simulationTps).during(simulationDuration))
            ).protocols(httpProtocol);
        }
    }

    @Override
    public void after() {
        log.info("Simulation finished. Avg completion tokens: {}",
                responseProcessor.getAvgCompletionTokens());
    }

    /**
     * Load the entire prompts.jsonl into memory and return a Gatling
     * list-feeder over the records. Done once at simulation class init.
     *
     * Each record is exposed as a Gatling session entry with a single key:
     *   - "prompt" -> List<Map<String,Object>> (the OpenAI messages array)
     *
     * Other top-level fields are dropped to keep session payload small.
     */
    private static FeederBuilder<Object> loadJsonlFeeder() {
        try (InputStream stream = JsonlChatCompletionsSimulation.class
                .getClassLoader()
                .getResourceAsStream(JSONL_RESOURCE);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(
                             requireResource(stream), StandardCharsets.UTF_8))) {
            List<Map<String, Object>> records = new ArrayList<>();
            String line;
            int total = 0;
            int skipped = 0;
            TypeReference<List<Map<String, Object>>> messagesType =
                    new TypeReference<List<Map<String, Object>>>() {
                    };
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) continue;
                total++;
                try {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> raw = objectMapper.readValue(line, Map.class);
                    Object prompt = raw.get("prompt");
                    if (!(prompt instanceof List)) {
                        skipped++;
                        continue;
                    }
                    // Re-parse to enforce the messages-list shape and drop everything else.
                    String promptJson = objectMapper.writeValueAsString(prompt);
                    List<Map<String, Object>> messages =
                            objectMapper.readValue(promptJson, messagesType);
                    Map<String, Object> entry = new HashMap<>(1);
                    entry.put("prompt", messages);
                    records.add(entry);
                } catch (Exception ex) {
                    skipped++;
                }
            }
            log.info("Loaded {} chat records from {} ({} total lines, {} skipped)",
                    records.size(), JSONL_RESOURCE, total, skipped);
            if (records.isEmpty()) {
                throw new IllegalStateException(
                        JSONL_RESOURCE + " produced zero valid records");
            }
            return listFeeder(records);
        } catch (IOException ex) {
            throw new RuntimeException("Failed to load " + JSONL_RESOURCE, ex);
        }
    }

    private static InputStream requireResource(InputStream stream) {
        if (stream == null) {
            throw new IllegalStateException(
                    JSONL_RESOURCE + " not found on classpath -- it must be supplied via the SageMaker data channel.");
        }
        return stream;
    }
}
