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
 * THROTTLED, FIXED-TARGET-TPS closed-model load test, fed from JSONL prompts.
 *
 * <p>Question this answers: <em>"At a target arrival rate of N TPS, does the
 * server hold the SLO?"</em> A large standing pool of virtual users
 * ({@code tps * 100}) guarantees a user is always ready, and
 * {@code .throttle(jumpToRps(tps))} paces the actual send rate to exactly the
 * target TPS. Reported throughput is therefore the REQUESTED arrival rate (as
 * long as the server keeps up); latency / KO percentages reveal whether the SLO
 * holds at that rate. Contrast with:
 * <ul>
 *   <li>{@link JsonlChatCompletionsSimulation} — OPEN model (fixed arrival rate,
 *       no standing pool); the default for real-world traffic shaping.</li>
 *   <li>{@link ClosedConcurrencySimulation} — UNTHROTTLED fixed-{@code N} pool
 *       run flat-out to measure the MAX sustainable TPS per instance.</li>
 * </ul>
 *
 * <p>Injection profile (identical to {@link ThrottledTpsCsvSimulation}, which is
 * why this is the apples-to-apples harness for reconciling against TPS numbers
 * produced by the CSV-driven baseline):
 * <pre>
 *   int concurrentUsers = simulationTps * 100;
 *   injectClosed(
 *       rampConcurrentUsers(1).to(concurrentUsers).during(30),
 *       constantConcurrentUsers(concurrentUsers).during(duration))
 *   .throttle(jumpToRps(simulationTps), holdFor(duration + 30));
 * </pre>
 *
 * <p>The ONLY difference from {@link ThrottledTpsCsvSimulation} is the data
 * source: instead of {@code prompts.csv} + {@code system-message.txt} +
 * {@code ChatRequestTemplates.template} (which hardcodes max_tokens=600 and a
 * synthetic body), this reads whole recorded requests from
 * {@code prompts.jsonl}, so the workload matches production traffic shape
 * rather than a synthetic one. max_tokens etc. come from
 * {@code chat-config.json} on the classpath, exactly like
 * {@link JsonlChatCompletionsSimulation}.
 *
 * <p>Unlike {@link JsonlChatCompletionsSimulation}'s "closed" branch (which
 * trusts the runner's reduced {@code -DconcurrentUsers = tps *
 * latency_budget}), this class defaults to {@code tps * 100} so the throttled
 * profile is byte-for-byte the reference methodology regardless of how the
 * runner is configured. {@code -DconcurrentUsers} is read only as an optional
 * explicit override; if absent the {@code tps * 100} pool is used.
 *
 * <p>Driven by JVM system properties:
 * <pre>
 *   -DsimulationTps=N         requests-per-second target (paced by the throttle)
 *   -DsimulationDuration=SEC  steady-state duration in seconds
 *   -DconcurrentUsers=N       OPTIONAL explicit pool override (default tps*100)
 * </pre>
 */
public class ThrottledTpsJsonlSimulation extends Simulation {
    private static final Logger log = LoggerFactory.getLogger(ThrottledTpsJsonlSimulation.class);
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

    ScenarioBuilder users = scenario("Chat Completions").exec(invocations);

    {
        Integer simulationTps = Integer.getInteger("simulationTps");
        Integer simulationDuration = Integer.getInteger("simulationDuration");
        log.info("Simulation TPS: " + simulationTps);
        log.info("Simulation Duration: " + simulationDuration);

        if (simulationTps == null || simulationDuration == null) {
            throw new IllegalStateException(
                    "ThrottledTpsJsonlSimulation requires -DsimulationTps and -DsimulationDuration JVM properties");
        }

        // Default pool is tps * 100 (matching ThrottledTpsCsvSimulation), so the
        // throttled profile is byte-for-byte the reference methodology. An
        // explicit -DconcurrentUsers wins only if the operator deliberately
        // sets one.
        Integer override = Integer.getInteger("concurrentUsers");
        int concurrentUsers = (override != null) ? override : simulationTps * 100;
        log.info("Concurrent Users (throttled pool, tps*100 default): " + concurrentUsers);

        setUp(
                users.injectClosed(
                                rampConcurrentUsers(1).to(concurrentUsers).during(30),
                                constantConcurrentUsers(concurrentUsers).during(simulationDuration))
                        .throttle(
                                jumpToRps(simulationTps), holdFor(simulationDuration + 30)
                        )
        ).protocols(httpProtocol);
    }

    @Override
    public void after() {
        log.info("Simulation finished. Avg completion tokens: {}",
                responseProcessor.getAvgCompletionTokens());
    }

    private static FeederBuilder<Object> loadJsonlFeeder() {
        try (InputStream stream = ThrottledTpsJsonlSimulation.class
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
