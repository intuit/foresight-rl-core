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
 * CLOSED-MODEL CONCURRENCY PROBE — "what can ONE instance sustain?"
 *
 * Unlike {@link JsonlChatCompletionsSimulation} (open model, fixed arrival
 * rate) and unlike the legacy closed model (fixed pool + a TPS throttle on
 * top), this simulation runs a FIXED POOL of N concurrent virtual users with
 * NO throttle and NO arrival-rate cap. Each user loops: send one request, wait
 * for the full response, immediately send the next — for the whole duration.
 *
 * Why this design answers the sizing question cleanly:
 *   - Concurrency is HARD-CAPPED at N by construction (a user never has more
 *     than one request in flight, and there are exactly N users). The runaway
 *     pile-up that produces client timeouts in the open model is IMPOSSIBLE
 *     here — so a healthy server never times out.
 *   - With no throttle, the pool runs flat-out, so the ACHIEVED throughput
 *     (requests/sec, reported by Gatling as mean req/s) equals the maximum the
 *     server can sustain at concurrency N. That is exactly "TPS per instance
 *     at N concurrent".
 *   - Sweeping N (via the runner's tps_levels, each value reused as the pool
 *     size) traces the throughput-vs-concurrency curve and the latency at each
 *     point, so we can read the largest N whose p95 latency still meets the SLO
 *     and the TPS it delivers there.
 *
 * Driven by JVM system properties (set by the Python runner):
 *   -DconcurrentUsers=N        size of the fixed standing pool (the concurrency)
 *   -DsimulationDuration=SEC   steady-state duration in seconds
 *
 * Note: -DsimulationTps is accepted but IGNORED (there is no throttle); the
 * runner passes it because it sweeps tps_levels — we reuse each level as N via
 * -DconcurrentUsers. Set gatling.concurrent_users to pin N explicitly, or rely
 * on the runner deriving it (see config notes).
 */
public class ClosedConcurrencySimulation extends Simulation {
    private static final Logger log = LoggerFactory.getLogger(ClosedConcurrencySimulation.class);
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
            log.error("Error building request body: {}", ex.getMessage(), ex);
            return "{}";
        }
    };

    ChainBuilder oneRequest =
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

    {
        Integer simulationDuration = Integer.getInteger("simulationDuration");
        Integer concurrentUsers = Integer.getInteger("concurrentUsers");
        Integer simulationTps = Integer.getInteger("simulationTps"); // accepted, ignored
        log.info("Simulation Duration: " + simulationDuration);
        log.info("Concurrent Users (fixed pool, no throttle): " + concurrentUsers);
        if (simulationTps != null) {
            log.info("simulationTps=" + simulationTps + " is IGNORED by ClosedConcurrencySimulation "
                    + "(no throttle); concurrency is fixed at concurrentUsers).");
        }

        if (simulationDuration == null || concurrentUsers == null) {
            throw new IllegalStateException(
                    "ClosedConcurrencySimulation requires -DsimulationDuration and -DconcurrentUsers");
        }

        // Each virtual user loops `oneRequest` for a TIME-BOUNDED window, then
        // exits. This is the critical difference from `.forever()`: with
        // `.forever()` a closed-model user never finishes its scenario, so
        // Gatling waits indefinitely for in-flight users to complete and the
        // `constantConcurrentUsers(...).during(duration)` window is effectively
        // ignored (the run never terminates on its own — it overruns by 40x and
        // hits the SageMaker MaxRuntime). Bounding the loop with `during(...)`
        // makes each user stop after `simulationDuration` seconds + its current
        // in-flight request, so the simulation actually ends near the configured
        // duration while still holding exactly N requests in flight throughout.
        ScenarioBuilder users = scenario("Closed Concurrency Probe")
                .during(simulationDuration).on(oneRequest);

        // Pure closed model: ramp the pool up over 30s to avoid a cold-start
        // spike, then hold exactly `concurrentUsers` in flight for the duration.
        // No .throttle() — the pool runs as fast as the server allows, so the
        // achieved req/s is the sustainable TPS at this concurrency.
        setUp(
                users.injectClosed(
                        rampConcurrentUsers(1).to(concurrentUsers).during(30),
                        constantConcurrentUsers(concurrentUsers).during(simulationDuration))
        ).protocols(httpProtocol);
    }

    @Override
    public void after() {
        log.info("Simulation finished. Avg completion tokens: {}",
                responseProcessor.getAvgCompletionTokens());
    }

    private static FeederBuilder<Object> loadJsonlFeeder() {
        try (InputStream stream = ClosedConcurrencySimulation.class
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
