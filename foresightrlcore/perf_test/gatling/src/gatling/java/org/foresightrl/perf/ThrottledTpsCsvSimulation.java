package org.foresightrl.perf;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.gatling.javaapi.core.ChainBuilder;
import io.gatling.javaapi.core.FeederBuilder;
import io.gatling.javaapi.core.ScenarioBuilder;
import io.gatling.javaapi.core.Simulation;
import io.gatling.javaapi.http.HttpProtocolBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.gatling.javaapi.core.CoreDsl.*;
import static io.gatling.javaapi.http.HttpDsl.http;
import static io.gatling.javaapi.http.HttpDsl.status;

/**
 * Standard TPS-throttled chat-completion load test against a local vLLM
 * server on http://localhost:8080. Prompts come from `prompts.csv`.
 *
 * This is the default simulation and the baseline for a first perf run: it
 * needs only a CSV of prompts and a `chat-config.json`, both supplied through
 * the SageMaker data channel.
 *
 * Driven by JVM system properties:
 *   -DsimulationTps=N         requests-per-second target
 *   -DsimulationDuration=SEC  steady-state duration in seconds
 */
public class ThrottledTpsCsvSimulation extends Simulation {
    private static final Logger log = LoggerFactory.getLogger(ThrottledTpsCsvSimulation.class);

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ChatCompletionResponseProcessor chatCompletionResponseProcessor = new ChatCompletionResponseProcessor();

    FeederBuilder<String> csvFeeder = csv("prompts.csv").random();

    HttpProtocolBuilder httpProtocol = http.baseUrl(Constants.BASE_URL_LOCAL)
            .acceptHeader("application/json")
            .contentTypeHeader("application/json");

    ChainBuilder invocations =
            feed(csvFeeder).exec(
                            http("Invocations")
                                    .post(Constants.MODEL_URL_LOCAL)
                                    .body(StringBody(ChatRequestTemplates.template))
                                    .check(status().is(200))
                                    .check(bodyString().saveAs("responseBody"),
                                            status().saveAs("responseStatus")
                                    ))
                    .exec(session -> {
                        // Sessions in flight when the simulation's max-duration
                        // cutoff fires never complete the .check(...) block, so
                        // responseBody / responseStatus are absent. Skip those
                        // to avoid noisy NoSuchElementException at shutdown.
                        if (!session.contains("responseStatus")) {
                            return session;
                        }
                        String responseBody = session.getString("responseBody");
                        Integer responseStatus = session.getInt("responseStatus");
                        chatCompletionResponseProcessor.processResponse(null, responseBody, responseStatus, null, null);
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
                    "ThrottledTpsCsvSimulation requires -DsimulationTps and -DsimulationDuration JVM properties");
        }

        Integer concurrentUsers = simulationTps * 100;
        log.info("Concurrent Users: " + concurrentUsers);

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
        log.info("Simulation finished. Avg completion tokens: {}", chatCompletionResponseProcessor.getAvgCompletionTokens());
    }
}
