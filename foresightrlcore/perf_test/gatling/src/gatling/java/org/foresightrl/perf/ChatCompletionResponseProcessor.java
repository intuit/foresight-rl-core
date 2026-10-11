package org.foresightrl.perf;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Parses chat-completion responses and tracks aggregate completion-token
 * statistics across the simulation run.
 */
@Slf4j
public class ChatCompletionResponseProcessor {
    private final AtomicInteger completionsCount = new AtomicInteger(0);
    private final AtomicLong totalCompletionTokens = new AtomicLong(0);

    @Getter
    private Double avgCompletionTokens = 0.0;

    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * @param traceId optional correlation id, logged only on a non-200 response.
     */
    public void processResponse(String requestBody, String response, Integer responseStatus, String traceId, Integer responseTime) {
        try {
            if (responseStatus != null && responseStatus != 200) {
                log.info("Request body: " + StringUtils.defaultString(requestBody, ""));
                log.info("Response status: " + StringUtils.defaultString(responseStatus.toString(), ""));
                log.info("Trace id: " + StringUtils.defaultString(traceId, ""));
            }

            if (!StringUtils.isEmpty(response)) {
                log.debug("Response: " + response);

                JsonNode completionNode = objectMapper.readTree(response);
                JsonNode usageNode = completionNode.path("usage");
                log.debug("Usage: " + usageNode.toString());

                long completionTokens = usageNode.path("completion_tokens").asLong();

                long totalCompletionTokensLocal = totalCompletionTokens.addAndGet(completionTokens);
                int completionTokensLocal = completionsCount.incrementAndGet();

                avgCompletionTokens = (double) totalCompletionTokensLocal / (double) completionTokensLocal;

                if (responseTime != null) {
                    log.debug("Response time: {} ms, completion tokens: {}", responseTime, completionTokens);
                }

                if (completionsCount.get() % 10 == 0) {
                    log.info("Avg completion tokens: {}", avgCompletionTokens);
                }
            }
        } catch (Exception ex) {
            log.error("Error parsing response: " + ex.getMessage(), ex);
        }
    }
}
