package org.foresightrl.perf.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * OpenAI-compatible chat completion request body.
 *
 * Defaults are loaded once at classload time from `chat-config.json` on the
 * classpath. Per-request fields can be overridden in the simulation
 * (e.g. {@code chatRequest.setMax_tokens(...)}).
 */
@NoArgsConstructor
@Getter
@Setter
public class ChatRequest {
    private static final ChatConfig config = loadConfig();

    private String model = config.model;
    private List<ConversationMessage> messages;
    private Integer n = config.n;
    private Double temperature = config.temperature;
    private Double top_p = config.top_p;
    private Integer top_k = config.top_k;
    private Double repetition_penalty = config.repetition_penalty;
    private Integer max_tokens = config.max_tokens;

    private static ChatConfig loadConfig() {
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream inputStream = ChatRequest.class.getClassLoader().getResourceAsStream("chat-config.json")) {
            if (inputStream == null) {
                throw new RuntimeException("chat-config.json not found in resources folder");
            }
            return mapper.readValue(inputStream, ChatConfig.class);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load chat configuration from chat-config.json", e);
        }
    }

    @Getter
    @Setter
    @NoArgsConstructor
    public static class ChatConfig {
        private String model;
        private Integer n;
        private Double temperature;
        private Double top_p;
        private Integer top_k;
        private Double repetition_penalty;
        private Integer max_tokens;
    }
}
