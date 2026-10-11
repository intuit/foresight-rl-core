package org.foresightrl.perf;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.foresightrl.perf.model.ChatRequest;
import org.foresightrl.perf.model.ConversationMessage;
import io.gatling.javaapi.core.Session;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.function.Function;

/**
 * Reusable chat-completion request body templates for the load simulations.
 *
 * The system message is loaded from `system-message.txt` on the classpath
 * (typically supplied by the user via the SageMaker data channel and copied
 * into `src/gatling/resources/` by the perf-test runner).
 *
 * {@link #systemMessage()} is public so extension packages can add templates
 * with different sampling parameters without duplicating the loading logic;
 * see the perf-test docs on shipping your own simulations.
 */
@Slf4j
public class ChatRequestTemplates {
    private static final String SYSTEM_MESSAGE = loadSystemMessage();

    /** The shared system message, or "" when no `system-message.txt` was supplied. */
    public static String systemMessage() {
        return SYSTEM_MESSAGE;
    }

    private static String loadSystemMessage() {
        // `system-message.txt` is only required by simulations that build
        // their request body from a ChatRequestTemplates template.
        // Other simulations (e.g. JsonlChatCompletionsSimulation) embed the
        // system message in each JSONL record and don't need a separate file.
        // Returning "" instead of throwing keeps class-loading lazy-tolerant
        // so users can run the JSONL simulation without supplying the file.
        try (InputStream inputStream = ChatRequestTemplates.class.getClassLoader().getResourceAsStream("system-message.txt")) {
            if (inputStream == null) {
                log.warn("system-message.txt not found on classpath; "
                        + "ChatRequestTemplates templates will use an empty system message. "
                        + "Supply system-message.txt via the data channel if you use the CSV-based simulations.");
                return "";
            }
            return new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            log.warn("Failed to load system message from system-message.txt: {}", e.getMessage());
            return "";
        }
    }

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static final Function<Session, String> template = session -> {
        String requestBody = "";

        try {
            String prompt = session.getString("prompt");

            ChatRequest chatRequest = new ChatRequest();
            chatRequest.setMax_tokens(600);

            ConversationMessage systemMessage = new ConversationMessage();
            systemMessage.setRole(Constants.ROLE_SYSTEM);
            systemMessage.setContent(SYSTEM_MESSAGE);

            ConversationMessage conversationMessage = new ConversationMessage();
            conversationMessage.setRole(Constants.ROLE_USER);
            conversationMessage.setContent(prompt);

            chatRequest.setMessages(Arrays.asList(systemMessage, conversationMessage));

            requestBody = objectMapper.writeValueAsString(chatRequest);
        } catch (Exception ex) {
            log.error("Error building request body : " + ex.getMessage());
        }

        return requestBody;
    };

}
