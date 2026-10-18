package org.foresightrl.perf;

/**
 * URL constants for local vLLM serving used by the simulations.
 *
 * The simulations here only ever target the `vllm serve` process the perf-test
 * runner starts alongside them, so there are no remote endpoints and no
 * credentials -- both were dropped when this was vendored from its original
 * internal harness.
 */
public final class Constants {
    public static final String BASE_URL_LOCAL = "http://localhost:8080";
    public static final String MODEL_URL_LOCAL = "/v1/chat/completions";

    public static final String ROLE_SYSTEM = "system";
    public static final String ROLE_USER = "user";
    public static final String ROLE_ASSISTANT = "assistant";

    private Constants() {
    }
}
