package org.foresightrl.perf.model;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@NoArgsConstructor
@Getter
@Setter
public class ConversationMessage {
    private String role;
    private String content;
}
