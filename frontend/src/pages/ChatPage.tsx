// <ChatPage /> — dedicated full-page chat experience
//
// The chat state lives HERE (shared controller) so the Topics card
// and the conversation panel drive the SAME conversation — clicking
// a topic fires that question into the visible chat.

import { ChatPanel } from "@/components/chat";
import { ToolCallChips } from "@/components/chat/ToolCallChips";
import { Card } from "@/components/common";
import { CHAT_SUGGESTIONS } from "@/constants";
import { useChat } from "@/hooks/useChat";

export const ChatPage = () => {
  const chat = useChat();

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Full-height conversation: the panel stretches to the viewport
          (minus header/padding; extra clearance for the mobile tab bar)
          and the message list grows to fill it. */}
      <div className="h-[calc(100dvh-14rem)] min-h-[28rem] lg:col-span-2 lg:h-[calc(100dvh-11rem)]">
        <ChatPanel
          controller={chat}
          renderExtras={(m) =>
            m.toolCalls?.length ? (
              <div className="mt-3 border-t border-amber-800/50 pt-2">
                <ToolCallChips calls={m.toolCalls} />
              </div>
            ) : null
          }
        />
      </div>
      <Card title="Topics" subtitle="Suggested avenues of inquiry" icon="📜">
        <ul className="space-y-2 text-sm">
          {CHAT_SUGGESTIONS.map((q) => (
            <li key={q}>
              <button
                onClick={() => chat.send(q)}
                disabled={chat.isLoading}
                className="w-full rounded border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-left text-amber-200 transition-colors hover:bg-amber-900/60 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
};

export default ChatPage;
