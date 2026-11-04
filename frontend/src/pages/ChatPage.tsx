// <ChatPage /> — dedicated full-page chat experience
//
// The chat state lives HERE (shared controller) so the Topics card
// and the conversation panel drive the SAME conversation — clicking
// a topic fires that question into the visible chat. The right column
// holds a short Topics card above a taller one with the book passages
// behind the last reply.

import { ChatPanel } from "@/components/chat";
import { Chunks } from "@/components/chat/Chunks";
import { ToolCallChips } from "@/components/chat/ToolCallChips";
import { Card } from "@/components/common";
import { CHAT_SUGGESTIONS } from "@/constants";
import { useChat } from "@/hooks/useChat";

export const ChatPage = () => {
  const chat = useChat();
  const lastReply = [...chat.messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,4fr)_minmax(0,1fr)]">
      {/* Full-height conversation: the panel stretches to the viewport
          (minus header/padding; extra clearance for the mobile tab bar)
          and the message list grows to fill it. Chat 4fr : side 1fr. */}
      <div className="h-[calc(100dvh-14rem)] min-h-[28rem] lg:h-[calc(100dvh-11rem)]">
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

      {/* Side column: a short Topics card (3) above a tall Scrolls card (7); each scrolls on its own. */}
      <div className="flex flex-col gap-6 lg:h-[calc(100dvh-11rem)]">
        <Card
          title="Topics"
          icon="📜"
          className="flex min-h-0 flex-col lg:flex-[3]"
          bodyClassName="min-h-0 flex-1 overflow-y-auto p-4"
          padded={false}
        >
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

        <Card
          title="Scrolls"
          icon="𓏞"
          className="flex min-h-0 flex-col lg:flex-[7]"
          bodyClassName="min-h-0 flex-1 overflow-y-auto p-4"
          padded={false}
        >
          <Chunks passages={lastReply?.bookPassages ?? []} loading={chat.isLoading} />
        </Card>
      </div>
    </div>
  );
};

export default ChatPage;
