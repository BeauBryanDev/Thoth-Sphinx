// ============================================================
// <ChatPanel /> — the SphinxChat conversation UI
// ============================================================

import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { Card, Button, Input } from "@/components/common";
import { CHAT_SUGGESTIONS } from "@/constants";
import { useChat, type UseChatReturn } from "@/hooks/useChat";
import { formatTime } from "@/utils/formatters";
import { cn } from "@/utils/cn";
import chatEyeIcon from "@/assets/chat_eye_icon.svg";
import type { ChatMessage } from "@/types";

const SphinxAvatar = () => (
  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-amber-500 to-amber-700 p-1.5">
    <img src={chatEyeIcon} alt="Thot-Sphinx" className="h-full w-full" />
  </div>
);

interface ChatPanelProps {
  /** Share chat state with sibling panels (e.g. the Topics card). */
  controller?: UseChatReturn;
  /** Overrides for other chat contexts (the reading chat drawer); defaults keep the Chat page as is. */
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  suggestions?: readonly string[];
  placeholder?: string;
  className?: string;
  /** Decoration shown on both sides of the title, midway to the card edges (mirrored on the right). */
  headerOrnament?: ReactNode;
  /** Extra content under an assistant message (tool calls, proposed corrections). */
  renderExtras?: (message: ChatMessage) => ReactNode;
}

export const ChatPanel = ({
  controller,
  title = "SphinxChat",
  subtitle = "Your AI expert on Ancient Egypt",
  icon = "𓁹",
  suggestions = CHAT_SUGGESTIONS.slice(0, 2),
  placeholder = "Ask Sphinx…",
  className,
  headerOrnament,
  renderExtras,
}: ChatPanelProps = {}) => {
  // Hooks must run unconditionally; the internal instance is simply
  // unused when the page supplies a shared controller.
  const internal = useChat();
  const { messages, isLoading, error, send } = controller ?? internal;
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!draft.trim() || isLoading) return;
    send(draft);
    setDraft("");
  };

  return (
    <Card
      title={title}
      subtitle={subtitle}
      icon={icon}
      className={cn("flex h-full flex-col", className)}
      headerBalance={Boolean(headerOrnament)}
      headerStart={headerOrnament ? <div className="mx-auto">{headerOrnament}</div> : undefined}
      headerEnd={headerOrnament ? <div className="mx-auto -scale-x-100">{headerOrnament}</div> : undefined}
      bodyClassName="flex min-h-0 flex-1 flex-col"
    >
      <div
        ref={scrollRef}
        className="mb-4 min-h-0 flex-1 space-y-4 overflow-y-auto pr-2"
        style={{ minHeight: 280 }}
      >
        {messages.map((m) => {
          const isUser = m.role === "user";
          return (
            <div
              key={m.id}
              className={cn(
                "flex gap-3",
                isUser ? "flex-row-reverse" : "flex-row",
              )}
            >
              {isUser ? (
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-amber-700/60 text-lg text-amber-100">
                  👤
                </div>
              ) : (
                <SphinxAvatar />
              )}
              <div
                className={cn(
                  "max-w-[85%] rounded-lg border px-4 py-3 text-base leading-relaxed",
                  isUser
                    ? "border-amber-600/40 bg-amber-800/40 text-amber-100"
                    : "border-amber-700/50 bg-amber-950/60 text-amber-200",
                )}
              >
                {isUser ? (
                  <p className="whitespace-pre-wrap">{m.content}</p>
                ) : (
                  /* Claude often answers in markdown (**bold**, lists,
                     headings) — render it instead of showing raw marks.
                     No images: a reply must never auto-load a URL (prompt-injection exfiltration). */
                  <div className="chat-markdown">
                    <ReactMarkdown disallowedElements={["img"]}>{m.content}</ReactMarkdown>
                  </div>
                )}
                {!isUser && renderExtras?.(m)}
                <p className="mt-2 text-right text-[10px] text-amber-500/70">
                  {formatTime(m.timestamp)}
                </p>
              </div>
            </div>
          );
        })}

        {isLoading && (
          <div className="flex gap-3">
            <SphinxAvatar />
            <div className="rounded-lg border border-amber-700/50 bg-amber-950/60 px-4 py-3 text-amber-200">
              <span className="inline-flex gap-1">
                <span className="h-2 w-2 animate-bounce rounded-full bg-amber-400 [animation-delay:-0.3s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-amber-400 [animation-delay:-0.15s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-amber-400" />
              </span>
            </div>
          </div>
        )}

        {error && (
          <p className="rounded border border-red-500/50 bg-red-900/40 px-3 py-2 text-xs text-red-200">
            {error}
          </p>
        )}
      </div>

      {/* Suggestion chips */}
      <div className="mb-3 flex flex-wrap gap-2">
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => send(s)}
            disabled={isLoading}
            className="rounded-full border border-amber-700/50 bg-amber-900/40 px-3 py-1 text-xs text-amber-200 hover:bg-amber-800/60 disabled:opacity-60"
          >
            {s}
          </button>
        ))}
      </div>

      <form onSubmit={onSubmit} className="flex gap-2">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={placeholder}
          disabled={isLoading}
          className="flex-1"
        />
        <Button type="submit" isLoading={isLoading} rightIcon="➤">
          Send
        </Button>
      </form>
    </Card>
  );
};

export default ChatPanel;
