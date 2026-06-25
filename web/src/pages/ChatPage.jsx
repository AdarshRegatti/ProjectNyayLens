import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import './ChatPage.css'

// Copy-to-clipboard helper
const copyText = (text) => navigator.clipboard?.writeText(text).catch(() => {})

const API = import.meta.env.VITE_API_URL || ''

const SLASH_COMMANDS = [
  { cmd: '/summarize', label: 'Summarize This', desc: 'Generate a full structured case brief', fill: 'Summarize this' },
  { cmd: '/decision',  label: 'Final Decision',  desc: 'What is the final decision of this case?', fill: 'What is the final decision of this case?' },
]

import localforage from 'localforage'

// ── Storage helpers ──────────────────────────────────────────────────
const STORAGE_KEY = 'nyaylens_sessions'

function newSession() {
  return { id: Date.now().toString(), name: 'New Chat', messages: [], serverFilePath: null, uploadedFileName: null }
}

export default function ChatPage() {
  const navigate = useNavigate()

  const [sessions,        setSessions]        = useState([])
  const [activeId,        setActiveId]        = useState(null)
  const [input,           setInput]           = useState('')
  const [isLoading,       setIsLoading]       = useState(false)
  const [showCommands,    setShowCommands]    = useState(false)
  const [copiedId,        setCopiedId]        = useState(null)

  const bottomRef  = useRef(null)
  const fileRef    = useRef(null)
  const textareaRef = useRef(null)

  // Derived state from active session
  const activeSession = sessions.find(s => s.id === activeId) || null
  const messages      = activeSession?.messages      || []
  const serverFilePath = activeSession?.serverFilePath || null
  const uploadedFileName = activeSession?.uploadedFileName || null

  // Load sessions on mount
  useEffect(() => {
    localforage.getItem(STORAGE_KEY).then(data => {
      if (data && data.length > 0) {
        setSessions(data)
        // Only set activeId if we don't have one yet
        setActiveId(prev => prev || data[0].id)
      }
    })
  }, [])

  // Persist whenever sessions change
  useEffect(() => { 
    if (sessions.length > 0) {
      localforage.setItem(STORAGE_KEY, sessions) 
    } else if (sessions.length === 0) {
      localforage.removeItem(STORAGE_KEY)
    }
  }, [sessions])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Session helpers ───────────────────────────────────────────────────
  const updateSession = useCallback((id, patch) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s))
  }, [])

  const createNewChat = () => {
    const session = newSession()
    setSessions(prev => [session, ...prev])
    setActiveId(session.id)
    setInput('')
  }

  const deleteSession = (e, id) => {
    e.stopPropagation()
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id)
      if (activeId === id) setActiveId(next.length > 0 ? next[0].id : null)
      return next
    })
  }

  const switchSession = (id) => {
    setActiveId(id)
    setInput('')
    setShowCommands(false)
  }

  // ── Message helpers ───────────────────────────────────────────────────
  const addMessage = (role, content, sources = [], summary = null) => {
    const msg = { role, content, sources, summary, id: Date.now() }
    setSessions(prev => prev.map(s => {
      if (s.id !== activeId) return s
      const messages = [...s.messages, msg]
      // Auto-name session from first user message
      const name = s.name === 'New Chat' && role === 'user'
        ? content.slice(0, 40) + (content.length > 40 ? '…' : '')
        : s.name
      return { ...s, messages, name }
    }))
    return msg
  }

  // ── File Upload ───────────────────────────────────────────────────────
  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    // Ensure there's an active session
    if (!activeId) createNewChat()

    updateSession(activeId, { serverFilePath: null, uploadedFileName: file.name })
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await axios.post(`${API}/api/upload`, formData)
      updateSession(activeId, { serverFilePath: res.data.filepath, uploadedFileName: file.name })
      addMessage('system', `✅ **${file.name}** uploaded and connected. You can now ask questions about this document.`)
    } catch {
      updateSession(activeId, { uploadedFileName: null })
      addMessage('system', '❌ Upload failed. Please check the server is running.')
    }
  }

  const removeFile = () => {
    updateSession(activeId, { serverFilePath: null, uploadedFileName: null })
    fileRef.current.value = ''
  }

  // ── Send ──────────────────────────────────────────────────────────────
  const handleSend = async (text = input) => {
    const question = (typeof text === 'string' ? text : input).trim()
    if (!question || isLoading) return

    // Auto-create session if none exists
    if (!activeId) {
      const session = newSession()
      setSessions(prev => [session, ...prev])
      setActiveId(session.id)
    }

    setInput('')
    setShowCommands(false)
    addMessage('user', question)
    setIsLoading(true)

    try {
      // Find the current session to get its history (excluding the message we just added since it's not in state yet)
      const currentSess = sessions.find(s => s.id === activeId) || null
      const currentPath = currentSess?.serverFilePath || null
      
      const history = (currentSess?.messages || [])
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({
          role: m.role,
          content: m.content
        }))

      const body = { message: question, top_k: 5, chat_history: history }
      if (currentPath) body.filepath = currentPath

      const res  = await axios.post(`${API}/api/chat`, body)
      const data = res.data

      if (data.answer === '__STRUCTURED_SUMMARY__' && data.summary) {
        addMessage('assistant', '__STRUCTURED_SUMMARY__', data.sources || [], data.summary)
      } else {
        addMessage('assistant', data.answer, data.sources || [])
      }
    } catch (err) {
      addMessage('assistant', `❌ **Error:** ${err.response?.data?.detail || err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (e) => {
    const val = e.target.value
    setInput(val)
    setShowCommands(val === '/' || val.startsWith('/'))
    // Auto-resize textarea
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px'
  }

  const applyCommand = (fill) => {
    setInput('')
    setShowCommands(false)
    handleSend(fill)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') { setShowCommands(false); return }
    if (e.key === 'Enter' && !e.shiftKey) {
      if (showCommands) return
      e.preventDefault()
      handleSend()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-page">

      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-logo" onClick={() => navigate('/')}>
          <span>⚖️</span>
          <span className="logo-text">NyayLens</span>
        </div>

        <button className="new-chat-btn" onClick={createNewChat}>+ New Chat</button>

        <div className="sessions-list">
          {sessions.length === 0 && (
            <div className="sessions-empty">No chats yet. Start one above.</div>
          )}
          {sessions.map(s => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeId ? 'active' : ''}`}
              onClick={() => switchSession(s.id)}
            >
              <span className="session-icon">
                {s.uploadedFileName ? '📄' : '💬'}
              </span>
              <span className="session-name">{s.name}</span>
              <button className="session-delete" onClick={(e) => deleteSession(e, s.id)} title="Delete">✕</button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-model-badge">⚡ Llama 3.1 8B · Groq</div>
          <div className="sidebar-model-badge">🧠 Legal-BERT + Legal-PEGASUS</div>
          <div className="sidebar-model-badge">🔍 FAISS · 298K vectors</div>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className="chat-main">

        {isEmpty ? (
          <div className="chat-welcome">
            <div className="welcome-icon">⚖️</div>
            <h1>How can I help you research today?</h1>
            <p>Ask any legal question, or upload a judgment document to get started.</p>
            <div className="welcome-suggestions">
              {['what is section 100', 'Grant of Bail', 'Cancellation of Bail'].map(s => (
                <button key={s} className="welcome-pill" onClick={() => handleSend(s)}>{s}</button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map(msg => (
              <div key={msg.id} className={`message-row ${msg.role}`}>
                <div className="msg-avatar">{msg.role === 'user' ? '👤' : '⚖️'}</div>
                <div className="msg-bubble">
                  {msg.summary ? (
                    <div className="structured-summary">
                      {msg.summary.overview && (
                        <div className="summary-overview">
                          <div className="summary-section-label">📋 Case Overview</div>
                          <p>{msg.summary.overview}</p>
                        </div>
                      )}
                      {Object.entries(msg.summary)
                        .filter(([k]) => k !== 'overview')
                        .map(([section, text]) => (
                          <div className="summary-section" key={section}>
                            <div className="summary-section-label">{section.toUpperCase()}</div>
                            <p>{text}</p>
                          </div>
                        ))
                      }
                    </div>
                  ) : (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  )}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-block">
                      <div className="sources-title">📚 Sources</div>
                      {msg.sources.map((s, i) => (
                        <span key={i} className="source-tag">
                          [{i+1}] {s.judgment_id} {s.score < 1 && <span className="source-score">({s.score?.toFixed(3)})</span>}
                        </span>
                      ))}
                    </div>
                  )}
                  {msg.role === 'assistant' && !msg.summary && (
                    <button
                      className={`copy-btn ${copiedId === msg.id ? 'copied' : ''}`}
                      onClick={() => {
                        copyText(msg.content)
                        setCopiedId(msg.id)
                        setTimeout(() => setCopiedId(null), 2000)
                      }}
                    >
                      {copiedId === msg.id ? '✓ Copied!' : '⎘ Copy'}
                    </button>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message-row assistant">
                <div className="msg-avatar">⚖️</div>
                <div className="msg-bubble loading-bubble">
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}

        {/* ── INPUT ── */}
        <div className="input-area">
          {showCommands && (
            <div className="slash-menu">
              <div className="slash-menu-title">Commands</div>
              {SLASH_COMMANDS.map(c => (
                <button key={c.cmd} className="slash-item" onMouseDown={() => applyCommand(c.fill)}>
                  <span className="slash-cmd">{c.cmd}</span>
                  <div className="slash-info">
                    <span className="slash-label">{c.label}</span>
                    <span className="slash-desc">{c.desc}</span>
                  </div>
                </button>
              ))}
            </div>
          )}

          {uploadedFileName && (
            <div className="file-pill">
              📄 {uploadedFileName}
              {serverFilePath
                ? <span className="file-status connected">● Connected</span>
                : <span className="file-status uploading">Uploading…</span>
              }
              <button onClick={removeFile} className="file-remove">✕</button>
            </div>
          )}

          <div className="input-box">
            <button className="attach-btn" onClick={() => fileRef.current.click()} title="Upload a judgment PDF">📎</button>
            <input type="file" ref={fileRef} accept=".txt,.pdf" onChange={handleFileUpload} style={{ display: 'none' }} />
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder="Ask a legal question, type / for commands, or upload a document…"
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <button
              className={`send-btn ${(!input.trim() || isLoading) ? 'disabled' : ''}`}
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
            >↑</button>
          </div>
          <div className="input-hint">NyayLens answers legal questions only · Enter to send · Shift+Enter for new line</div>
        </div>

      </main>
    </div>
  )
}
