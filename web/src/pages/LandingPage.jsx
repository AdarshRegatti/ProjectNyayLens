import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './LandingPage.css'

const features = [
  {
    icon: '⚖',
    title: 'Hybrid RAG Retrieval',
    desc: 'FAISS dense vector search fused with SQLite FTS5 keyword search via Reciprocal Rank Fusion — maximising recall across 298K indexed paragraphs.',
  },
  {
    icon: '🧠',
    title: 'Legal-BERT Extraction',
    desc: 'Domain-trained Legal-BERT scores every sentence globally. Top-15 sentences are selected by relevance score, then reordered by document position for coherent context.',
  },
  {
    icon: '📑',
    title: 'Legal-PEGASUS Abstraction',
    desc: 'Gap-Sentence Generation model fine-tuned on legal corpora rewrites the extracted sentences into a structured Case Overview — not headlines, but actual legal reasoning.',
  },
  {
    icon: '🔎',
    title: 'Cross-Encoder Reranking',
    desc: 'BAAI/bge-reranker-base re-scores all retrieved passages with full query context, dramatically improving precision over naive cosine similarity.',
  },
  {
    icon: '📄',
    title: 'Document-Aware QA',
    desc: 'Upload any judgment PDF. pdfplumber extracts clean text. Llama 3.1 8B is forced to answer from the document text only — no hallucinated citations.',
  },
  {
    icon: '🛡',
    title: 'Legal Guardrails',
    desc: 'System-level prompts enforce a strict legal persona. Off-topic queries are rejected. The assistant answers from evidence, not imagination.',
  },
]

const stack = [
  { layer: 'Interface', items: ['React 18', 'Vite', 'React Router'] },
  { layer: 'API', items: ['FastAPI', 'Uvicorn', 'pdfplumber'] },
  { layer: 'Extraction', items: ['Legal-BERT', 'BAAI Reranker', 'sentence-transformers'] },
  { layer: 'Summarization', items: ['Legal-PEGASUS', 'HuggingFace Transformers', 'sentencepiece'] },
  { layer: 'Retrieval', items: ['FAISS (298K vectors)', 'SQLite FTS5', 'Reciprocal Rank Fusion'] },
  { layer: 'Generation', items: ['Llama 3.1 8B (Groq)', 'Legal guardrail prompts'] },
]

const pipeline = [
  {
    id: '01',
    phase: 'Ingest',
    title: 'PDF → Clean Text',
    desc: 'pdfplumber extracts and normalizes raw judgment text. JudgmentSegmenter classifies paragraphs into Facts / Arguments / Decision sections.',
    tags: ['pdfplumber', 'JudgmentSegmenter'],
  },
  {
    id: '02',
    phase: 'Retrieve',
    title: 'Hybrid Search',
    desc: 'Parallel FAISS vector search + SQLite FTS5 keyword search. Results fused via RRF. BAAI Cross-Encoder reranks the top candidates.',
    tags: ['FAISS', 'SQLite FTS5', 'BAAI Reranker'],
  },
  {
    id: '03',
    phase: 'Extract',
    title: 'Legal-BERT Ranking',
    desc: 'All sentences in the document are scored globally. Top-15 by relevance, restored to document order, concatenated into a coherent context window.',
    tags: ['Legal-BERT', 'sentence-transformers'],
  },
  {
    id: '04',
    phase: 'Abstract',
    title: 'Legal-PEGASUS Synthesis',
    desc: 'GSG-trained transformer writes the Case Overview. Section breakdowns (Arguments, Decision) generated from extractive top-3 per section.',
    tags: ['Legal-PEGASUS', 'num_beams=4'],
  },
  {
    id: '05',
    phase: 'Answer',
    title: 'Llama 3.1 QA',
    desc: 'For conversational queries, Llama 3.1 8B (via Groq) synthesizes a cited legal answer using the retrieved and reranked passages.',
    tags: ['Llama 3.1 8B', 'Groq API'],
  },
]

const stats = [
  { value: '26,688', label: 'SC Judgments Indexed' },
  { value: '298K+', label: 'Paragraph Vectors' },
  { value: '5', label: 'AI Models in Pipeline' },
  { value: '<30s', label: 'Full Case Summary' },
]

export default function LandingPage() {
  const navigate = useNavigate()
  const [showNotice, setShowNotice] = useState(true)
  const [countdown,  setCountdown]  = useState(15)

  useEffect(() => {
    if (!showNotice) return
    const interval = setInterval(() => {
      setCountdown(c => {
        if (c <= 1) { clearInterval(interval); setShowNotice(false); return 0 }
        return c - 1
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [showNotice])

  return (
    <div className="landing">

      {/* DEPLOYMENT NOTICE */}
      {showNotice && (
        <div className="deploy-overlay">
          <div className="deploy-modal">
            <div className="deploy-hf-badge">🤗 Hugging Face Spaces</div>
            <h2 className="deploy-headline">Free Tier Deployment</h2>
            <p className="deploy-body">
              This application is running on a <strong>CPU-only</strong> Hugging Face Space.
              <br /><br />
              ⚡ <strong>RAG queries</strong> (chat questions) are <strong>instant</strong>.
              <br />
              📊 <strong>Summarization</strong> runs Legal-BERT + Legal-PEGASUS locally and may take <strong>1–2 minutes</strong> on CPU.
            </p>
            <button className="deploy-btn" onClick={() => setShowNotice(false)}>
              Continue
            </button>
            <div className="deploy-timer">Closes in {countdown}s</div>
          </div>
        </div>
      )}

      {/* NAV */}
      <nav className="nav">
        <div className="nav-logo">
          <span className="logo-mark">⚖</span>
          <span className="logo-word">NyayLens</span>
        </div>
        <div className="nav-links">
          <a href="#pipeline">Pipeline</a>
          <a href="#stack">Stack</a>
          <button className="nav-cta" onClick={() => navigate('/chat')}>
            Open Research Tool →
          </button>
        </div>
      </nav>

      {/* HERO */}
      <section className="hero">
        <div className="hero-eyebrow">Indian Supreme Court · AI Research Platform</div>
        <h1 className="hero-headline">
          Every Precedent.<br />
          <em>Instantly Understood.</em>
        </h1>
        <p className="hero-sub">
          NyayLens indexes 26,688 Supreme Court judgments and runs a five-stage AI pipeline — 
          from hybrid retrieval to legal-domain abstraction — so lawyers spend time reasoning, not reading.
        </p>
        <div className="hero-actions">
          <button className="btn-primary" onClick={() => navigate('/chat')}>
            Start Researching
          </button>
          <a className="btn-outline" href="#pipeline">View Pipeline →</a>
        </div>

        <div className="stats-row">
          {stats.map(s => (
            <div className="stat" key={s.label}>
              <span className="stat-val">{s.value}</span>
              <span className="stat-lbl">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* FEATURES */}
      <section className="features-section" id="features">
        <div className="section-label">Capabilities</div>
        <h2 className="section-heading">Precision-built for legal intelligence</h2>
        <div className="features-grid">
          {features.map(f => (
            <div className="feat-card" key={f.title}>
              <div className="feat-icon">{f.icon}</div>
              <h3 className="feat-title">{f.title}</h3>
              <p className="feat-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* PIPELINE */}
      <section className="pipeline-section" id="pipeline">
        <div className="section-label">Architecture</div>
        <h2 className="section-heading">The five-stage AI pipeline</h2>
        <p className="section-sub">Every query passes through a purpose-built sequence of domain-specific models.</p>

        <div className="pipeline-track">
          {pipeline.map((step, i) => (
            <div className="pipeline-step" key={step.id}>
              <div className="step-left">
                <div className="step-id">{step.id}</div>
                {i < pipeline.length - 1 && <div className="step-line" />}
              </div>
              <div className="step-body">
                <div className="step-phase">{step.phase}</div>
                <h3 className="step-title">{step.title}</h3>
                <p className="step-desc">{step.desc}</p>
                <div className="step-tags">
                  {step.tags.map(t => (
                    <span className="tag" key={t}>{t}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* TECH STACK */}
      <section className="stack-section" id="stack">
        <div className="section-label">Technology</div>
        <h2 className="section-heading">Full-stack breakdown</h2>
        <div className="stack-grid">
          {stack.map(s => (
            <div className="stack-col" key={s.layer}>
              <div className="stack-layer">{s.layer}</div>
              {s.items.map(item => (
                <div className="stack-item" key={item}>{item}</div>
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2 className="cta-headline">Ready to research smarter?</h2>
        <p className="cta-sub">Upload a judgment or ask any legal question — the entire pipeline runs in seconds.</p>
        <button className="btn-primary btn-lg" onClick={() => navigate('/chat')}>
          ⚖ Open NyayLens Chat
        </button>
      </section>

      {/* FOOTER */}
      <footer className="footer">
        <span className="footer-logo">⚖ NyayLens</span>
        <span className="footer-sub">
          Legal-BERT · Legal-PEGASUS · FAISS · SQLite FTS5 · Llama 3.1 · Groq · 2025
        </span>
      </footer>

      {/* FLOATING CTA */}
      <button className="float-btn" onClick={() => navigate('/chat')}>
        Open Chat →
      </button>

    </div>
  )
}
