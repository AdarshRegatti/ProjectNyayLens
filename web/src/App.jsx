import { useState } from 'react'
import axios from 'axios'
import './index.css'

function App() {
  // Chat State
  const [question, setQuestion] = useState('')
  const [chatAnswer, setChatAnswer] = useState('')
  const [chatSources, setChatSources] = useState([])
  const [isChatLoading, setIsChatLoading] = useState(false)

  // Summary State
  const [filepath, setFilepath] = useState('data/processed/extracted/texts/1950_Abdulla_Ahmed_vs_Animendra_Kissen_Mitter_on_14_March_1950_1.txt')
  const [summary, setSummary] = useState('')
  const [isSummaryLoading, setIsSummaryLoading] = useState(false)

  const handleChat = async () => {
    if (!question) return;
    setIsChatLoading(true);
    setChatAnswer('');
    
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/chat', {
        question: question,
        top_k: 5
      });
      setChatAnswer(response.data.answer);
      setChatSources(response.data.sources || []);
    } catch (error) {
      setChatAnswer('Error: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsChatLoading(false);
    }
  }

  const handleSummarize = async () => {
    if (!filepath) return;
    setIsSummaryLoading(true);
    setSummary('');
    
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/summarize', {
        filepath: filepath
      });
      // The API returns { summary: "..." }
      setSummary(response.data.summary);
    } catch (error) {
      setSummary('Error: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsSummaryLoading(false);
    }
  }

  return (
    <>
      <div className="header-brand">Nyay<span>Lens</span></div>
      
      <div className="container">
        {/* Chat / RAG Panel */}
        <div className="card">
          <div>
            <h2 className="card-title">Legal Synthesizer</h2>
            <p className="card-desc">Ask a legal question across 26,000 precedents.</p>
          </div>
          
          <div className="input-group">
            <textarea 
              placeholder="e.g. What are the conditions for anticipatory bail?"
              rows="3"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <button onClick={handleChat} disabled={isChatLoading || !question}>
              {isChatLoading ? <span className="loader"></span> : 'Search & Synthesize'}
            </button>
          </div>
          
          <div className="result-box">
            {chatAnswer ? (
              <>
                <div style={{ whiteSpace: 'pre-wrap', marginBottom: '1.5rem', lineHeight: '1.6' }}>
                  {chatAnswer}
                </div>
                {chatSources && chatSources.length > 0 && (
                  <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                    <strong style={{ color: '#fff', fontSize: '1rem' }}>Sources & Precedents Cited:</strong>
                    <ul style={{ paddingLeft: '1.2rem', marginTop: '0.8rem', color: 'var(--text-muted)' }}>
                      {chatSources.map((src, i) => (
                        <li key={i} style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                          <span style={{ color: 'var(--primary)' }}>[{i + 1}]</span> <b>{src.judgment_id}</b> 
                          <span style={{ opacity: 0.6 }}> (Score: {src.score.toFixed(3)})</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <span style={{color: '#8b949e'}}>The AI answer will appear here...</span>
            )}
          </div>
        </div>

        {/* Summarizer Panel */}
        <div className="card">
          <div>
            <h2 className="card-title">Document Summarizer</h2>
            <p className="card-desc">Extract an abstractive summary of a judgment.</p>
          </div>
          
          <div className="input-group">
            <input 
              type="text" 
              placeholder="Enter file path relative to root"
              value={filepath}
              onChange={(e) => setFilepath(e.target.value)}
            />
            <button onClick={handleSummarize} disabled={isSummaryLoading || !filepath}>
              {isSummaryLoading ? <span className="loader"></span> : 'Generate Summary'}
            </button>
          </div>
          
          <div className="result-box">
            {summary ? (
              typeof summary === 'object' ? (
                Object.entries(summary).map(([section, text]) => (
                  <div key={section} style={{ marginBottom: '1.5rem' }}>
                    <h3 style={{ 
                      color: 'var(--primary)', 
                      marginTop: 0, 
                      marginBottom: '0.5rem',
                      textTransform: 'uppercase',
                      fontSize: '1rem',
                      letterSpacing: '1px'
                    }}>
                      [{section}]
                    </h3>
                    <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{text}</p>
                  </div>
                ))
              ) : (
                <span style={{color: '#ff7b72'}}>{summary}</span>
              )
            ) : (
              <span style={{color: '#8b949e'}}>The structured summary will appear here...</span>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

export default App
