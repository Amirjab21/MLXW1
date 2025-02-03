import { useState } from 'react'
import './App.css'

// API URL based on environment
const API_URL = import.meta.env.PROD 
  ? '/api' // Production: Use nginx proxy
  : 'http://localhost:8000' // Development: Direct connection

function App() {
  const [text, setText] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    try {
      const res = await fetch(`${API_URL}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
        setResults(null)
      } else {
        setResults(data.results)
        setError(null)
      }
    } catch (error) {
      console.error('Error:', error)
      setError('Error submitting text')
      setResults(null)
    }
  }

  return (
    <div className="container">
      <h1>Word Similarity Search</h1>
      <div className="input-container">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to find similar words..."
          className="text-input"
        />
        <button onClick={handleSubmit} className="submit-button">
          Find Similar Words
        </button>
      </div>
      
      {error && <p className="error">{error}</p>}
      
      {results && results.length > 0 && (
        <div className="results-container">
          {results.map((result, idx) => (
            <div key={idx} className="word-result">
              <h3>Similar words to "{result.word}":</h3>
              <ul className="similar-words">
                {result.similar.map(([word, similarity], wordIdx) => (
                  <li key={wordIdx}>
                    {word} <span className="similarity">({(similarity * 100).toFixed(2)}% similar)</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App
