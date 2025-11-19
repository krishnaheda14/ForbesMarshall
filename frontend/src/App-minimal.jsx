// Minimal App for debugging
import React from 'react';

function App() {
  return (
    <div style={{ 
      padding: '40px', 
      fontFamily: 'Arial, sans-serif',
      maxWidth: '800px',
      margin: '0 auto'
    }}>
      <h1 style={{ color: '#1e3a8a', borderBottom: '3px solid #3b82f6', paddingBottom: '10px' }}>
        🏭 CNC Scheduling System
      </h1>
      <div style={{ 
        background: '#f8f9fa', 
        padding: '20px', 
        borderRadius: '8px',
        marginTop: '20px'
      }}>
        <h2>✅ React App is Running!</h2>
        <p>The frontend server is working correctly.</p>
        <ul>
          <li>Backend API: <a href="http://localhost:8001" target="_blank">http://localhost:8001</a></li>
          <li>Frontend: <a href="http://localhost:5173" target="_blank">http://localhost:5173</a></li>
        </ul>
      </div>
      
      <div style={{ marginTop: '20px', padding: '15px', background: '#dbeafe', borderRadius: '8px' }}>
        <h3>🔧 Next Steps:</h3>
        <ol>
          <li>Verify backend is running at port 8001</li>
          <li>Check browser console for any errors (F12)</li>
          <li>Install missing dependencies if any</li>
        </ol>
      </div>

      <button 
        onClick={() => {
          fetch('http://localhost:8001/')
            .then(res => res.json())
            .then(data => alert(JSON.stringify(data, null, 2)))
            .catch(err => alert('Backend Error: ' + err.message));
        }}
        style={{
          marginTop: '20px',
          padding: '12px 24px',
          background: '#1e3a8a',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          fontSize: '16px',
          fontWeight: 'bold'
        }}
      >
        Test Backend Connection
      </button>
    </div>
  );
}

export default App;
