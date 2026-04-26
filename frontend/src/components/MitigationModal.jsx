import React, { useState } from 'react';
import { mitigateDataset } from '../api';

export default function MitigationModal({ show, onClose, targetColumn, sensitiveColumn }) {
  const [method, setMethod] = useState("reweighing");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!show) return null;

  const handleApply = async () => {
    if (!file) {
      setError("Please select the original CSV file.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const blob = await mitigateDataset({
        file,
        targetColumn,
        sensitiveColumn,
        method
      });
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mitigated_${file.name}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modalOverlay" onClick={onClose}>
      <div className="modalContent animate-slideUp" onClick={e => e.stopPropagation()} style={{background: '#fff', padding: '30px', maxWidth: '500px', margin: '100px auto', border: '1px solid #e5e5e5'}}>
        <h2>Active Bias Mitigation</h2>
        <p style={{marginBottom: '20px', color: '#666'}}>Select a method to reduce bias in your dataset. You will need to re-upload your original CSV to apply the transformation.</p>
        
        {error && <div className="errorMessage" style={{marginBottom: '15px', color: 'red'}}>{error}</div>}
        
        <div className="formGroup" style={{marginBottom: '20px'}}>
          <label style={{display: 'block', fontWeight: 'bold', marginBottom: '8px'}}>Mitigation Technique</label>
          <select value={method} onChange={e => setMethod(e.target.value)} style={{width: '100%', padding: '10px', border: '1px solid #ccc'}}>
            <option value="reweighing">Reweighing (Adds fairness weights)</option>
            <option value="dir">Disparate Impact Remover (Transforms features)</option>
          </select>
        </div>
        
        <div className="formGroup" style={{marginBottom: '30px'}}>
          <label style={{display: 'block', fontWeight: 'bold', marginBottom: '8px'}}>Original Dataset (CSV)</label>
          <input 
            type="file" 
            accept=".csv" 
            onChange={e => setFile(e.target.files[0])}
            style={{width: '100%'}} 
          />
        </div>

        <div className="modalActions" style={{display: 'flex', gap: '10px', justifyContent: 'flex-end'}}>
          <button className="secondaryButton" onClick={onClose} disabled={loading}>Cancel</button>
          <button 
            className="primaryButton" 
            onClick={handleApply} 
            disabled={loading}
          >
            {loading ? "Processing..." : "Apply & Download"}
          </button>
        </div>
      </div>
    </div>
  );
}
