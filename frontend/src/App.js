// /var/www/project-root/frontend/src/App.js

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [annotatedImage, setAnnotatedImage] = useState('');
  const [report, setReport] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setStatusMessage("Please select a file first.");
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setStatusMessage(response.data.message);
      setReport(response.data.report);
      setAnnotatedImage(response.data.annotated_image);
    } catch (error) {
      console.error(error);
      setStatusMessage("Error uploading image");
    }
  };

  return (
    <div className="page-container">
      <header className="hero-section">
        <div className="hero-overlay">
          <h1 className="hero-title">Circuit Grader</h1>
          <p className="hero-subtitle">
            Upload your circuit diagram and get an instant professional grading report.
          </p>
        </div>
      </header>

      <main className="main-content">
        <div className="upload-card">
          <h2 className="card-title">Upload Your Circuit</h2>
          <p className="card-description">
            Select an image file and click "Upload &amp; Grade" to see the annotated results and detailed report.
          </p>

          {/* Custom file upload button */}
          <label className="custom-file-upload">
            <input
              type="file"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            Choose File
          </label>

          {/* Display the selected file name if any */}
          {fileName && <span className="file-name">{fileName}</span>}

          {/* Upload button */}
          <button className="upload-button" onClick={handleUpload}>
            Upload &amp; Grade
          </button>

          {statusMessage && <p className="status-message">{statusMessage}</p>}
        </div>

        {/* Annotated image display */}
        {annotatedImage && (
          <div className="result-section">
            <h3>Annotated Image</h3>
            <img
              className="annotated-image"
              src={annotatedImage}
              alt="Annotated Circuit"
            />
          </div>
        )}

        {/* Grading report display */}
        {report && (
          <div className="report-section">
            <h3>Circuit Grading Report</h3>
            <table className="report-table">
              <thead>
                <tr>
                  <th>Criterion</th>
                  <th>Score</th>
                  <th>Analysis</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Overall Score</td>
                  <td>{report["Overall Score"]}%</td>
                  <td>Final circuit grading result.</td>
                </tr>
                <tr>
                  <td>Drawing Clarity</td>
                  <td>{report["Drawing Clarity"]}%</td>
                  <td>{report.Analysis["Drawing Clarity"]}</td>
                </tr>
                <tr>
                  <td>Labeling Accuracy</td>
                  <td>{report["Labeling Accuracy"]}%</td>
                  <td>{report.Analysis["Labeling Accuracy"]}</td>
                </tr>
                <tr>
                  <td>Voltage Accuracy</td>
                  <td>
                    {typeof report["Voltage Accuracy"] === 'number'
                      ? report["Voltage Accuracy"] + '%'
                      : 'Skipped'}
                  </td>
                  <td>{report.Analysis["Voltage Accuracy"]}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </main>

      <footer className="site-footer">
        <p>&copy; {new Date().getFullYear()} Genesis Creative Collective Inspired</p>
      </footer>
    </div>
  );
}

export default App;

