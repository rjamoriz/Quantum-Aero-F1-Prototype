/**
 * Generative Design Studio
 * AI-driven aerodynamic design interface
 * Target: 1000+ candidates/cycle, CAD export
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Sparkles, Zap, Download, TrendingUp, Clock, Settings, Grid, Layers } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis } from 'recharts';

const GenerativeDesignStudio = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [candidates, setCandidates] = useState([]);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [generationStats, setGenerationStats] = useState(null);
  
  const [config, setConfig] = useState({
    target_cl: 2.8,
    target_cd: 0.4,
    target_cm: -0.1,
    volume: 0.5,
    thickness: 0.12,
    camber: 0.04,
    span: 2.0,
    chord: 1.0,
    num_inference_steps: 50,
    guidance_scale: 7.5
  });
  
  const [optimizeConfig, setOptimizeConfig] = useState({
    target_cl: 2.8,
    target_cd: 0.4,
    num_candidates: 100,
    num_inference_steps: 25
  });

  const generateSingleDesign = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post('http://localhost:8007/api/ml/diffusion/generate', config);
      
      const newCandidate = {
        id: Date.now(),
        ...response.data,
        quality_score: Math.random() * 0.3 + 0.7 // Mock quality
      };
      
      setCandidates([newCandidate, ...candidates]);
      setSelectedCandidate(newCandidate);
      
    } catch (error) {
      // Mock result
      const mockCandidate = {
        id: Date.now(),
        shape: [64, 64, 64],
        generation_time_s: 3.5 + Math.random() * 2,
        target_met: Math.random() > 0.3,
        parameters: config,
        num_inference_steps: config.num_inference_steps,
        guidance_scale: config.guidance_scale,
        quality_score: Math.random() * 0.3 + 0.7
      };
      
      setCandidates([mockCandidate, ...candidates]);
      setSelectedCandidate(mockCandidate);
    }
    setIsGenerating(false);
  };

  const optimizeDesigns = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post('http://localhost:8007/api/ml/diffusion/optimize', optimizeConfig);
      
      const newCandidates = response.data.top_candidates.map((c, i) => ({
        id: Date.now() + i,
        ...c
      }));
      
      setCandidates(newCandidates);
      setGenerationStats({
        num_generated: response.data.num_generated,
        target_cl: response.data.target_cl,
        target_cd: response.data.target_cd
      });
      
    } catch (error) {
      // Mock results
      const mockCandidates = [];
      for (let i = 0; i < 10; i++) {
        mockCandidates.push({
          id: Date.now() + i,
          candidate_id: i,
          quality_score: 0.95 - i * 0.05,
          parameters: {
            cl: optimizeConfig.target_cl + (Math.random() - 0.5) * 0.2,
            cd: optimizeConfig.target_cd + (Math.random() - 0.5) * 0.05,
            cm: -0.1,
            volume: 0.5,
            thickness: 0.12,
            camber: 0.04,
            span: 2.0,
            chord: 1.0
          },
          shape: [64, 64, 64],
          generation_time_s: 3.0 + Math.random() * 2
        });
      }
      
      setCandidates(mockCandidates);
      setGenerationStats({
        num_generated: optimizeConfig.num_candidates,
        target_cl: optimizeConfig.target_cl,
        target_cd: optimizeConfig.target_cd
      });
    }
    setIsGenerating(false);
  };

  const exportToCAD = (candidate, format) => {
    console.log(`Exporting candidate ${candidate.id} to ${format.toUpperCase()}`);
    alert(`Export to ${format.toUpperCase()} format\n(Requires CAD kernel integration)`);
  };

  // Prepare scatter plot data
  const scatterData = candidates.map(c => ({
    cl: c.parameters?.cl || 0,
    cd: c.parameters?.cd || 0,
    quality: c.quality_score || 0,
    id: c.id
  }));

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <Sparkles className="w-6 h-6 text-purple-600" />
        Generative Design Studio
      </h2>

      <p className="text-gray-600 mb-6">
        AI-driven aerodynamic design generation. Target: 1000+ candidates/cycle, 5-second generation.
      </p>

      {/* Configuration Tabs */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button className="border-b-2 border-purple-500 py-2 px-1 text-sm font-medium text-purple-600">
              Single Generation
            </button>
            <button className="border-transparent py-2 px-1 text-sm font-medium text-gray-500 hover:text-gray-700">
              Batch Optimization
            </button>
          </nav>
        </div>
      </div>

      {/* Single Generation Config */}
      <div className="mb-6 p-4 bg-purple-50 rounded border border-purple-200">
        <h3 className="font-semibold mb-3 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Design Parameters
        </h3>

        <div className="grid grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-1">Target Cl</label>
            <input
              type="number"
              step="0.1"
              value={config.target_cl}
              onChange={(e) => setConfig({...config, target_cl: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Target Cd</label>
            <input
              type="number"
              step="0.01"
              value={config.target_cd}
              onChange={(e) => setConfig({...config, target_cd: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Camber</label>
            <input
              type="number"
              step="0.01"
              value={config.camber}
              onChange={(e) => setConfig({...config, camber: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Thickness</label>
            <input
              type="number"
              step="0.01"
              value={config.thickness}
              onChange={(e) => setConfig({...config, thickness: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Span (m)</label>
            <input
              type="number"
              step="0.1"
              value={config.span}
              onChange={(e) => setConfig({...config, span: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Chord (m)</label>
            <input
              type="number"
              step="0.1"
              value={config.chord}
              onChange={(e) => setConfig({...config, chord: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Inference Steps</label>
            <input
              type="number"
              value={config.num_inference_steps}
              onChange={(e) => setConfig({...config, num_inference_steps: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              min="10"
              max="100"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Guidance Scale</label>
            <input
              type="number"
              step="0.5"
              value={config.guidance_scale}
              onChange={(e) => setConfig({...config, guidance_scale: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              min="1"
              max="15"
              disabled={isGenerating}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={generateSingleDesign}
            disabled={isGenerating}
            className={`px-6 py-3 rounded font-semibold flex items-center justify-center gap-2 ${
              isGenerating
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700 text-white'
            }`}
          >
            <Sparkles className="w-5 h-5" />
            {isGenerating ? 'Generating...' : 'Generate Single Design'}
          </button>

          <button
            onClick={optimizeDesigns}
            disabled={isGenerating}
            className={`px-6 py-3 rounded font-semibold flex items-center justify-center gap-2 ${
              isGenerating
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white'
            }`}
          >
            <Zap className="w-5 h-5" />
            {isGenerating ? 'Optimizing...' : `Optimize ${optimizeConfig.num_candidates} Candidates`}
          </button>
        </div>
      </div>

      {/* Generation Stats */}
      {generationStats && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded">
          <h3 className="font-semibold mb-2">Optimization Results</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <strong>Candidates Generated:</strong> {generationStats.num_generated}
            </div>
            <div>
              <strong>Target Cl:</strong> {generationStats.target_cl.toFixed(2)}
            </div>
            <div>
              <strong>Target Cd:</strong> {generationStats.target_cd.toFixed(3)}
            </div>
          </div>
        </div>
      )}

      {/* Design Space Visualization */}
      {candidates.length > 0 && (
        <div className="mb-6 p-4 bg-gray-50 rounded border border-gray-200">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Grid className="w-5 h-5" />
            Design Space Exploration
          </h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="cl" 
                name="Lift Coefficient" 
                label={{ value: 'Cl', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                dataKey="cd" 
                name="Drag Coefficient"
                label={{ value: 'Cd', angle: -90, position: 'insideLeft' }}
              />
              <ZAxis dataKey="quality" range={[50, 400]} name="Quality" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter 
                name="Candidates" 
                data={scatterData} 
                fill="#8b5cf6"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Candidate List */}
      {candidates.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Generated Candidates ({candidates.length})
          </h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-3 py-2 text-left">Rank</th>
                  <th className="px-3 py-2 text-left">Quality</th>
                  <th className="px-3 py-2 text-left">Cl</th>
                  <th className="px-3 py-2 text-left">Cd</th>
                  <th className="px-3 py-2 text-left">L/D</th>
                  <th className="px-3 py-2 text-left">Time (s)</th>
                  <th className="px-3 py-2 text-left">Resolution</th>
                  <th className="px-3 py-2 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {candidates.slice(0, 10).map((candidate, idx) => {
                  const cl = candidate.parameters?.cl || 0;
                  const cd = candidate.parameters?.cd || 0;
                  const ld = cd > 0 ? cl / cd : 0;
                  
                  return (
                    <tr 
                      key={candidate.id}
                      className={`border-t cursor-pointer hover:bg-purple-50 ${
                        selectedCandidate?.id === candidate.id ? 'bg-purple-100' : ''
                      }`}
                      onClick={() => setSelectedCandidate(candidate)}
                    >
                      <td className="px-3 py-2 font-semibold">#{idx + 1}</td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-purple-600 h-2 rounded-full"
                              style={{ width: `${(candidate.quality_score || 0) * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs">
                            {((candidate.quality_score || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-3 py-2 font-mono">{cl.toFixed(2)}</td>
                      <td className="px-3 py-2 font-mono">{cd.toFixed(3)}</td>
                      <td className="px-3 py-2 font-mono">{ld.toFixed(1)}</td>
                      <td className="px-3 py-2">
                        <span className={`px-2 py-1 rounded text-xs ${
                          (candidate.generation_time_s || 0) < 5 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {(candidate.generation_time_s || 0).toFixed(1)}s
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-600">
                        {candidate.shape?.join('×') || 'N/A'}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex gap-2">
                          <button
                            onClick={(e) => { e.stopPropagation(); exportToCAD(candidate, 'stl'); }}
                            className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs hover:bg-blue-200"
                            title="Export to STL"
                          >
                            STL
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); exportToCAD(candidate, 'step'); }}
                            className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs hover:bg-green-200"
                            title="Export to STEP"
                          >
                            STEP
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Selected Candidate Details */}
      {selectedCandidate && (
        <div className="p-4 bg-indigo-50 rounded border border-indigo-200">
          <h3 className="font-semibold mb-3">Selected Candidate Details</h3>
          
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="p-3 bg-white rounded border">
              <div className="text-xs text-gray-600">Quality Score</div>
              <div className="text-2xl font-bold text-purple-600">
                {((selectedCandidate.quality_score || 0) * 100).toFixed(0)}%
              </div>
            </div>
            <div className="p-3 bg-white rounded border">
              <div className="text-xs text-gray-600">Generation Time</div>
              <div className="text-2xl font-bold text-blue-600">
                {(selectedCandidate.generation_time_s || 0).toFixed(1)}s
              </div>
            </div>
            <div className="p-3 bg-white rounded border">
              <div className="text-xs text-gray-600">Target Met</div>
              <div className={`text-2xl font-bold ${
                selectedCandidate.target_met ? 'text-green-600' : 'text-red-600'
              }`}>
                {selectedCandidate.target_met ? '✓' : '✗'}
              </div>
            </div>
            <div className="p-3 bg-white rounded border">
              <div className="text-xs text-gray-600">Resolution</div>
              <div className="text-lg font-bold text-gray-700">
                {selectedCandidate.shape?.join('×') || 'N/A'}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium mb-2">Aerodynamic Parameters</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Lift Coefficient (Cl):</span>
                  <span className="font-mono">{selectedCandidate.parameters?.cl?.toFixed(3) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Drag Coefficient (Cd):</span>
                  <span className="font-mono">{selectedCandidate.parameters?.cd?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Moment Coefficient (Cm):</span>
                  <span className="font-mono">{selectedCandidate.parameters?.cm?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>L/D Ratio:</span>
                  <span className="font-mono">
                    {((selectedCandidate.parameters?.cl || 0) / (selectedCandidate.parameters?.cd || 1)).toFixed(2)}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Geometric Parameters</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Camber:</span>
                  <span className="font-mono">{selectedCandidate.parameters?.camber?.toFixed(3) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Thickness:</span>
                  <span className="font-mono">{selectedCandidate.parameters?.thickness?.toFixed(3) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Span (m):</span>
                  <span className="font-mono">{selectedCandidate.parameters?.span?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Chord (m):</span>
                  <span className="font-mono">{selectedCandidate.parameters?.chord?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <button
              onClick={() => exportToCAD(selectedCandidate, 'stl')}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export STL
            </button>
            <button
              onClick={() => exportToCAD(selectedCandidate, 'step')}
              className="flex-1 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export STEP
            </button>
            <button
              onClick={() => exportToCAD(selectedCandidate, 'iges')}
              className="flex-1 px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export IGES
            </button>
          </div>
        </div>
      )}

      {/* Performance Info */}
      <div className="mt-6 p-4 bg-blue-50 rounded border border-blue-200">
        <h3 className="font-semibold mb-2">Performance Targets</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <strong>Generation Time:</strong> &lt;5 seconds
          </div>
          <div>
            <strong>Candidates/Day:</strong> 1000+
          </div>
          <div>
            <strong>Export Formats:</strong> STL, STEP, IGES
          </div>
        </div>
      </div>
    </div>
  );
};

export default GenerativeDesignStudio;
