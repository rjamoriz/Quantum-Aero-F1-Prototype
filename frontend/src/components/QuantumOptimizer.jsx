/**
 * Quantum Optimization Component
 * Create and manage QUBO/QAOA optimization problems from synthetic datasets
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Grid,
  LinearProgress,
  TextField,
  Typography,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stepper,
  Step,
  StepLabel
} from '@mui/material';
import {
  PlayArrow,
  Delete,
  Visibility,
  Refresh,
  Science,
  TrendingUp
} from '@mui/icons-material';

const API_BASE = 'http://localhost:8000/api/quantum';
const DATASET_API = 'http://localhost:8000/api/synthetic-data';

const QuantumOptimizer = () => {
  const [problems, setProblems] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [selectedProblem, setSelectedProblem] = useState(null);
  
  // New problem configuration
  const [newProblem, setNewProblem] = useState({
    name: '',
    description: '',
    source_dataset_id: '',
    objective: 'maximize_L_over_D',
    quantum_backend: 'qiskit_simulator',
    num_shots: 1000,
    design_variables: []
  });

  // Fetch problems and datasets
  useEffect(() => {
    fetchProblems();
    fetchDatasets();
    const interval = setInterval(fetchProblems, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchProblems = async () => {
    try {
      const response = await axios.get(`${API_BASE}/qubo-problems?limit=50`);
      setProblems(response.data.data);
    } catch (error) {
      console.error('Error fetching problems:', error);
    }
  };

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${DATASET_API}/datasets?status=completed&limit=100`);
      setDatasets(response.data.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const createProblem = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE}/qubo-problems`, newProblem);
      
      // Automatically formulate QUBO
      await axios.post(`${API_BASE}/qubo-problems/${response.data.data._id}/formulate`);
      
      setCreateDialogOpen(false);
      fetchProblems();
      
      // Reset form
      setNewProblem({
        name: '',
        description: '',
        source_dataset_id: '',
        objective: 'maximize_L_over_D',
        quantum_backend: 'qiskit_simulator',
        num_shots: 1000,
        design_variables: []
      });
    } catch (error) {
      console.error('Error creating problem:', error);
      alert('Error: ' + error.response?.data?.error);
    } finally {
      setLoading(false);
    }
  };

  const executeOptimization = async (problemId) => {
    try {
      await axios.post(`${API_BASE}/qubo-problems/${problemId}/execute`);
      fetchProblems();
    } catch (error) {
      console.error('Error executing optimization:', error);
      alert('Error: ' + error.response?.data?.error);
    }
  };

  const deleteProblem = async (problemId) => {
    if (!window.confirm('Are you sure you want to delete this optimization problem?')) {
      return;
    }
    
    try {
      await axios.delete(`${API_BASE}/qubo-problems/${problemId}`);
      fetchProblems();
    } catch (error) {
      console.error('Error deleting problem:', error);
    }
  };

  const viewDetails = async (problemId) => {
    try {
      const response = await axios.get(`${API_BASE}/qubo-problems/${problemId}`);
      setSelectedProblem(response.data.data);
      setDetailsDialogOpen(true);
    } catch (error) {
      console.error('Error fetching details:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'primary';
      case 'failed': return 'error';
      case 'ready': return 'info';
      case 'formulating': return 'warning';
      default: return 'default';
    }
  };

  const getStatusStep = (status) => {
    const steps = ['created', 'formulating', 'ready', 'running', 'completed'];
    return steps.indexOf(status);
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" component="h1">
          ⚛️ Quantum Aerodynamic Optimization
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchProblems}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Science />}
            onClick={() => setCreateDialogOpen(true)}
          >
            New Optimization
          </Button>
        </Box>
      </Box>

      {/* Info Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Problems
              </Typography>
              <Typography variant="h3">
                {problems.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Running
              </Typography>
              <Typography variant="h3" color="primary">
                {problems.filter(p => p.status === 'running').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Completed
              </Typography>
              <Typography variant="h3" color="success.main">
                {problems.filter(p => p.status === 'completed').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg Improvement
              </Typography>
              <Typography variant="h3" color="success.main">
                {problems.filter(p => p.results?.improvement_over_classical).length > 0
                  ? (problems
                      .filter(p => p.results?.improvement_over_classical)
                      .reduce((sum, p) => sum + p.results.improvement_over_classical, 0) /
                    problems.filter(p => p.results?.improvement_over_classical).length).toFixed(1)
                  : '0'}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Problems Table */}
      <Card>
        <CardHeader title="Optimization Problems" />
        <CardContent>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Objective</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Backend</TableCell>
                  <TableCell>Qubits</TableCell>
                  <TableCell>Best L/D</TableCell>
                  <TableCell>Improvement</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {problems.map((problem) => (
                  <TableRow key={problem._id}>
                    <TableCell>
                      <Typography variant="subtitle2">{problem.name}</Typography>
                      {problem.description && (
                        <Typography variant="caption" color="textSecondary">
                          {problem.description}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={problem.formulation.objective.replace('_', ' ')}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={problem.status}
                        color={getStatusColor(problem.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        {problem.quantum_config.backend}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {problem.quantum_config.num_qubits || 'N/A'}
                    </TableCell>
                    <TableCell>
                      {problem.results?.optimal_solution?.predicted_performance?.L_over_D?.toFixed(2) || 'N/A'}
                    </TableCell>
                    <TableCell>
                      {problem.results?.improvement_over_classical ? (
                        <Chip
                          icon={<TrendingUp />}
                          label={`+${problem.results.improvement_over_classical.toFixed(1)}%`}
                          color="success"
                          size="small"
                        />
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => viewDetails(problem._id)}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      
                      {problem.status === 'ready' && (
                        <Tooltip title="Execute">
                          <IconButton
                            size="small"
                            onClick={() => executeOptimization(problem._id)}
                          >
                            <PlayArrow />
                          </IconButton>
                        </Tooltip>
                      )}
                      
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          onClick={() => deleteProblem(problem._id)}
                        >
                          <Delete />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Create Problem Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create Quantum Optimization Problem</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Problem Name"
                value={newProblem.name}
                onChange={(e) => setNewProblem({ ...newProblem, name: e.target.value })}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={newProblem.description}
                onChange={(e) => setNewProblem({ ...newProblem, description: e.target.value })}
                multiline
                rows={2}
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Source Dataset</InputLabel>
                <Select
                  value={newProblem.source_dataset_id}
                  onChange={(e) => setNewProblem({ ...newProblem, source_dataset_id: e.target.value })}
                  required
                >
                  {datasets.map((dataset) => (
                    <MenuItem key={dataset._id} value={dataset._id}>
                      {dataset.name} ({dataset.statistics?.n_samples || 0} samples)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Optimization Objective</InputLabel>
                <Select
                  value={newProblem.objective}
                  onChange={(e) => setNewProblem({ ...newProblem, objective: e.target.value })}
                >
                  <MenuItem value="maximize_L_over_D">Maximize L/D Ratio</MenuItem>
                  <MenuItem value="minimize_drag">Minimize Drag</MenuItem>
                  <MenuItem value="maximize_downforce">Maximize Downforce</MenuItem>
                  <MenuItem value="balance_optimization">Optimize Balance</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Quantum Backend</InputLabel>
                <Select
                  value={newProblem.quantum_backend}
                  onChange={(e) => setNewProblem({ ...newProblem, quantum_backend: e.target.value })}
                >
                  <MenuItem value="qiskit_simulator">Qiskit Simulator (Fast)</MenuItem>
                  <MenuItem value="ibm_quantum">IBM Quantum (Real Hardware)</MenuItem>
                  <MenuItem value="aws_braket">AWS Braket</MenuItem>
                  <MenuItem value="azure_quantum">Azure Quantum</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                type="number"
                label="Number of Shots"
                value={newProblem.num_shots}
                onChange={(e) => setNewProblem({ ...newProblem, num_shots: parseInt(e.target.value) })}
                helperText="More shots = better accuracy but longer runtime"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Estimated Runtime:</strong> {' '}
                  {newProblem.quantum_backend === 'qiskit_simulator' ? '5-10 minutes' : '30-60 minutes'}
                </Typography>
                <Typography variant="body2">
                  <strong>Qubits Required:</strong> ~16-20 qubits
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={createProblem}
            disabled={!newProblem.name || !newProblem.source_dataset_id || loading}
          >
            Create & Formulate
          </Button>
        </DialogActions>
      </Dialog>

      {/* Details Dialog */}
      <Dialog
        open={detailsDialogOpen}
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        {selectedProblem && (
          <>
            <DialogTitle>{selectedProblem.name}</DialogTitle>
            <DialogContent>
              <Grid container spacing={3}>
                {/* Status Stepper */}
                <Grid item xs={12}>
                  <Stepper activeStep={getStatusStep(selectedProblem.status)}>
                    <Step><StepLabel>Created</StepLabel></Step>
                    <Step><StepLabel>Formulating</StepLabel></Step>
                    <Step><StepLabel>Ready</StepLabel></Step>
                    <Step><StepLabel>Running</StepLabel></Step>
                    <Step><StepLabel>Completed</StepLabel></Step>
                  </Stepper>
                </Grid>
                
                {/* Configuration */}
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Configuration
                  </Typography>
                  <Typography variant="body2">
                    <strong>Objective:</strong> {selectedProblem.formulation.objective}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Backend:</strong> {selectedProblem.quantum_config.backend}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Qubits:</strong> {selectedProblem.quantum_config.num_qubits}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Shots:</strong> {selectedProblem.quantum_config.num_shots}
                  </Typography>
                </Grid>
                
                {/* Results */}
                {selectedProblem.results?.optimal_solution && (
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Optimal Solution
                    </Typography>
                    <Typography variant="body2">
                      <strong>L/D:</strong> {selectedProblem.results.optimal_solution.predicted_performance.L_over_D.toFixed(3)}
                    </Typography>
                    <Typography variant="body2">
                      <strong>CL:</strong> {selectedProblem.results.optimal_solution.predicted_performance.CL.toFixed(3)}
                    </Typography>
                    <Typography variant="body2">
                      <strong>CD:</strong> {selectedProblem.results.optimal_solution.predicted_performance.CD.toFixed(4)}
                    </Typography>
                    {selectedProblem.results.improvement_over_classical && (
                      <Typography variant="body2" color="success.main">
                        <strong>Improvement:</strong> +{selectedProblem.results.improvement_over_classical.toFixed(1)}% vs classical
                      </Typography>
                    )}
                  </Grid>
                )}
                
                {/* Quantum Metrics */}
                {selectedProblem.results?.quantum_metrics && (
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      Quantum Metrics
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="textSecondary">Circuit Depth</Typography>
                        <Typography variant="h6">{selectedProblem.results.quantum_metrics.circuit_depth}</Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="textSecondary">Gate Count</Typography>
                        <Typography variant="h6">{selectedProblem.results.quantum_metrics.gate_count}</Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="textSecondary">Execution Time</Typography>
                        <Typography variant="h6">{selectedProblem.results.quantum_metrics.execution_time_seconds}s</Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" color="textSecondary">Fidelity</Typography>
                        <Typography variant="h6">{(selectedProblem.results.quantum_metrics.fidelity * 100).toFixed(1)}%</Typography>
                      </Grid>
                    </Grid>
                  </Grid>
                )}
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
              {selectedProblem.status === 'ready' && (
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={() => {
                    executeOptimization(selectedProblem._id);
                    setDetailsDialogOpen(false);
                  }}
                >
                  Execute Optimization
                </Button>
              )}
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default QuantumOptimizer;
