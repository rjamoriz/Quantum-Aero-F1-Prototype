/**
 * Synthetic Data Generator Component
 * Integrated with MongoDB backend for F1 aerodynamic dataset generation
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
  Alert
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Delete,
  Visibility,
  Refresh
} from '@mui/icons-material';

const API_BASE = 'http://localhost:8000/api/synthetic-data';

const SyntheticDataGenerator = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState(null);
  
  // New dataset configuration
  const [newDataset, setNewDataset] = useState({
    name: '',
    description: '',
    tier1_samples: 100,
    tier2_samples: 0,
    tier3_samples: 0,
    workers: 4,
    sampling_method: 'lhs',
    tags: []
  });

  // Fetch datasets on mount and periodically
  useEffect(() => {
    fetchDatasets();
    const interval = setInterval(fetchDatasets, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API_BASE}/datasets?limit=50`);
      setDatasets(response.data.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const createDataset = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE}/datasets`, newDataset);
      
      // Automatically start generation
      await axios.post(`${API_BASE}/datasets/${response.data.data._id}/start`);
      
      setCreateDialogOpen(false);
      fetchDatasets();
      
      // Reset form
      setNewDataset({
        name: '',
        description: '',
        tier1_samples: 100,
        tier2_samples: 0,
        tier3_samples: 0,
        workers: 4,
        sampling_method: 'lhs',
        tags: []
      });
    } catch (error) {
      console.error('Error creating dataset:', error);
      alert('Error creating dataset: ' + error.response?.data?.error);
    } finally {
      setLoading(false);
    }
  };

  const startGeneration = async (datasetId) => {
    try {
      await axios.post(`${API_BASE}/datasets/${datasetId}/start`);
      fetchDatasets();
    } catch (error) {
      console.error('Error starting generation:', error);
      alert('Error: ' + error.response?.data?.error);
    }
  };

  const cancelGeneration = async (datasetId) => {
    try {
      await axios.post(`${API_BASE}/datasets/${datasetId}/cancel`);
      fetchDatasets();
    } catch (error) {
      console.error('Error cancelling generation:', error);
    }
  };

  const deleteDataset = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }
    
    try {
      await axios.delete(`${API_BASE}/datasets/${datasetId}`);
      fetchDatasets();
    } catch (error) {
      console.error('Error deleting dataset:', error);
    }
  };

  const viewDetails = async (datasetId) => {
    try {
      const response = await axios.get(`${API_BASE}/datasets/${datasetId}`);
      setSelectedDataset(response.data.data);
      setDetailsDialogOpen(true);
    } catch (error) {
      console.error('Error fetching details:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'generating': return 'primary';
      case 'failed': return 'error';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" component="h1">
          🏎️ Synthetic Aerodynamic Dataset Generator
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchDatasets}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={() => setCreateDialogOpen(true)}
          >
            New Dataset
          </Button>
        </Box>
      </Box>

      {/* Info Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Datasets
              </Typography>
              <Typography variant="h3">
                {datasets.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Generating
              </Typography>
              <Typography variant="h3" color="primary">
                {datasets.filter(d => d.status === 'generating').length}
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
                {datasets.filter(d => d.status === 'completed').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Samples
              </Typography>
              <Typography variant="h3">
                {datasets.reduce((sum, d) => sum + (d.statistics?.n_samples || 0), 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Datasets Table */}
      <Card>
        <CardHeader title="Datasets" />
        <CardContent>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Progress</TableCell>
                  <TableCell>Samples</TableCell>
                  <TableCell>Config</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {datasets.map((dataset) => (
                  <TableRow key={dataset._id}>
                    <TableCell>
                      <Typography variant="subtitle2">{dataset.name}</Typography>
                      {dataset.description && (
                        <Typography variant="caption" color="textSecondary">
                          {dataset.description}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={dataset.status}
                        color={getStatusColor(dataset.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ width: 150 }}>
                        <LinearProgress
                          variant="determinate"
                          value={dataset.progress?.percentage || 0}
                        />
                        <Typography variant="caption">
                          {dataset.progress?.current || 0} / {dataset.progress?.total || 0}
                          {dataset.progress?.estimated_time_remaining && (
                            <> • ETA: {formatTime(dataset.progress.estimated_time_remaining)}</>
                          )}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      {dataset.statistics?.n_samples || 0}
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        T1: {dataset.config.tier1_samples}<br />
                        T2: {dataset.config.tier2_samples}<br />
                        T3: {dataset.config.tier3_samples}<br />
                        Workers: {dataset.config.workers}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        {dataset.created_at && new Date(dataset.created_at).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => viewDetails(dataset._id)}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      
                      {dataset.status === 'generating' && (
                        <Tooltip title="Cancel">
                          <IconButton
                            size="small"
                            onClick={() => cancelGeneration(dataset._id)}
                          >
                            <Stop />
                          </IconButton>
                        </Tooltip>
                      )}
                      
                      {dataset.status === 'pending' && (
                        <Tooltip title="Start">
                          <IconButton
                            size="small"
                            onClick={() => startGeneration(dataset._id)}
                          >
                            <PlayArrow />
                          </IconButton>
                        </Tooltip>
                      )}
                      
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          onClick={() => deleteDataset(dataset._id)}
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

      {/* Create Dataset Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Synthetic Dataset</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Dataset Name"
                value={newDataset.name}
                onChange={(e) => setNewDataset({ ...newDataset, name: e.target.value })}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={newDataset.description}
                onChange={(e) => setNewDataset({ ...newDataset, description: e.target.value })}
                multiline
                rows={2}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Sample Configuration
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Tier 1 Samples (VLM)"
                value={newDataset.tier1_samples}
                onChange={(e) => setNewDataset({ ...newDataset, tier1_samples: parseInt(e.target.value) })}
                helperText="Fast VLM simulations (~5s each)"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Tier 2 Samples (Transient)"
                value={newDataset.tier2_samples}
                onChange={(e) => setNewDataset({ ...newDataset, tier2_samples: parseInt(e.target.value) })}
                helperText="Unsteady scenarios (~30s each)"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Tier 3 Samples (CFD)"
                value={newDataset.tier3_samples}
                onChange={(e) => setNewDataset({ ...newDataset, tier3_samples: parseInt(e.target.value) })}
                helperText="High-fidelity CFD (~1h each)"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Parallel Workers"
                value={newDataset.workers}
                onChange={(e) => setNewDataset({ ...newDataset, workers: parseInt(e.target.value) })}
                helperText="Number of CPU cores to use"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                select
                label="Sampling Method"
                value={newDataset.sampling_method}
                onChange={(e) => setNewDataset({ ...newDataset, sampling_method: e.target.value })}
                SelectProps={{ native: true }}
              >
                <option value="lhs">Latin Hypercube Sampling</option>
                <option value="stratified">Stratified Sampling</option>
              </TextField>
            </Grid>
            
            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Estimated Time:</strong> {' '}
                  {Math.ceil((newDataset.tier1_samples * 5 + newDataset.tier2_samples * 30 + newDataset.tier3_samples * 3600) / newDataset.workers / 60)} minutes
                </Typography>
                <Typography variant="body2">
                  <strong>Total Samples:</strong> {newDataset.tier1_samples + newDataset.tier2_samples + newDataset.tier3_samples}
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={createDataset}
            disabled={!newDataset.name || loading}
          >
            Create & Start
          </Button>
        </DialogActions>
      </Dialog>

      {/* Details Dialog */}
      <Dialog
        open={detailsDialogOpen}
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedDataset && (
          <>
            <DialogTitle>{selectedDataset.name}</DialogTitle>
            <DialogContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Description
                  </Typography>
                  <Typography>{selectedDataset.description || 'No description'}</Typography>
                </Grid>
                
                {selectedDataset.statistics && selectedDataset.statistics.n_samples > 0 && (
                  <>
                    <Grid item xs={12}>
                      <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                        Statistics
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="subtitle2" color="textSecondary">CL</Typography>
                      <Typography>
                        {selectedDataset.statistics.CL?.mean?.toFixed(3)} ± {selectedDataset.statistics.CL?.std?.toFixed(3)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="subtitle2" color="textSecondary">CD</Typography>
                      <Typography>
                        {selectedDataset.statistics.CD?.mean?.toFixed(3)} ± {selectedDataset.statistics.CD?.std?.toFixed(3)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="subtitle2" color="textSecondary">L/D</Typography>
                      <Typography>
                        {selectedDataset.statistics.L_over_D?.mean?.toFixed(2)} ± {selectedDataset.statistics.L_over_D?.std?.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="subtitle2" color="textSecondary">Balance</Typography>
                      <Typography>
                        {(selectedDataset.statistics.balance?.mean * 100)?.toFixed(1)}% front
                      </Typography>
                    </Grid>
                  </>
                )}
                
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                    Storage
                  </Typography>
                  <Typography variant="body2">
                    Output Directory: {selectedDataset.storage?.output_dir}
                  </Typography>
                  {selectedDataset.storage?.total_size_mb && (
                    <Typography variant="body2">
                      Size: {selectedDataset.storage.total_size_mb.toFixed(2)} MB
                    </Typography>
                  )}
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default SyntheticDataGenerator;
