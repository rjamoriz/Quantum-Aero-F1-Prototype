# 🔗 Frontend & MongoDB Integration Guide

Complete guide for integrating synthetic aerodynamic dataset generation with the React frontend and MongoDB backend.

---

## 📋 Overview

The synthetic data generation system is now fully integrated with:
- **MongoDB** for persistent storage
- **Express.js** REST API backend
- **React** frontend with Material-UI
- **Real-time progress tracking**
- **Job queue management**

---

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │
│  (Port 3000)    │
└────────┬────────┘
         │ HTTP REST
         ▼
┌─────────────────┐
│  Express API    │
│  (Port 8000)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ MongoDB │ │ Python VLM   │
│ (27017) │ │ Generator    │
└─────────┘ └──────────────┘
```

---

## 🚀 Setup Instructions

### 1. Install Dependencies

#### Backend
```bash
cd services/backend
npm install express mongoose cors
```

#### Frontend
```bash
cd frontend
npm install @mui/material @mui/icons-material axios
```

#### Python
```bash
cd synthetic_data_generation
pip install -r requirements.txt
```

### 2. Start MongoDB

```bash
# Windows
mongod --dbpath C:\data\db

# Linux/Mac
mongod --dbpath /data/db
```

### 3. Start Backend Server

```bash
cd services/backend
node src/server.js
```

Expected output:
```
✓ MongoDB connected
🚀 Server running on port 8000
📊 Synthetic Data API: http://localhost:8000/api/synthetic-data
```

### 4. Start Frontend

```bash
cd frontend
npm start
```

Navigate to: `http://localhost:3000`

---

## 📡 API Endpoints

### GET `/api/synthetic-data/datasets`
Get all datasets with optional filters

**Query Parameters:**
- `status` - Filter by status (pending, generating, completed, failed, cancelled)
- `limit` - Number of results (default: 20)
- `skip` - Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "_id": "...",
      "name": "Test Dataset",
      "status": "generating",
      "progress": {
        "current": 45,
        "total": 100,
        "percentage": 45,
        "estimated_time_remaining": 275
      },
      "config": {
        "tier1_samples": 100,
        "workers": 4
      }
    }
  ],
  "pagination": {
    "total": 10,
    "hasMore": false
  }
}
```

### POST `/api/synthetic-data/datasets`
Create new dataset

**Request Body:**
```json
{
  "name": "My Dataset",
  "description": "Test dataset for ML training",
  "tier1_samples": 100,
  "tier2_samples": 0,
  "tier3_samples": 0,
  "workers": 4,
  "sampling_method": "lhs",
  "tags": ["test", "ml-training"]
}
```

**Response:**
```json
{
  "success": true,
  "data": { /* dataset object */ },
  "message": "Dataset created. Use /start endpoint to begin generation."
}
```

### POST `/api/synthetic-data/datasets/:id/start`
Start dataset generation

**Response:**
```json
{
  "success": true,
  "data": { /* updated dataset */ },
  "message": "Dataset generation started"
}
```

### POST `/api/synthetic-data/datasets/:id/cancel`
Cancel running generation

### DELETE `/api/synthetic-data/datasets/:id`
Delete dataset and all associated files

### GET `/api/synthetic-data/datasets/:id/samples`
Get samples from a dataset

**Query Parameters:**
- `limit` - Number of samples (default: 100)
- `skip` - Pagination offset
- `tier` - Filter by fidelity tier (0-4)

### GET `/api/synthetic-data/datasets/:id/statistics`
Get dataset statistics (CL, CD, L/D, balance)

---

## 💾 MongoDB Schema

### SyntheticDataset Collection

```javascript
{
  _id: ObjectId,
  name: String,
  description: String,
  
  config: {
    tier1_samples: Number,
    tier2_samples: Number,
    tier3_samples: Number,
    workers: Number,
    sampling_method: String  // 'lhs' or 'stratified'
  },
  
  status: String,  // 'pending', 'generating', 'completed', 'failed', 'cancelled'
  
  progress: {
    current: Number,
    total: Number,
    percentage: Number,
    estimated_time_remaining: Number,  // seconds
    samples_per_second: Number
  },
  
  statistics: {
    n_samples: Number,
    CL: { mean, std, min, max },
    CD: { mean, std, min, max },
    L_over_D: { mean, std, min, max },
    balance: { mean, std, min, max }
  },
  
  storage: {
    output_dir: String,
    scalars_file: String,
    hdf5_file: String,
    visualizations_dir: String,
    total_size_mb: Number
  },
  
  created_at: Date,
  started_at: Date,
  completed_at: Date,
  
  error: {
    message: String,
    stack: String,
    timestamp: Date
  },
  
  tags: [String]
}
```

### AeroSample Collection

```javascript
{
  _id: ObjectId,
  sample_id: String,
  dataset_id: ObjectId,  // Reference to SyntheticDataset
  timestamp: Date,
  fidelity_tier: Number,  // 0-4
  
  geometry_params: {
    main_plane_chord: Number,
    main_plane_span: Number,
    main_plane_angle_deg: Number,
    // ... 13 more parameters
  },
  
  flow_conditions: {
    V_inf: Number,
    rho: Number,
    nu: Number,
    yaw: Number,
    ground_gap: Number,
    Re: Number
  },
  
  global_outputs: {
    CL: Number,
    CD_total: Number,
    L_over_D: Number,
    downforce_front: Number,
    downforce_rear: Number,
    balance: Number
  },
  
  provenance: {
    solver: String,
    mesh_size: Number,
    runtime_seconds: Number,
    convergence_achieved: Boolean
  },
  
  field_data_path: String,  // Path to HDF5 file
  created_at: Date
}
```

---

## 🎨 Frontend Usage

### Import Component

```javascript
import SyntheticDataGenerator from './components/SyntheticDataGenerator';

function App() {
  return (
    <div>
      <SyntheticDataGenerator />
    </div>
  );
}
```

### Features

1. **Dashboard View**
   - Total datasets count
   - Active generations
   - Completed datasets
   - Total samples generated

2. **Dataset Table**
   - Name and description
   - Status with color coding
   - Real-time progress bar
   - Configuration summary
   - Action buttons (view, start, cancel, delete)

3. **Create Dialog**
   - Dataset name and description
   - Tier configuration (1, 2, 3)
   - Worker count
   - Sampling method selection
   - Estimated time calculation

4. **Details Dialog**
   - Full statistics (CL, CD, L/D, balance)
   - Storage information
   - File paths

### Real-Time Updates

The frontend polls the API every 5 seconds to update:
- Progress percentages
- ETA calculations
- Status changes
- New datasets

---

## 🔄 Workflow Example

### 1. Create Dataset via Frontend

User fills form:
- Name: "Training Dataset v1"
- Tier 1: 1000 samples
- Workers: 8

### 2. Backend Creates Record

```javascript
POST /api/synthetic-data/datasets
→ Creates MongoDB document
→ Status: 'pending'
```

### 3. Start Generation

```javascript
POST /api/synthetic-data/datasets/:id/start
→ Spawns Python process
→ Status: 'generating'
```

### 4. Python Process Runs

```bash
python batch_orchestrator.py \
  --tier1 1000 \
  --workers 8 \
  --output /data/synthetic_datasets/dataset_1234567890
```

### 5. Progress Updates

Backend parses Python stdout:
```
Processing: 45/1000
→ Updates MongoDB: progress.current = 45
→ Calculates ETA
```

### 6. Frontend Polls

Every 5 seconds:
```javascript
GET /api/synthetic-data/datasets
→ Receives updated progress
→ Updates UI
```

### 7. Completion

Python exits with code 0:
```javascript
→ Reads scalars.json
→ Computes statistics
→ Saves samples to MongoDB (AeroSample collection)
→ Status: 'completed'
```

---

## 📊 Data Flow

### Generation → Storage

```
Python VLM Solver
    ↓
scalars.json + field_data.h5
    ↓
Backend reads files
    ↓
MongoDB (AeroSample documents)
    ↓
Frontend queries via API
```

### Query → Visualization

```
Frontend requests samples
    ↓
GET /api/synthetic-data/datasets/:id/samples
    ↓
MongoDB query with filters
    ↓
JSON response
    ↓
Frontend renders charts/tables
```

---

## 🛠️ Advanced Features

### 1. Batch Operations

Delete multiple datasets:
```javascript
const deleteMultiple = async (ids) => {
  await Promise.all(
    ids.map(id => axios.delete(`${API_BASE}/datasets/${id}`))
  );
};
```

### 2. Export Dataset

Download samples as JSON:
```javascript
const exportDataset = async (datasetId) => {
  const response = await axios.get(
    `${API_BASE}/datasets/${datasetId}/samples?limit=10000`
  );
  
  const blob = new Blob([JSON.stringify(response.data.data, null, 2)], {
    type: 'application/json'
  });
  
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `dataset_${datasetId}.json`;
  link.click();
};
```

### 3. Filter Samples

Get only high L/D samples:
```javascript
const getTopSamples = async (datasetId) => {
  const samples = await AeroSample.find({
    dataset_id: datasetId,
    'global_outputs.L_over_D': { $gt: 5.0 }
  })
  .sort({ 'global_outputs.L_over_D': -1 })
  .limit(10);
  
  return samples;
};
```

### 4. Real-Time WebSocket (Future)

For live progress updates without polling:
```javascript
// Backend
const io = require('socket.io')(server);

io.on('connection', (socket) => {
  socket.on('subscribe', (datasetId) => {
    socket.join(`dataset_${datasetId}`);
  });
});

// Emit progress
io.to(`dataset_${datasetId}`).emit('progress', {
  current: 45,
  total: 100
});

// Frontend
const socket = io('http://localhost:8000');
socket.emit('subscribe', datasetId);
socket.on('progress', (data) => {
  setProgress(data);
});
```

---

## 🐛 Troubleshooting

### Backend won't start

**Error:** `MongoNetworkError: connect ECONNREFUSED`

**Solution:** Start MongoDB first
```bash
mongod --dbpath C:\data\db
```

### Python process fails

**Error:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:** Install dependencies
```bash
cd synthetic_data_generation
pip install -r requirements.txt
```

### Frontend can't connect

**Error:** `Network Error`

**Solution:** Check CORS and backend URL
```javascript
// In synthetic-data.js
const cors = require('cors');
app.use(cors());

// In SyntheticDataGenerator.jsx
const API_BASE = 'http://localhost:8000/api/synthetic-data';
```

### Progress not updating

**Issue:** Frontend shows 0% forever

**Solution:** Check Python output parsing in backend
```javascript
// In synthetic-data.js startGeneration()
process.stdout.on('data', (data) => {
  console.log('Python output:', data.toString());
  // Verify regex matches your output format
});
```

---

## 📈 Performance Tips

### 1. Database Indexing

Already configured in schema:
```javascript
SyntheticDatasetSchema.index({ status: 1, created_at: -1 });
AeroSampleSchema.index({ dataset_id: 1, fidelity_tier: 1 });
```

### 2. Pagination

Always use limit/skip:
```javascript
GET /api/synthetic-data/datasets?limit=20&skip=0
```

### 3. Selective Fields

Don't load unnecessary data:
```javascript
.select('-error.stack -field_data_path')
```

### 4. Batch Inserts

Insert samples in batches of 100:
```javascript
for (let i = 0; i < samples.length; i += 100) {
  const batch = samples.slice(i, i + 100);
  await AeroSample.insertMany(batch);
}
```

---

## ✅ Testing

### 1. Test Backend API

```bash
# Create dataset
curl -X POST http://localhost:8000/api/synthetic-data/datasets \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","tier1_samples":10,"workers":2}'

# Get datasets
curl http://localhost:8000/api/synthetic-data/datasets

# Start generation
curl -X POST http://localhost:8000/api/synthetic-data/datasets/<ID>/start
```

### 2. Test Frontend

1. Open `http://localhost:3000`
2. Click "New Dataset"
3. Fill form with small sample count (10)
4. Click "Create & Start"
5. Watch progress bar update
6. Click "View Details" when complete

### 3. Verify MongoDB

```bash
mongosh
use f1-quantum-aero
db.syntheticdatasets.find().pretty()
db.aerosamples.count()
```

---

## 🎯 Next Steps

1. **Add WebSocket support** for real-time updates
2. **Implement job queue** (Bull/Redis) for better scaling
3. **Add user authentication** for multi-user support
4. **Create visualization dashboard** for sample analysis
5. **Add export to CSV/Parquet** for ML frameworks
6. **Implement dataset versioning**
7. **Add sample filtering UI**
8. **Create comparison view** for multiple datasets

---

## 📚 Related Documentation

- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [README.md](README.md) - Full technical documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - System overview

---

**Questions?** Check the main README or open an issue.

**Ready to integrate?** Follow the setup instructions above!
