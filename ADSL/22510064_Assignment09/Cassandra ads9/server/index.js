const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const cassandra = require('cassandra-driver');

// Create Express app
const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Cassandra connection setup
const client = new cassandra.Client({
  contactPoints: ['127.0.0.1'], // or your Cassandra host(s)
  localDataCenter: 'datacenter1', // replace with your datacenter
  keyspace: 'companies_keyspace',
});

client.connect()
  .then(() => console.log('Connected to Cassandra'))
  .catch(err => console.error('Error connecting to Cassandra:', err));

// Get all companies
app.get('/companies', async (req, res) => {
  const query = 'SELECT * FROM companies';
  try {
    const result = await client.execute(query);
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching companies:', err);
    res.status(500).json({ error: 'Failed to fetch companies' });
  }
});

// Get a company by ID
app.get('/companies/:id', async (req, res) => {
  const query = 'SELECT * FROM companies WHERE id = ?';
  try {
    const result = await client.execute(query, [req.params.id], { prepare: true });
    if (result.rowLength === 0) return res.status(404).json({ error: 'Company not found' });
    res.json(result.rows[0]);
  } catch (err) {
    console.error('Error fetching company:', err);
    res.status(500).json({ error: 'Failed to fetch company' });
  }
});

// Add a new company
app.post('/companies', async (req, res) => {
  const { id, name, location, employee_count, founded_year, industry } = req.body;
  const query = `
    INSERT INTO companies (id, name, location, employee_count, founded_year, industry)
    VALUES (?, ?, ?, ?, ?, ?)`;
  try {
    await client.execute(query, [id, name, location, employee_count, founded_year, industry], { prepare: true });
    res.status(201).json({ message: 'Company added' });
  } catch (err) {
    console.error('Error adding company:', err);
    res.status(500).json({ error: 'Failed to add company' });
  }
});

// Update a company
app.put('/companies/:id', async (req, res) => {
  const { name, location, employee_count, founded_year, industry } = req.body;
  const query = `
    UPDATE companies SET name = ?, location = ?, employee_count = ?, founded_year = ?, industry = ?
    WHERE id = ?`;
  try {
    await client.execute(query, [name, location, employee_count, founded_year, industry, req.params.id], { prepare: true });
    res.json({ message: 'Company updated' });
  } catch (err) {
    console.error('Error updating company:', err);
    res.status(500).json({ error: 'Failed to update company' });
  }
});

// Delete a company
app.delete('/companies/:id', async (req, res) => {
  const query = 'DELETE FROM companies WHERE id = ?';
  try {
    await client.execute(query, [req.params.id], { prepare: true });
    res.json({ message: 'Company deleted' });
  } catch (err) {
    console.error('Error deleting company:', err);
    res.status(500).json({ error: 'Failed to delete company' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
